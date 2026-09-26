"""Recompute selected complete Fenwick periods from their zero origins.

Inputs are already normalized chunk tensors. This helper does not normalize
them again and does not apply the final value-read scale. It returns only the
selected periods, so the caller can replace matching fast-path outputs with
``index_copy`` while preserving gradients to the original inputs.
"""

import math
from typing import NamedTuple

import torch

from .bucket_states import BucketStates
from .energy_bucket_norm import local_bucket_norm2, high_bucket_norm2
from .refined_bucket_states import RefinedStates


def _gather_tokens(tensor, heads, positions, valid):
    flat = tensor.flatten(1, 2)
    indices = positions.clamp_max(flat.shape[1] - 1)
    gathered = flat[heads[:, None], indices].double()
    mask = valid.reshape(*valid.shape, *((1,) * (gathered.ndim - valid.ndim)))
    return torch.where(mask, gathered, 0.)


@torch.compile
def _two_token_period(r, k, v, beta, gc, u, q, temperature, norm_floor):
    # One write followed by one erase is still rank one. Evaluating its
    # remaining key direction directly avoids subtractive energy cancellation.
    alignment = (k[:, 1] * k[:, 0]).sum(-1, keepdim=True)
    direction = k[:, 0] - beta[:, 1, None] * k[:, 1] * alignment
    direction = direction * (beta[:, 0] * (gc[:, 1] - gc[:, 0]).exp())[:, None]
    y = (r[:, 1] * direction).sum(-1, keepdim=True) * v[:, 0]
    read = (q[:, 1] * direction).sum(-1, keepdim=True) * v[:, 0]
    norm2 = direction.square().sum(-1) * v[:, 0].square().sum(-1)
    score = temperature[:, 0, 0] * (read * u[:, 1]).sum(-1) * norm2.clamp_min(norm_floor ** 2).rsqrt()
    return torch.stack((torch.zeros_like(y), y), dim=1), torch.stack((torch.zeros_like(score), score), dim=1)


@torch.compile
def _coarse_state_factors(k, v, beta, gc):
    """Right triangular factors without any token-query projections."""
    chunk = k.shape[-2]
    gram = k @ k.transpose(-1, -2)
    eye = torch.eye(chunk, device=k.device, dtype=k.dtype)
    system = eye + (gram * beta.unsqueeze(-2)).tril(-1)
    inverse = torch.linalg.solve_triangular(system, eye.expand_as(system), upper=False, unitriangular=True)
    causal = torch.ones((chunk, chunk), device=k.device, dtype=torch.bool).tril()
    decay = (gc.unsqueeze(-1) - gc.unsqueeze(-2)).masked_fill(~causal, -torch.inf).exp().to(k.dtype)
    z = inverse @ k
    gain = gc.exp().to(k.dtype).unsqueeze(-1)
    wn = beta.unsqueeze(-1) * z * gain
    writes = beta.unsqueeze(-1) * ((inverse * decay) @ v)
    kn = k * (gc[..., -1:] - gc).exp().to(k.dtype).unsqueeze(-1)
    end = gc[..., -1].exp().to(k.dtype).contiguous()
    return kn, wn, writes, end, z


@torch.compile
def _coarse_selected_reads(r, k, beta, gc, q, z, state):
    """Evaluate both reads only for the selected no-write chunks."""
    w = beta.unsqueeze(-1) * z
    rk = (r @ k.transpose(-1, -2)).tril()
    qk = (q @ k.transpose(-1, -2)).tril()
    gain = gc.exp().to(k.dtype).unsqueeze(-1)
    rn = gain * (r - rk @ w)
    qn = gain * (q - qk @ w)
    return rn @ state, qn @ state


class PreciseChunkCache(NamedTuple):
    """Sparse differentiable factors; lookup maps [BH,N] into factor rows.

    Row zero is the identity transition for padding. A lookup of -1 denotes
    an uncached real chunk and must never be used for a requested period.
    """
    lookup: torch.Tensor
    kn: torch.Tensor
    wn: torch.Tensor
    writes: torch.Tensor
    end: torch.Tensor
    z: torch.Tensor
    raw_k: torch.Tensor
    raw_beta: torch.Tensor
    raw_gc: torch.Tensor


def prepare_precise_chunk_cache(k, v, beta, gc, needed_chunks):
    """Prepare each needed chunk once, retaining its complete autograd graph.

    ``needed_chunks`` is Boolean [BH,N]. Factors are computed from FP64 copies
    of these same input tensors, then stored sparsely (no dense matrix-factor
    scatter). The cache may be reused by several levels in the same forward
    graph; it must not be reused with different inputs or across backward calls
    that have already freed its graph.
    """
    if k.ndim != 4 or k.shape[1] == 0:
        raise ValueError('Expected nonempty [batch_heads,chunks,chunk,key_dim] keys')
    if needed_chunks.shape != k.shape[:2] or needed_chunks.dtype != torch.bool:
        raise ValueError('Needed chunks must be Boolean [batch_heads,chunks]')
    heads, chunks = needed_chunks.nonzero(as_tuple=True)
    lookup = torch.full(k.shape[:2], -1, device=k.device, dtype=torch.long)
    lookup[heads, chunks] = torch.arange(1, heads.numel() + 1, device=k.device)
    # Share the FP64 nodes between state preparation and subsequent reads.
    # Independently promoting the same FP32 input would round their large,
    # cancelling gradients separately before they meet at that input.
    kk, vv, bb, gg = (x[heads, chunks].double() for x in (k, v, beta, gc))
    if heads.numel():
        factors = _coarse_state_factors(kk, vv, bb, gg)
    else:
        factors = (kk, kk, vv, gg.new_empty((0,)), kk)
    # Identity padding exactly matches preparing a zero key/value/beta chunk
    # with zero cumulative decay. Concatenation preserves all real gradients.
    padded = tuple(torch.cat((x.new_full((1, *x.shape[1:]), 1. if index == 3 else 0.), x), dim=0)
                   for index, x in enumerate(factors))
    raw = tuple(torch.cat((x.new_zeros((1, *x.shape[1:])), x), dim=0)
                for x in (kk, bb, gg))
    return PreciseChunkCache(lookup, *padded, *raw)


def precise_period_reads(r, k, v, beta, gc, u, q, temperature, level,
                         selected_periods, norm_floor=1e-6, output_dtype=None,
                         *, selected_chunks=None, chunk_cache=None):
    """Return ``(flat_period_ids, y, score)`` for exactly the selected periods.

    Vector inputs have shape [BH,N,C,D], beta/gc [BH,N,C], and temperature
    [BH,1,1]. ``gc`` contains FP64 cumulative log decay within each chunk.
    The Boolean selection mask is [BH,ceil(N*C/2**level)]. Returned y/score
    have shapes [selected,2**level,V] and [selected,2**level]. Inactive first
    halves and padded final tokens are zero; level zero is entirely active.
    IDs index the flattened selection mask. Coarse CUDA boundaries use the
    refined FP64-result scan; CPU uses the reference FP64 recurrence.

    For coarse levels (2**level > C), optional Boolean ``selected_chunks``
    [BH,N] restricts the returned reads and norms to requested active chunks
    within selected periods. This changes the return format to
    ``(flat_chunk_ids, y[J,C,V], score[J,C])``: IDs index flattened [BH,N],
    in ascending order. Requests in inactive first halves or unselected
    periods are ignored. Every selected period's complete differentiable
    state history is still computed; only its output evaluation is restricted.
    Optional ``chunk_cache`` from ``prepare_precise_chunk_cache`` shares chunk
    preparation across coarse selected-chunk calls. It must contain every
    real chunk in each selected period, including inactive prefix chunks.
    """
    if not isinstance(level, int) or level < 0:
        raise ValueError('Fenwick level must be a nonnegative integer')
    if k.ndim != 4 or k.shape[1] == 0:
        raise ValueError('Expected nonempty [batch_heads,chunks,chunk,key_dim] keys')
    if not (math.isfinite(norm_floor) and norm_floor > 0):
        raise ValueError('The norm floor must be finite and positive')
    batch, chunks, chunk, _ = k.shape
    if chunk < 1 or chunk & (chunk - 1):
        raise ValueError('Chunk length must be a power of two')
    period = 1 << level
    total = chunks * chunk
    groups = (total + period - 1) // period
    if selected_periods.shape != (batch, groups) or selected_periods.dtype != torch.bool:
        raise ValueError('Selection must be Boolean [batch_heads,period_groups]')
    if temperature.shape != (batch, 1, 1):
        raise ValueError('Temperature must have shape [batch_heads,1,1]')
    if selected_chunks is not None:
        if period <= chunk:
            raise ValueError('Chunk selection is only supported for coarse levels')
        if selected_chunks.shape != (batch, chunks) or selected_chunks.dtype != torch.bool:
            raise ValueError('Chunk selection must be Boolean [batch_heads,chunks]')
    if chunk_cache is not None:
        if selected_chunks is None:
            raise ValueError('Chunk caches require coarse selected-chunk evaluation')
        if chunk_cache.lookup.shape != (batch, chunks):
            raise ValueError('Chunk cache lookup must match [batch_heads,chunks]')
    output_dtype = v.dtype if output_dtype is None else output_dtype
    ids = selected_periods.reshape(-1).nonzero(as_tuple=False).flatten()
    if ids.numel() == 0:
        returned_tokens = chunk if selected_chunks is not None else period
        return (ids, v.new_empty((0, returned_tokens, v.shape[-1]), dtype=output_dtype),
                v.new_empty((0, returned_tokens), dtype=torch.float64))
    heads = torch.div(ids, groups, rounding_mode='floor')
    starts = (ids % groups) * period
    if selected_chunks is not None:
        offsets = torch.arange(period // chunk, device=k.device)
        original_chunks = starts[:, None] // chunk + offsets[None, :]
        wanted = (selected_chunks[heads[:, None], original_chunks.clamp_max(chunks - 1)]
                  & (original_chunks < chunks) & (offsets[None, :] >= period // chunk // 2))
        selected_rows, selected_offsets = wanted.nonzero(as_tuple=True)
        chunk_ids = heads[selected_rows] * chunks + original_chunks[selected_rows, selected_offsets]
        if chunk_ids.numel() == 0:
            return (chunk_ids, v.new_empty((0, chunk, v.shape[-1]), dtype=output_dtype),
                    v.new_empty((0, chunk), dtype=torch.float64))
    positions = starts[:, None] + torch.arange(period, device=k.device)[None, :]
    valid = positions < total
    raw = tuple(_gather_tokens(x, heads, positions, valid)
                for x in (r, k, v, beta, gc, u, q))
    rr, kk, vv, bb, gg, uu, qq = raw
    temp = temperature.index_select(0, heads).double()
    if level == 1 and period <= chunk:
        y, score = _two_token_period(rr, kk, vv, bb, gg, uu, qq, temp, norm_floor)
        return ids, y.to(output_dtype), score
    # Delayed import avoids a cycle if fast-path orchestration imports this
    # helper. The same RIGHT triangular preparation is used in both paths.
    from .fast_matrix_gdn import _prepare

    if period <= chunk:
        # Local periods may start inside a chunk. Reset cumulative log decay
        # using FP64 prefix differences; no previous period's state is reused.
        previous = (starts - 1).clamp_min(0)
        prefix = gc.flatten(1, 2)[heads, previous].double()
        prefix = torch.where(starts % chunk != 0, prefix, 0.)
        gg = gg - prefix[:, None]
        rr, kk, vv, bb, gg, uu, qq = (x.unsqueeze(1) for x in (rr, kk, vv, bb, gg, uu, qq))
        prepared = _prepare(rr, kk, vv, bb, gg, qq)
        ar, aq, terms = prepared[6], prepared[7], prepared[8]
        norm2 = local_bucket_norm2(kk, vv, bb, gg, level, terms=terms, norm_floor=norm_floor)
        if level == 0:
            y = ar.diagonal(dim1=-2, dim2=-1).unsqueeze(-1) * vv
            read = aq.diagonal(dim1=-2, dim2=-1).unsqueeze(-1) * vv
            score = temp * (read * uu).sum(-1) * norm2.clamp_min(norm_floor ** 2).rsqrt()
        else:
            half = period // 2
            y = ar[..., half:, :half] @ vv[..., :half, :]
            read = aq[..., half:, :half] @ vv[..., :half, :]
            score = temp * (read * uu[..., half:, :]).sum(-1) * norm2[..., half:].clamp_min(norm_floor ** 2).rsqrt()
            y = torch.cat((torch.zeros_like(y), y), dim=-2)
            score = torch.cat((torch.zeros_like(score), score), dim=-1)
        y, score = y.squeeze(1), score.squeeze(1)
    else:
        period_chunks = period // chunk
        selected = ids.numel()
        rr, kk, vv, bb, gg, uu, qq = (
            x.reshape(selected, period_chunks, chunk, *x.shape[2:])
            for x in (rr, kk, vv, bb, gg, uu, qq)
        )
        if selected_chunks is None:
            kn, wn, writes, end, rn, qn, _, _, terms = _prepare(rr, kk, vv, bb, gg, qq)
        elif chunk_cache is not None:
            cache_rows = chunk_cache.lookup[heads[:, None], original_chunks.clamp_max(chunks - 1)]
            cache_rows = torch.where(original_chunks < chunks, cache_rows, 0)
            if (cache_rows < 0).any().item():
                raise ValueError('Chunk cache is missing part of a selected period history')
            kn, wn, writes, end, z = (x[cache_rows] for x in chunk_cache[1:6])
            kk, bb, gg = (x[cache_rows] for x in
                          (chunk_cache.raw_k, chunk_cache.raw_beta, chunk_cache.raw_gc))
        else:
            kn, wn, writes, end, z = _coarse_state_factors(kk, vv, bb, gg)
        state = (RefinedStates.apply(kn, wn, writes, end, period_chunks, norm_floor)
                 if k.is_cuda else BucketStates.apply(kn, wn, writes, end, period_chunks))
        if selected_chunks is not None:
            # Gather after the full state scan. Thus gradients from a selected
            # read still reach every earlier write/erase in its own period.
            def gather(x):
                return x[selected_rows, selected_offsets]

            state = gather(state)
            active_z = gather(z)
            # high_bucket_norm2 consumes only terms[3] (the projected keys).
            active_terms = (None, None, None, active_z)
            norm2 = high_bucket_norm2(gather(kk), gather(bb), gather(gg), state,
                                     terms=active_terms, norm_floor=norm_floor)
            y, read = _coarse_selected_reads(gather(rr), gather(kk), gather(bb), gather(gg),
                                             gather(qq), active_z, state)
            score = (temp[selected_rows, 0] * (read * gather(uu)).sum(-1)
                     * norm2.clamp_min(norm_floor ** 2).rsqrt())
            return chunk_ids, y.to(output_dtype), score
        half = period_chunks // 2
        state = state[:, half:]
        active_terms = tuple(x[:, half:] for x in terms)
        norm2 = high_bucket_norm2(kk[:, half:], bb[:, half:], gg[:, half:], state,
                                 terms=active_terms, norm_floor=norm_floor)
        y = rn[:, half:] @ state
        read = qn[:, half:] @ state
        score = temp * (read * uu[:, half:]).sum(-1) * norm2.clamp_min(norm_floor ** 2).rsqrt()
        y = torch.cat((torch.zeros_like(y), y), dim=1).flatten(1, 2)
        score = torch.cat((torch.zeros_like(score), score), dim=1).flatten(1, 2)
    y = torch.where(valid.unsqueeze(-1), y, 0.).to(output_dtype)
    score = torch.where(valid, score, 0.)
    return ids, y, score
