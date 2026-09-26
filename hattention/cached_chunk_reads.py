"""Precise selected coarse reads with shared differentiable input caches."""

import math
from typing import NamedTuple

import torch

from .bucket_states import BucketStates
from .energy_bucket_norm import high_bucket_norm2
from .precise_period_reads import _coarse_selected_reads
from .refined_bucket_states import RefinedStates


class PreciseReadInputCache(NamedTuple):
    # No padding row: only real output chunks are requested. Missing rows are
    # rejected before indexing, including Python's otherwise valid -1 index.
    lookup: torch.Tensor
    r: torch.Tensor
    k: torch.Tensor
    beta: torch.Tensor
    gc: torch.Tensor
    u: torch.Tensor
    q: torch.Tensor


def prepare_precise_read_input_cache(r, k, beta, gc, u, q, needed_chunks, *, chunk_cache):
    """Gather the union of raw read-input chunks once in the current graph.

    Inputs are the same already-normalized tensors supplied to
    cached_chunk_reads. This cache is local to one forward/autograd graph,
    just like the existing differentiable chunk-factor cache. No histories
    or period-dependent matrices are cached here.
    """
    if needed_chunks.shape != k.shape[:2] or needed_chunks.dtype != torch.bool:
        raise ValueError('Read-input selection must be Boolean [batch_heads,chunks]')
    ids = needed_chunks.flatten().nonzero(as_tuple=False).flatten()
    lookup = torch.full(k.shape[:2], -1, device=k.device, dtype=torch.long)
    lookup.flatten()[ids] = torch.arange(ids.numel(), device=k.device)
    factor_rows = chunk_cache.lookup.flatten().index_select(0, ids)
    if (factor_rows < 0).any().item():
        raise ValueError('Factor cache is missing a requested read chunk')
    rr, uu, qq = (x.flatten(0, 1).index_select(0, ids).double() for x in (r, u, q))
    kk, bb, gg = (x.index_select(0, factor_rows)
                  for x in (chunk_cache.raw_k, chunk_cache.raw_beta, chunk_cache.raw_gc))
    return PreciseReadInputCache(lookup, rr, kk, bb, gg, uu, qq)


def cached_chunk_reads(r, k, beta, gc, u, q, temperature, level,
                       selected_periods, selected_chunks, chunk_cache, *,
                       norm_floor=1e-6, output_dtype=None, read_input_cache=None,
                       history_chunks=None):
    """Return ``(flat_chunk_ids, y[J,C,V], score[J,C])`` for active requests.

    Inputs are already-normalized chunk vectors [BH,N,C,D], beta/gc[BH,N,C],
    temperature[BH,1,1]. ``gc`` is cumulative log decay within each chunk.
    The level must be coarse, with token period ``2**level > C``. Boolean
    selections have shapes [BH,ceil(N*C/2**level)] and [BH,N]. IDs flatten
    [BH,N] and are sorted. Inactive first-half chunks and requests outside
    selected periods are ignored. Output dtype defaults to ``r.dtype``;
    callers wanting the original value dtype should pass it explicitly.

    ``chunk_cache`` comes from ``prepare_precise_chunk_cache`` on these same
    inputs. It must contain every real chunk in the requested period prefixes. The
    complete differentiable prefix state scan consumes cached factors only;
    raw inputs are gathered solely for requested output chunks. In particular,
    gradients to unselected prefix writes flow through the cache, without
    gathering those raw values again. No extra normalization or final
    inverse-square-root key-dimension scale is applied here.

    ``history_chunks`` optionally omits the tail after all requested reads.
    It counts computational chunks, and never changes the original period or
    its write half. The caller may quantize prefix lengths to limit compilation
    variants; every requested read must lie inside the supplied prefix.
    """
    if not isinstance(level, int) or level < 0:
        raise ValueError('Fenwick level must be a nonnegative integer')
    if k.ndim != 4 or k.shape[1] == 0:
        raise ValueError('Expected nonempty [batch_heads,chunks,chunk,key_dim] keys')
    batch, chunks, chunk, _ = k.shape
    if chunk < 1 or chunk & (chunk - 1):
        raise ValueError('Chunk length must be a power of two')
    if not (math.isfinite(norm_floor) and norm_floor > 0):
        raise ValueError('The norm floor must be finite and positive')
    period = 1 << level
    if period <= chunk:
        raise ValueError('Cached chunk reads require a coarse level')
    period_chunks = period // chunk
    if history_chunks is None:
        history_chunks = period_chunks
    if (not isinstance(history_chunks, int) or isinstance(history_chunks, bool)
            or not 1 <= history_chunks <= period_chunks):
        raise ValueError('History prefix must contain between one chunk and a complete period')
    groups = (chunks + period_chunks - 1) // period_chunks
    if selected_periods.shape != (batch, groups) or selected_periods.dtype != torch.bool:
        raise ValueError('Period selection must be Boolean [batch_heads,period_groups]')
    if selected_chunks.shape != (batch, chunks) or selected_chunks.dtype != torch.bool:
        raise ValueError('Chunk selection must be Boolean [batch_heads,chunks]')
    if chunk_cache.lookup.shape != (batch, chunks):
        raise ValueError('Chunk cache lookup must match [batch_heads,chunks]')
    if temperature.shape != (batch, 1, 1):
        raise ValueError('Temperature must have shape [batch_heads,1,1]')
    output_dtype = r.dtype if output_dtype is None else output_dtype

    def empty(ids):
        value_dim = chunk_cache.writes.shape[-1]
        return (ids, r.new_empty((0, chunk, value_dim), dtype=output_dtype),
                r.new_empty((0, chunk), dtype=torch.float64))

    ids = selected_periods.flatten().nonzero(as_tuple=False).flatten()
    if ids.numel() == 0:
        return empty(ids)
    heads = torch.div(ids, groups, rounding_mode='floor')
    offsets = torch.arange(period_chunks, device=k.device)
    original_chunks = (ids % groups)[:, None] * period_chunks + offsets[None, :]
    in_sequence = original_chunks < chunks
    valid_chunks = original_chunks.clamp_max(chunks - 1)
    requested = (selected_chunks[heads[:, None], valid_chunks] & in_sequence
                 & (offsets[None, :] >= period_chunks // 2))
    rows, positions = requested.nonzero(as_tuple=True)
    output_heads = heads.index_select(0, rows)
    chunk_ids = output_heads * chunks + original_chunks[rows, positions]
    if chunk_ids.numel() == 0:
        return empty(chunk_ids)

    factor_ids = chunk_cache.lookup[heads[:, None], valid_chunks[:, :history_chunks]]
    factor_ids = torch.where(in_sequence[:, :history_chunks], factor_ids, 0)
    if ((factor_ids < 0).any() | (positions >= history_chunks).any()).item():
        raise ValueError('Chunk cache is missing required history or its prefix omits a requested read')
    kn, wn, writes, end = (x[factor_ids] for x in chunk_cache[1:5])
    # One correction is sufficient only when its measured residual passes the
    # same forward/reverse checks; rejected periods still use the FP64 scan.
    state = (RefinedStates.apply(kn, wn, writes, end, period_chunks, norm_floor, 1)
             if k.is_cuda else BucketStates.apply(kn, wn, writes, end, period_chunks))
    state = state[rows, positions]
    z = chunk_cache.z[factor_ids[rows, positions]]

    def gather(x):
        return x.flatten(0, 1).index_select(0, chunk_ids).double()

    if read_input_cache is None:
        rr, uu, qq = (gather(x) for x in (r, u, q))
        read_rows = factor_ids[rows, positions]
        kk, bb, gg = (x.index_select(0, read_rows)
                      for x in (chunk_cache.raw_k, chunk_cache.raw_beta, chunk_cache.raw_gc))
    else:
        if read_input_cache.lookup.shape != (batch, chunks):
            raise ValueError('Read-input lookup must match [batch_heads,chunks]')
        read_ids = read_input_cache.lookup.flatten().index_select(0, chunk_ids)
        if (read_ids < 0).any().item():
            raise ValueError('Read-input cache is missing a requested chunk')
        rr, kk, bb, gg, uu, qq = (x.index_select(0, read_ids) for x in read_input_cache[1:])
    norm2 = high_bucket_norm2(kk, bb, gg, state,
                             terms=(None, None, None, z), norm_floor=norm_floor)
    y, read = _coarse_selected_reads(rr, kk, bb, gg, qq, z, state)
    temp = temperature.index_select(0, output_heads).double()[:, 0]
    score = temp * (read * uu).sum(-1) * norm2.clamp_min(norm_floor ** 2).rsqrt()
    return chunk_ids, y.to(output_dtype), score
