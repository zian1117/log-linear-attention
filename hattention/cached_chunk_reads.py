"""Precise selected coarse reads without gathering raw period histories."""

import math

import torch

from .bucket_states import BucketStates
from .energy_bucket_norm import high_bucket_norm2
from .precise_period_reads import _coarse_selected_reads
from .refined_bucket_states import RefinedStates


def cached_chunk_reads(r, k, beta, gc, u, q, temperature, level,
                       selected_periods, selected_chunks, chunk_cache, *,
                       norm_floor=1e-6, output_dtype=None):
    """Return ``(flat_chunk_ids, y[J,C,V], score[J,C])`` for active requests.

    Inputs are already-normalized chunk vectors [BH,N,C,D], beta/gc[BH,N,C],
    temperature[BH,1,1]. ``gc`` is cumulative log decay within each chunk.
    The level must be coarse, with token period ``2**level > C``. Boolean
    selections have shapes [BH,ceil(N*C/2**level)] and [BH,N]. IDs flatten
    [BH,N] and are sorted. Inactive first-half chunks and requests outside
    selected periods are ignored. Output dtype defaults to ``r.dtype``;
    callers wanting the original value dtype should pass it explicitly.

    ``chunk_cache`` comes from ``prepare_precise_chunk_cache`` on these same
    inputs. It must contain every real chunk in every selected period. The
    complete differentiable prefix state scan consumes cached factors only;
    raw inputs are gathered solely for requested output chunks. In particular,
    gradients to unselected prefix writes flow through the cache, without
    gathering those raw values again. No extra normalization or final
    inverse-square-root key-dimension scale is applied here.
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

    factor_ids = chunk_cache.lookup[heads[:, None], valid_chunks]
    factor_ids = torch.where(in_sequence, factor_ids, 0)
    if (factor_ids < 0).any().item():
        raise ValueError('Chunk cache is missing part of a selected period history')
    kn, wn, writes, end = (x[factor_ids] for x in chunk_cache[1:5])
    # One correction is sufficient only when its measured residual passes the
    # same forward/reverse checks; rejected periods still use the FP64 scan.
    state = (RefinedStates.apply(kn, wn, writes, end, period_chunks, norm_floor, 1)
             if k.is_cuda else BucketStates.apply(kn, wn, writes, end, period_chunks))
    state = state[rows, positions]
    z = chunk_cache.z[factor_ids[rows, positions]]

    def gather(x):
        return x.flatten(0, 1).index_select(0, chunk_ids).double()

    rr, kk, bb, gg, uu, qq = (gather(x) for x in (r, k, beta, gc, u, q))
    norm2 = high_bucket_norm2(kk, bb, gg, state,
                             terms=(None, None, None, z), norm_floor=norm_floor)
    y, read = _coarse_selected_reads(rr, kk, bb, gg, qq, z, state)
    temp = temperature.index_select(0, output_heads).double()[:, 0]
    score = temp * (read * uu).sum(-1) * norm2.clamp_min(norm_floor ** 2).rsqrt()
    return chunk_ids, y.to(output_dtype), score
