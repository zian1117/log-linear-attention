"""Share exact local Fenwick residual projections across hierarchy levels.

For a token in a bucket's erase half, D is the RIGHT triangular inverse times
the bucket's written values (including decay). Lower active Fenwick buckets
partition all prior writes within a larger period. Therefore its write-half
residual is exactly v plus the lower levels' active D vectors.

Only active half-by-half matrix products are required. The returned energy
expansions still need the caller's existing cancellation diagnostics/repair;
this module does not change that numerical contract.
"""

import torch

from .bucket_frobenius import _segment_blocks
from .energy_bucket_norm import _right_terms


@torch.compile(fullgraph=True)
def _shared_core(k, v, beta, inverse, decay, levels, active_residuals):
    chunk = k.shape[-2]
    key_norm2 = k.square().sum(-1)
    value_norm2 = v.square().sum(-1)
    current_norm2 = beta.square() * key_norm2 * value_norm2
    results = [(current_norm2, current_norm2.detach())]
    # At the beginning of level l, prefix = v + sum_{s<l} active_D_s.
    prefix = v
    for level in range(1, levels):
        period = 1 << level
        half = period // 2
        groups = chunk // period
        vector_shape = (*v.shape[:-2], groups, period, v.shape[-1])
        scalar_shape = (*beta.shape[:-1], groups, period)
        value = v.reshape(vector_shape)
        preceding = prefix.reshape(vector_shape)
        # Materialize block layouts so dynamic batch sizes do not expose
        # symbolic diagonal strides to Inductor's CUDA matrix codegen.
        inv_block = _segment_blocks(inverse, period).contiguous()
        decay_block = _segment_blocks(decay, period).contiguous()
        # The full period-by-period norm product would redundantly reconstruct
        # the first-half residuals already represented by lower levels.
        if active_residuals is None:
            active = ((inv_block[..., half:, :half] * decay_block[..., half:, :half])
                      @ value[..., :half, :])
        else:
            active = active_residuals[level - 1]
        residual = torch.cat((preceding[..., :half, :], active), dim=-2)
        residual_norm2 = residual.square().sum(-1)
        b = beta.reshape(scalar_shape)
        coefficient = b * (2 - b * key_norm2.reshape(scalar_shape))
        cross = (value[..., :half, :] * preceding[..., :half, :]).sum(-1)
        cross = torch.cat((cross, torch.zeros_like(cross)), dim=-1)
        increments = 2 * b * cross - coefficient * residual_norm2
        norm2 = (decay_block.square() @ increments.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            left_cross_bound = (value_norm2.reshape(scalar_shape)[..., :half].sqrt()
                                * residual_norm2[..., :half].sqrt())
            cross_bound = torch.cat((left_cross_bound, torch.zeros_like(left_cross_bound)), dim=-1)
            absolute_increments = 2 * b.abs() * cross_bound + coefficient.abs() * residual_norm2
            bound = (decay_block.square() @ absolute_increments.unsqueeze(-1)).squeeze(-1)
        results.append((norm2.reshape_as(beta), bound.reshape_as(beta)))
        if level + 1 < levels:
            active = torch.cat((torch.zeros_like(active), active), dim=-2).reshape_as(v)
            prefix = prefix + active
    return tuple(results)


def shared_local_norms(k, v, beta, cumulative_decay, *, terms=None, max_levels=None,
                       active_residuals=None):
    """Return ``((norm2, bound), ...)`` in increasing local Fenwick level.

    Inputs end in [C,K/V] and [C], where C is a power-of-two chunk length.
    Supplied terms must be RIGHT-system ``(gram, inverse, decay, z)`` factors.
    Arithmetic follows the key tensor's dtype; no normalization or precision
    conversion is added. Each output has the same shape as beta. Padded chunks
    are supported by zero keys/values/beta and their supplied cumulative decay.
    ``max_levels`` includes level zero and defaults to every local level.
    Optional ``active_residuals`` contains levels 1 onward, each shaped
    [...,C/period,period/2,V], allowing a fused read kernel to supply D once.
    """
    chunk = k.shape[-2]
    if chunk < 1 or chunk & (chunk - 1):
        raise ValueError('Chunk length must be a power of two')
    available = chunk.bit_length()
    levels = available if max_levels is None else max_levels
    if not isinstance(levels, int) or not 1 <= levels <= available:
        raise ValueError('max_levels must select between one and all local levels')
    if active_residuals is not None and len(active_residuals) != levels - 1:
        raise ValueError('active_residuals must contain one tensor per nonzero local level')
    if terms is None:
        _, inverse, decay, _ = _right_terms(k, beta, cumulative_decay)
        decay = decay.to(k.dtype)
    else:
        inverse, decay = terms[1:3]
    return _shared_core(k, v, beta, inverse, decay, levels,
                        None if active_residuals is None else tuple(active_residuals))
