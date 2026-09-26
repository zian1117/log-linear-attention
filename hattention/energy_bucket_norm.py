"""Experimental exact GDN bucket norms from the per-write energy identity.

The supplied shared factors must use the RIGHT triangular system: beta scales
columns of its strict-lower Gram matrix. They are (gram, inverse, decay, z),
where z = inverse @ k, without beta. This differs from bucket_frobenius.py.
"""

import torch

from .bucket_frobenius import _repair_cancellation, _segment_blocks


def _right_terms(k, beta, cumulative_decay):
    length = k.shape[-2]
    gram = k @ k.transpose(-1, -2)
    eye = torch.eye(length, dtype=k.dtype, device=k.device)
    system = eye + (gram * beta.unsqueeze(-2)).tril(-1)
    inverse = torch.linalg.solve_triangular(
        system, eye.expand_as(system), upper=False, unitriangular=True
    )
    causal = torch.ones((length, length), dtype=torch.bool, device=k.device).tril()
    difference = cumulative_decay.unsqueeze(-1) - cumulative_decay.unsqueeze(-2)
    decay = difference.masked_fill(~causal, -torch.inf).exp()
    return gram, inverse, decay, inverse @ k


@torch.compile
def _local_energy(inverse, decay, beta, key_norm2, value, writes):
    written_value = value * writes.unsqueeze(-1)
    residual = (inverse * decay) @ written_value
    residual_norm2 = residual.square().sum(-1)
    coefficient = beta * (2 - beta * key_norm2)
    increments = 2 * beta * (written_value * residual).sum(-1) - coefficient * residual_norm2
    norm2 = (decay.square() @ increments.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        cross_bound = written_value.square().sum(-1).sqrt() * residual_norm2.sqrt()
        absolute_increments = 2 * beta.abs() * cross_bound + coefficient.abs() * residual_norm2
        bound = (decay.square() @ absolute_increments.unsqueeze(-1)).squeeze(-1)
    return norm2, bound


@torch.compile
def _high_energy(z, beta, key_norm2, cumulative_decay, state):
    projected = z @ state
    coefficient = beta * (2 - beta * key_norm2)
    energy = coefficient * projected.square().sum(-1)
    initial_norm2 = state.square().sum((-2, -1)).unsqueeze(-1)
    gain2 = (2 * cumulative_decay).exp().to(state.dtype)
    norm2 = gain2 * (initial_norm2 - energy.cumsum(-1))
    with torch.no_grad():
        bound = gain2 * (initial_norm2 + energy.abs().cumsum(-1))
    return norm2, bound


def local_bucket_norm2(k, v, beta, cumulative_decay, level, *, terms=None, norm_floor=0.):
    """Local squared Frobenius norms, using one value matrix multiplication."""
    if level < 0:
        raise ValueError("Bucket level must be nonnegative")
    k, v, beta, cumulative_decay = (
        x.double() for x in (k, v, beta, cumulative_decay)
    )
    if level == 0:
        return beta.square() * k.square().sum(-1) * v.square().sum(-1)
    length = k.shape[-2]
    period = 1 << level
    if period > length or length % period:
        raise ValueError("Local bucket period must divide the chunk length")
    shape = (*beta.shape[:-1], length // period, period)
    key = k.reshape(*shape, k.shape[-1])
    value = v.reshape(*shape, v.shape[-1])
    b, gc = beta.reshape(shape), cumulative_decay.reshape(shape)
    if terms is None:
        _, inverse, decay, _ = _right_terms(key, b, gc)
    else:
        inverse, decay = (_segment_blocks(x.double(), period) for x in terms[1:3])
    writes = torch.arange(period, device=k.device) < period // 2
    norm2, bound = _local_energy(inverse, decay, b, key.square().sum(-1), value, writes)
    return _repair_cancellation(norm2, bound, key, b, gc, v=value, norm_floor=norm_floor).reshape_as(beta)


def high_bucket_norm2(k, beta, cumulative_decay, state, *, terms=None, norm_floor=0.):
    """No-write bucket norms from one projected-state matrix multiplication."""
    k, beta, cumulative_decay, state = (
        x.double() for x in (k, beta, cumulative_decay, state)
    )
    if terms is None:
        _, _, _, z = _right_terms(k, beta, cumulative_decay)
    else:
        z = terms[3].double()
    norm2, bound = _high_energy(z, beta, k.square().sum(-1), cumulative_decay, state)
    return _repair_cancellation(norm2, bound, k, beta, cumulative_decay, state=state, norm_floor=norm_floor)
