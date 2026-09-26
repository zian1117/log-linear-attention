"""Detached decay-only upper bounds for Fenwick GDN matrix norms.

Assume 0 <= beta <= 1, ||k||_2 <= 1 and log decays g <= 0. GDN's erase
operator then has spectral norm <= 1. Ignoring erasure and adding the norms
of the rank-one writes gives a Frobenius upper bound in real arithmetic.
These FP32 sums are a conditioning indicator, not a certified floating-point
error bound. They deliberately carry no gradients.
"""
import torch
import triton
import triton.language as tl


def _write_mass(k, v, beta):
    return (beta.float() * torch.linalg.vector_norm(k.float(), dim=-1)
            * torch.linalg.vector_norm(v.float(), dim=-1))


def _diagonal_blocks(matrix, period):
    groups = matrix.shape[-1] // period
    blocks = matrix.reshape(*matrix.shape[:-2], groups, period, groups, period)
    return blocks.diagonal(dim1=-4, dim2=-2).movedim(-1, -3)


@torch.no_grad()
def local_mass(k, v, beta, gc, level, E=None):
    """Decay-only mass at each position of a local Fenwick level.

Inputs end in [chunk,key/value_dim] and scalar inputs end in [chunk]. E may
be a precomputed causal [chunk,chunk] decay matrix. Level zero is the current
write. Higher levels write in each period's first half; only the second half
is an active Fenwick bucket, but masses are returned at every position.
    """
    if not isinstance(level, int) or level < 0:
        raise ValueError('Local mass level must be a nonnegative integer.')
    mass = _write_mass(k, v, beta)
    if level == 0:
        return mass
    chunk = k.shape[-2]
    period = 1 << level
    if period > chunk or chunk % period:
        raise ValueError('Local mass period must divide the chunk length.')
    scalar_shape = (*mass.shape[:-1], chunk // period, period)
    mass = mass.reshape(scalar_shape)
    if E is None:
        cumulative = gc.reshape(scalar_shape)
        positions = torch.arange(period, device=k.device)
        causal = positions[:, None] >= positions[None, :]
        difference = cumulative.unsqueeze(-1) - cumulative.unsqueeze(-2)
        decay = difference.masked_fill(~causal, -torch.inf).exp().float()
    else:
        decay = _diagonal_blocks(E.float(), period)
    writes = torch.arange(period, device=k.device) < period // 2
    return (decay @ (mass * writes).unsqueeze(-1)).squeeze(-1).reshape_as(beta)


@torch.no_grad()
def chunk_mass_summaries(k, v, beta, gc):
    """Return chunk decay and the decay-weighted sum of its write norms."""
    cumulative = gc
    decay = cumulative[..., -1].exp().float()
    addition = (_write_mass(k, v, beta)
                * (cumulative[..., -1:] - cumulative).exp().float()).sum(-1)
    return decay, addition


@triton.jit
def _boundary_mass_fwd(DECAY, ADDITION, OUTPUT,
                       N: tl.constexpr, PERIOD: tl.constexpr):
    batch, group = tl.program_id(0), tl.program_id(1)
    mass = tl.full((), 0., tl.float32)
    for position in range(PERIOD):
        chunk = group * PERIOD + position
        if chunk < N:
            offset = batch * N + chunk
            tl.store(OUTPUT + offset, mass)
            decay = tl.load(DECAY + offset).to(tl.float32)
            addition = tl.load(ADDITION + offset).to(tl.float32)
            mass = decay * mass + tl.where(position < PERIOD // 2, addition, 0.)


@torch.no_grad()
def boundary_mass_reference(decay, addition, period):
    """CPU-capable scalar recurrence; return masses before each chunk."""
    decay, addition = decay.float(), addition.float()
    current = torch.zeros_like(decay[..., 0])
    result = []
    for chunk in range(decay.shape[-1]):
        if chunk % period == 0:
            current = torch.zeros_like(current)
        result.append(current)
        current = current * decay[..., chunk]
        if chunk % period < period // 2:
            current = current + addition[..., chunk]
    return torch.stack(result, -1)


@torch.no_grad()
def boundary_mass(decay, addition, period):
    """Fused GPU scalar scan, preserving arbitrary leading batch dimensions."""
    if not isinstance(period, int) or period < 2:
        raise ValueError('Boundary mass period must be an integer >= 2.')
    if decay.shape != addition.shape or decay.ndim < 1 or decay.shape[-1] == 0:
        raise ValueError('Mass summaries must have equal nonempty [...,chunks] shapes.')
    if not decay.is_cuda:
        return boundary_mass_reference(decay, addition, period)
    shape = decay.shape
    chunks = shape[-1]
    decay = decay.float().reshape(-1, chunks).contiguous()
    addition = addition.float().reshape(-1, chunks).contiguous()
    result = torch.empty_like(decay)
    _boundary_mass_fwd[(decay.shape[0], triton.cdiv(chunks, period))](
        decay, addition, result, chunks, period, num_warps=1,
    )
    return result.reshape(shape)


@torch.no_grad()
def high_mass(boundary, gc):
    """Decay a pre-chunk mass to each token; no new writes join this bucket."""
    return boundary.float().unsqueeze(-1) * gc.exp().float()
