"""Detached positive largest/remainder summaries of decay-weighted writes.

Model contract: 0 <= beta <= 1, ||k||_2 <= 1, and token log decays g <= 0.
The GDN erase operator is then contractive. Ignoring erasure, decay-weighted
write norms give an upper bound on the state Frobenius norm in real arithmetic.
`gc` is the cumulative token log decay within each computational chunk.

The remainder is accumulated directly from nonnegative contributions; it is
never formed by subtracting the largest source from a rounded total. These
FP32 conditioning indicators are not certified floating-point error bounds.
"""
import torch
import triton
import triton.language as tl


@torch.no_grad()
def positive_pair(contributions):
    """Return the largest source and positive sum of all other sources.

    Ties exclude exactly one maximum, so the other tied contributions remain
    in the remainder. Inputs must be nonnegative and nonempty on the last axis.
    """
    largest, index = contributions.max(-1)
    indices = torch.arange(contributions.shape[-1], device=contributions.device)
    rest = torch.where(indices == index.unsqueeze(-1), 0., contributions).sum(-1)
    return largest, rest


@torch.no_grad()
def chunk_pair(k, v, beta, gc):
    """Summarize each chunk's individual writes at that chunk's final token."""
    amplitude = beta.float() * k.float().norm(dim=-1) * v.float().norm(dim=-1)
    return positive_pair(amplitude * (gc[..., -1:] - gc).exp().float())


@torch.no_grad()
def boundary_pair_reference(decay, largest, rest, period):
    """CPU recurrence returning largest/remainder before every chunk."""
    decay, largest, rest = (x.float() for x in (decay, largest, rest))
    a = torch.zeros_like(decay[..., 0])
    r = torch.zeros_like(a)
    outputs_a = []
    outputs_r = []
    for index in range(decay.shape[-1]):
        if index % period == 0:
            a = torch.zeros_like(a)
            r = torch.zeros_like(r)
        outputs_a.append(a)
        outputs_r.append(r)
        a = a * decay[..., index]
        r = r * decay[..., index]
        if index % period < period // 2:
            r = r + rest[..., index] + torch.minimum(a, largest[..., index])
            a = torch.maximum(a, largest[..., index])
    return torch.stack(outputs_a, -1), torch.stack(outputs_r, -1)


@triton.jit
def _boundary_pair(DECAY, LARGEST, REST, OUT_A, OUT_R,
                   N: tl.constexpr, PERIOD: tl.constexpr):
    batch, group = tl.program_id(0), tl.program_id(1)
    a = tl.full((), 0., tl.float32)
    r = tl.full((), 0., tl.float32)
    for position in range(PERIOD):
        index = group * PERIOD + position
        if index < N:
            offset = batch * N + index
            tl.store(OUT_A + offset, a)
            tl.store(OUT_R + offset, r)
            decay = tl.load(DECAY + offset)
            a = a * decay
            r = r * decay
            if position < PERIOD // 2:
                addition = tl.load(LARGEST + offset)
                r = r + tl.load(REST + offset) + tl.minimum(a, addition)
                a = tl.maximum(a, addition)


@torch.no_grad()
def boundary_pair(decay, largest, rest, period):
    """Scan scalar summaries, resetting each period and allowing partial tails.

    Inputs share [..., chunks] shape. Decay is each chunk's exp(final gc);
    largest/rest are the positive summaries returned by ``chunk_pair``.
    Only the first floor(period/2) chunks of each period add new writes.
    """
    if not isinstance(period, int) or period < 2:
        raise ValueError('period must be an integer >=2')
    if (decay.shape != largest.shape or decay.shape != rest.shape
            or decay.ndim < 1 or decay.shape[-1] == 0):
        raise ValueError('summaries need equal nonempty [...,chunks] shapes')
    if any(x.device != decay.device for x in (largest, rest)):
        raise ValueError('summary devices must match')
    if not decay.is_cuda:
        return boundary_pair_reference(decay, largest, rest, period)
    shape = decay.shape
    n = shape[-1]
    decay, largest, rest = (
        x.float().reshape(-1, n).contiguous() for x in (decay, largest, rest)
    )
    a, r = torch.empty_like(decay), torch.empty_like(decay)
    _boundary_pair[(decay.shape[0], triton.cdiv(n, period))](
        decay, largest, rest, a, r, n, period,
        num_warps=1, enable_fp_fusion=False,
    )
    return a.reshape(shape), r.reshape(shape)
