"""A-posteriori checks for mixed-precision GDN boundary-state refinement.

The defect is evaluated in FP64 and propagated with the underlying GDN decay
contraction. This assumes normalized GDN keys, beta in [0, 1], and nonpositive
log decays. It is not a certificate for arbitrary affine chunk factors, their
preparation roundoff, or the complete model's loss gradient. A failing period
should use the precise recurrence once, without recursively applying this check.
"""

import math

import torch
import triton
import triton.language as tl


def _gamma(count):
    unit = torch.finfo(torch.float64).eps / 2
    return count * unit / (1 - count * unit)


def _norm(matrix):
    return matrix.norm(dim=(-2, -1))


def _norm_upper(matrix):
    # Account for reductions used to evaluate the norms in roundoff margins.
    count = matrix.shape[-2] * matrix.shape[-1]
    return _norm(matrix) * (1 + _gamma(2 * count + 4))


@torch.compile
def _forward_quantities(k, w, value, decay, state, period, norm_floor):
    position = torch.arange(k.shape[1], device=k.device)
    writes = (position % period < period // 2)[None, :, None, None]
    value = torch.where(writes, value, 0.)
    residual = value - w @ state
    transition = decay[..., None, None] * state + k.transpose(-1, -2) @ residual
    following = state.roll(-1, 1)
    defect = _norm_upper(following - transition)
    kn, wn, hn = _norm_upper(k), _norm_upper(w), _norm_upper(state)
    product = wn * hn
    residual_margin = (_gamma(k.shape[-1]) * product
                       + _gamma(1) * (_norm_upper(value) + product))
    transition_product = kn * _norm_upper(residual)
    margin = (kn * residual_margin
              + _gamma(k.shape[-2]) * transition_product
              + _gamma(3) * (decay.abs() * hn + transition_product
                             + _norm_upper(following)))
    valid = ((position + 1 < k.shape[1]) & ((position + 1) % period != 0))[None, :]
    local_error = torch.where(valid, defect + margin, 0.)
    scale = _norm(state).clamp_min(norm_floor)
    finite = torch.isfinite(state).all(dim=-1).all(dim=-1)
    # Initial states are fixed zero, rather than another approximate unknown.
    finite = finite & torch.where((position % period == 0)[None, :], (state == 0).all(dim=-1).all(dim=-1), True)
    return local_error, scale, finite


@torch.compile
def _reverse_quantities(k, w, decay, direct, adjoint, period):
    position = torch.arange(k.shape[1], device=k.device)
    intermediate = k @ adjoint
    transition = (decay[..., None, None] * adjoint
                  - w.transpose(-1, -2) @ intermediate + direct)
    previous = adjoint.roll(1, 1)
    defect = _norm_upper(previous - transition)
    kn, wn, an = _norm_upper(k), _norm_upper(w), _norm_upper(adjoint)
    product = wn * _norm_upper(intermediate)
    margin = (wn * _gamma(k.shape[-1]) * kn * an
              + _gamma(k.shape[-2]) * product
              + _gamma(4) * (decay.abs() * an + product
                             + _norm_upper(direct) + _norm_upper(previous)))
    valid = (position % period != 0)[None, :]
    local_error = torch.where(valid, defect + margin, 0.)
    relevant_direct = torch.where(valid, _norm(direct), 0.)
    finite = (torch.isfinite(adjoint).all(dim=-1).all(dim=-1)
              & torch.where(valid, torch.isfinite(direct).all(dim=-1).all(dim=-1), True))
    terminal = ((position + 1 == k.shape[1]) | ((position + 1) % period == 0))[None, :]
    finite = finite & torch.where(terminal, (adjoint == 0).all(dim=-1).all(dim=-1), True)
    return local_error, _norm(adjoint), relevant_direct, finite


@triton.jit
def _compose(a_left, b_left, a_right, b_right):
    return a_right * a_left, b_right + a_right * b_left


@triton.jit
def _group_failures(AP, EP, SP, DP, FP, OUT,
                    N: tl.constexpr, PERIOD: tl.constexpr, GROUPS: tl.constexpr,
                    BLOCK: tl.constexpr, REVERSE: tl.constexpr,
                    TOLERANCE: tl.constexpr, ROUNDING_MARGIN: tl.constexpr):
    bh, group = tl.program_id(0), tl.program_id(1)
    position = tl.arange(0, BLOCK)
    n = group * PERIOD + position
    valid = (position < PERIOD) & (n < N)
    offset = bh * N + n
    decay = tl.load(AP + offset, valid, 1.)
    local = tl.load(EP + offset, valid, 0.)
    scale = tl.load(SP + offset, valid, 0.)
    finite = tl.load(FP + offset, valid, True)
    if REVERSE:
        direct = tl.load(DP + offset, valid, 0.)
        scale = tl.maximum(scale, tl.max(direct, 0))
    _, inclusive = tl.associative_scan((decay, local), 0, _compose, reverse=REVERSE)
    if REVERSE:
        error = tl.gather(inclusive, tl.minimum(position + 1, BLOCK - 1), 0)
        error = tl.where(position + 1 < PERIOD, error, 0.)
    else:
        error = tl.gather(inclusive, tl.maximum(position - 1, 0), 0)
        error = tl.where(position > 0, error, 0.)
    error = error * ROUNDING_MARGIN
    # Explicit nonfinite checks are necessary: NaN > tolerance is false.
    nonfinite = ((finite == 0) | (tl.abs(local) == float('inf')) | (local != local)
                 | (tl.abs(scale) == float('inf')) | (scale != scale)
                 | (tl.abs(error) == float('inf')) | (error != error)
                 | (decay != decay) | (decay < 0.) | (decay > 1.))
    failed = valid & (nonfinite | (error > TOLERANCE * scale))
    tl.store(OUT + bh * GROUPS + group, tl.sum(failed.to(tl.int32), 0) > 0)


def _validate(k, period):
    if not isinstance(period, int) or period < 2:
        raise ValueError('Boundary-state period must be an integer >= 2')
    if k.ndim != 4 or k.shape[1] == 0:
        raise ValueError('Expected nonempty [batch_heads,chunks,chunk,key_dim] keys')


def _dispatch(decay, local, scale, direct, finite, period, reverse):
    batch, chunks = decay.shape
    groups = triton.cdiv(chunks, period)
    result = torch.empty((batch, groups), dtype=torch.bool, device=decay.device)
    _group_failures[(batch, groups)](
        decay.contiguous(), local.contiguous(), scale.contiguous(),
        direct.contiguous() if direct is not None else None, finite.contiguous(), result,
        chunks, period, groups, triton.next_power_of_2(period), reverse,
        torch.finfo(torch.float64).eps ** .5, 1 + _gamma(4 * period + 4),
        num_warps=4)
    return result


@torch.no_grad()
def forward_failures(k, w, value, decay, state, period, norm_floor):
    """Return [batch_heads,period_groups] masks needing a precise forward scan."""
    _validate(k, period)
    if not (math.isfinite(norm_floor) and norm_floor > 0):
        raise ValueError('The caller must provide its positive finite norm floor')
    k, w, value, decay, state = (x.double() for x in (k, w, value, decay, state))
    local, scale, finite = _forward_quantities(k, w, value, decay, state, period, norm_floor)
    return _dispatch(decay, local, scale, None, finite, period, False)


@torch.no_grad()
def reverse_failures(k, w, decay, direct, adjoint, period):
    """Check the adjoint solve for the supplied direct gradient, not the loss."""
    _validate(k, period)
    k, w, decay, direct, adjoint = (x.double() for x in (k, w, decay, direct, adjoint))
    local, scale, relevant_direct, finite = _reverse_quantities(
        k, w, decay, direct, adjoint, period)
    return _dispatch(decay, local, scale, relevant_direct, finite, period, True)
