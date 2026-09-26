"""Experimental mixed-precision iterative refinement of GDN bucket states.

Only the serial scan uses FP32. Residuals, corrections of the returned state,
and factor gradients use FP64. The reverse scan is independently refined;
backward does not differentiate through nearly cancelling approximate scans.
"""

import torch
import torch.nn.functional as F
import triton

from .fast_bucket_states import states as _scan, _adjoints, _correction
from .bucket_states import _scan_states
from .refinement_guard import forward_failures, reverse_failures


@torch.compile
def _forward_defect(k, w, value, decay, state, period):
    position = torch.arange(k.shape[1], device=k.device)
    write = (position % period < period//2)[None, :, None, None]
    residual = torch.where(write, value, 0.) - w @ state
    transition = decay[..., None, None]*state + k.transpose(-1, -2) @ residual
    valid = ((position+1 < k.shape[1]) & ((position+1) % period != 0))[None, :, None, None]
    return torch.where(valid, state.roll(-1, 1)-transition, 0.).float()


@torch.compile
def _reverse_defect(k, w, decay, direct, adjoint, period):
    position = torch.arange(k.shape[1], device=k.device)
    transition = decay[..., None, None]*adjoint - w.transpose(-1, -2) @ (k @ adjoint) + direct
    valid = (position % period != 0)[None, :, None, None]
    return torch.where(valid, adjoint.roll(1, 1)-transition, 0.).float()


@torch.compile
def _factor_gradients(k, w, value, state, adjoint, period, projected_adjoint=None):
    position = torch.arange(k.shape[1], device=k.device)
    write = (position % period < period//2)[None, :, None, None]
    residual = torch.where(write, value, 0.) - w @ state
    du = k @ adjoint if projected_adjoint is None else projected_adjoint
    dk = residual @ adjoint.transpose(-1, -2)
    dw = -(du @ state.transpose(-1, -2))
    dv = torch.where(write, du, 0.)
    da = (state*adjoint).sum((-2, -1))
    return dk, dw, dv, da


def _shape(k, value_dim, period):
    batch, chunks, chunk, key_dim = k.shape
    # These are kernel tiles, not architectural dimensions.
    tile = min(32, triton.next_power_of_2(value_dim))
    return (batch, triton.cdiv(chunks, period), triton.cdiv(value_dim, tile)), (chunks, chunk, key_dim, value_dim, period, tile)


def refined_states(k, w, value, decay, period, refinements=2):
    kf, wf, vf, af = (x.float().contiguous() for x in (k, w, value, decay))
    grid, dims = _shape(k, value.shape[-1], period)
    state = _scan(kf, wf, vf, af, period, value_tile=dims[-1]).double()
    for _ in range(refinements):
        defect = _forward_defect(k, w, value, decay, state, period).contiguous()
        correction = torch.empty_like(defect)
        _correction[grid](kf, wf, af, defect, correction, *dims, num_warps=4, num_stages=1)
        state = state - correction.double()
    return state


def refined_adjoints(k, w, decay, direct, period, refinements=2):
    kf, wf, af = (x.float().contiguous() for x in (k, w, decay))
    grid, dims = _shape(k, direct.shape[-1], period)
    adjoint = torch.empty_like(direct, dtype=torch.float32)
    _adjoints[grid](kf, wf, af, direct.float().contiguous(), adjoint, None, *dims,
                    False, num_warps=4, num_stages=1)
    adjoint = adjoint.double()
    for _ in range(refinements):
        defect = _reverse_defect(k, w, decay, direct, adjoint, period).contiguous()
        correction = torch.empty_like(defect)
        _adjoints[grid](kf, wf, af, defect, correction, None, *dims,
                        False, num_warps=4, num_stages=1)
        adjoint = adjoint - correction.double()
    return adjoint


def _gather_periods(tensor, indices, period, padding_value=0.):
    batch, chunks = tensor.shape[:2]
    groups = triton.cdiv(chunks, period)
    padding = groups * period - chunks
    if padding:
        tensor = F.pad(tensor, (0, 0) * (tensor.ndim - 2) + (0, padding), value=padding_value)
    return tensor.reshape(batch * groups, period, *tensor.shape[2:]).index_select(0, indices)


def _replace_periods(tensor, indices, replacement, period):
    batch, chunks = tensor.shape[:2]
    groups = triton.cdiv(chunks, period)
    padding = groups * period - chunks
    padded = F.pad(tensor, (0, 0) * (tensor.ndim - 2) + (0, padding)) if padding else tensor
    grouped = padded.reshape(batch * groups, period, *tensor.shape[2:])
    return grouped.index_copy(0, indices, replacement).reshape(batch, groups * period, *tensor.shape[2:])[:, :chunks].contiguous()


def _precise_forward_periods(k, w, value, decay, state, period, failed):
    indices = failed.reshape(-1).nonzero(as_tuple=False).flatten()
    if indices.numel() == 0:
        return state
    kk, ww, vv = (_gather_periods(x, indices, period).unsqueeze(1) for x in (k, w, value))
    aa = _gather_periods(decay, indices, period, padding_value=1.).unsqueeze(1)
    replacement = _scan_states(kk, ww, vv, aa, period).squeeze(1)
    return _replace_periods(state, indices, replacement, period)


def _precise_reverse_periods(k, w, decay, direct, adjoint, period, failed):
    indices = failed.reshape(-1).nonzero(as_tuple=False).flatten()
    if indices.numel() == 0:
        return adjoint
    kk, ww, dd = (_gather_periods(x, indices, period) for x in (k, w, direct))
    aa = _gather_periods(decay, indices, period, padding_value=1.)
    current = torch.zeros_like(dd[:, 0])
    result = [None] * period
    for position in range(period - 1, -1, -1):
        result[position] = current
        current = (aa[:, position, None, None] * current
                   - ww[:, position].transpose(-1, -2) @ (kk[:, position] @ current)
                   + dd[:, position])
    return _replace_periods(adjoint, indices, torch.stack(result, dim=1), period)


class RefinedStates(torch.autograd.Function):
    """FP64-result scan; supplying norm_floor guards both recurrence solves.

    The optional guard assumes genuine contractive GDN chunk factors. Failed
    independent periods are recomputed once with the FP64 recurrence. Omitting
    norm_floor retains the unguarded experimental backend for comparisons.
    """
    @staticmethod
    def forward(ctx, k, w, value, decay, period, norm_floor=None, refinements=2):
        if refinements not in (1, 2) or (refinements == 1 and norm_floor is None):
            raise ValueError('One correction requires residual guards; otherwise use two corrections.')
        ctx.input_count = len(ctx.needs_input_grad)
        ctx.refinements = refinements
        ctx.input_dtypes = tuple(x.dtype for x in (k, w, value, decay))
        k, w, value, decay = (x.double().contiguous() for x in (k, w, value, decay))
        chunk, key_dim, value_dim = k.shape[-2], k.shape[-1], value.shape[-1]
        cp = max(16, triton.next_power_of_2(chunk))-chunk
        kp = max(16, triton.next_power_of_2(key_dim))-key_dim
        vp = max(16, triton.next_power_of_2(value_dim))-value_dim
        ctx.dimensions = chunk, key_dim, value_dim
        if kp or cp:
            k, w = (F.pad(x, (0, kp, 0, cp)) for x in (k, w))
        if vp or cp:
            value = F.pad(value, (0, vp, 0, cp))
        state = refined_states(k, w, value, decay, period, refinements=refinements)
        if norm_floor is not None:
            failed = forward_failures(k, w, value, decay, state, period, norm_floor)
            state = _precise_forward_periods(k, w, value, decay, state, period, failed)
        ctx.save_for_backward(k, w, value, decay, state)
        ctx.period = period
        ctx.guarded = norm_floor is not None
        return state[..., :key_dim, :value_dim]

    @staticmethod
    def backward(ctx, direct):
        k, w, value, decay, state = ctx.saved_tensors
        chunk, key_dim, value_dim = ctx.dimensions
        direct = direct.double()
        vp, kp = state.shape[-1] - value_dim, state.shape[-2] - key_dim
        if vp or kp:
            direct = F.pad(direct, (0, vp, 0, kp))
        adjoint = refined_adjoints(k, w, decay, direct, ctx.period, refinements=ctx.refinements)
        projected_adjoint = None
        if ctx.guarded:
            failed, projected_adjoint = reverse_failures(
                k, w, decay, direct, adjoint, ctx.period, return_projection=True)
            previous_adjoint = adjoint
            adjoint = _precise_reverse_periods(k, w, decay, direct, adjoint, ctx.period, failed)
            if adjoint is not previous_adjoint:
                # The cached product used the rejected adjoint. Replace every
                # token of each repaired period, including nonfinite products.
                indices = failed.reshape(-1).nonzero(as_tuple=False).flatten()
                repaired = (_gather_periods(k, indices, ctx.period)
                            @ _gather_periods(adjoint, indices, ctx.period))
                projected_adjoint = _replace_periods(projected_adjoint, indices,
                                                     repaired, ctx.period)
        gradients = _factor_gradients(k, w, value, state, adjoint, ctx.period,
                                      projected_adjoint=projected_adjoint)
        gradients = (gradients[0][..., :chunk, :key_dim], gradients[1][..., :chunk, :key_dim],
                     gradients[2][..., :chunk, :value_dim], gradients[3])
        return tuple(x.to(dtype) for x, dtype in zip(gradients, ctx.input_dtypes)) + (None,) * (ctx.input_count - 4)
