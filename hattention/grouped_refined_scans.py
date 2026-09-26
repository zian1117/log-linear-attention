"""Batch independent FP32 refinement scans without changing FP64 checks.

All CUDA recurrence bodies are shared with the standalone state backend.
Residuals, exact failed-period repairs, and factor gradients reuse the existing
refined backend. CPU dispatch uses the existing exact recurrence.
"""
import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from . import refined_bucket_states as reference
from .bucket_states import BucketStates
from .fast_bucket_states import _forward_one, _correction_one, _adjoints_one

@triton.jit
def _grouped_scan(KS, WS, VS, AS, OS, OFFSETS, NS, PS, C: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BV: tl.constexpr, MODE: tl.constexpr):
    work = tl.program_id(0)
    row, tile = (work // triton.cdiv(V, BV), work % triton.cdiv(V, BV))
    for level in tl.static_range(len(NS)):
        if (row >= OFFSETS[level]) & (row < OFFSETS[level + 1]):
            groups = triton.cdiv(NS[level], PS[level])
            bh = (row - OFFSETS[level]) // groups
            seg = (row - OFFSETS[level]) % groups
            if MODE == 0:
                _forward_one(bh, seg, tile, KS[level], WS[level], VS[level], AS[level], OS[level], NS[level], C, K, V, PS[level], BV)
            elif MODE == 1:
                _correction_one(bh, seg, tile, KS[level], WS[level], AS[level], VS[level], OS[level], NS[level], C, K, V, PS[level], BV)
            else:
                _adjoints_one(bh, seg, tile, KS[level], WS[level], AS[level], VS[level], OS[level], None, NS[level], C, K, V, PS[level], BV, False)

def scans(ks, ws, values, decays, periods, mode, order='descending'):
    """Run independent forward/correction/reverse scans in one CUDA launch."""
    if not ks:
        return ()
    c, kdim = ks[0].shape[-2:]
    vdim = values[0].shape[-1]
    output = tuple((k.new_empty((*k.shape[:2], kdim, vdim)) for k in ks))
    permutation = sorted(range(len(ks)), key=lambda i: ks[i].shape[1], reverse=order == 'descending')
    ordered = lambda values: tuple((values[i] for i in permutation))
    ks, ws, values, decays, ordered_outputs, periods = map(ordered, (ks, ws, values, decays, output, periods))
    offsets = [0]
    for k, p in zip(ks, periods):
        offsets.append(offsets[-1] + k.shape[0] * triton.cdiv(k.shape[1], p))
    if offsets[-1]:
        tile = min(32, triton.next_power_of_2(vdim))
        _grouped_scan[offsets[-1] * triton.cdiv(vdim, tile),](tuple(ks), tuple(ws), tuple(values), tuple(decays), ordered_outputs, tuple((tl.constexpr(x) for x in offsets)), tuple((tl.constexpr(k.shape[1]) for k in ks)), tuple((tl.constexpr(x) for x in periods)), c, kdim, vdim, tile, mode, num_warps=4, num_stages=1)
    return output

class _GroupedRefined(torch.autograd.Function):

    @staticmethod
    def forward(ctx, periods, floor, refinements, launch_order, *flat):
        ctx.set_materialize_grads(False)
        ctx.periods, ctx.floor, ctx.refinements = (periods, floor, refinements)
        ctx.launch_order = launch_order
        ctx.dtypes = tuple((x.dtype for x in flat))
        factors = [tuple((x.double().contiguous() for x in flat[i:i + 4])) for i in range(0, len(flat), 4)]
        c, kdim = factors[0][0].shape[-2:]
        vdim = factors[0][2].shape[-1]
        ctx.original_dimensions = (c, kdim, vdim)
        cp = max(16, triton.next_power_of_2(c)) - c
        kp = max(16, triton.next_power_of_2(kdim)) - kdim
        vp = max(16, triton.next_power_of_2(vdim)) - vdim
        factors = [(F.pad(k, (0, kp, 0, cp)) if kp or cp else k, F.pad(w, (0, kp, 0, cp)) if kp or cp else w, F.pad(v, (0, vp, 0, cp)) if vp or cp else v, a) for k, w, v, a in factors]
        packed = tuple(zip(*factors))
        kf, wf, vf, af = tuple((tuple((x.float().contiguous() for x in group)) for group in packed))
        states = tuple((x.double() for x in scans(kf, wf, vf, af, periods, 0, order=ctx.launch_order)))
        for _ in range(refinements):
            defects = tuple((reference._forward_defect(*factor, state, p).contiguous() for factor, state, p in zip(factors, states, periods)))
            corrections = scans(kf, wf, defects, af, periods, 1, order=ctx.launch_order)
            states = tuple((s - d.double() for s, d in zip(states, corrections)))
        checked = []
        for factor, state, p in zip(factors, states, periods):
            if state.shape[0]:
                failed = reference.forward_failures(*factor, state, p, floor)
                state = reference._precise_forward_periods(*factor, state, p, failed)
            checked.append(state)
        ctx.save_for_backward(*(x for factor, state in zip(factors, checked) for x in (*factor, state)))
        return tuple((state[..., :kdim, :vdim] for state in checked))

    @staticmethod
    def backward(ctx, *directs):
        factors = []
        states = []
        active = []
        incoming = []
        saved_all = ctx.saved_tensors
        c, kdim, vdim = ctx.original_dimensions
        for i, direct in enumerate(directs):
            if direct is None or not direct.numel():
                continue
            saved = saved_all[5 * i:5 * i + 5]
            factor, state = (saved[:4], saved[4])
            kp, vp = (state.shape[-2] - kdim, state.shape[-1] - vdim)
            direct = direct.double()
            if kp or vp:
                direct = F.pad(direct, (0, vp, 0, kp))
            factors.append(factor)
            states.append(state)
            active.append(i)
            incoming.append(direct)
        gradients = [None] * (4 * len(ctx.periods))
        if factors:
            periods = tuple((ctx.periods[i] for i in active))
            kf = tuple((f[0].float().contiguous() for f in factors))
            wf = tuple((f[1].float().contiguous() for f in factors))
            af = tuple((f[3].float().contiguous() for f in factors))
            adjoints = tuple((x.double() for x in scans(kf, wf, tuple((d.float().contiguous() for d in incoming)), af, periods, 2, order=ctx.launch_order)))
            for _ in range(ctx.refinements):
                defects = tuple((reference._reverse_defect(f[0], f[1], f[3], d, a, p).contiguous() for f, d, a, p in zip(factors, incoming, adjoints, periods)))
                corrections = scans(kf, wf, defects, af, periods, 2, order=ctx.launch_order)
                adjoints = tuple((a - d.double() for a, d in zip(adjoints, corrections)))
            for original, factor, state, direct, adjoint, p in zip(active, factors, states, incoming, adjoints, periods):
                k, w, v, a = factor
                failed, projection = reference.reverse_failures(k, w, a, direct, adjoint, p, return_projection=True)
                previous = adjoint
                adjoint = reference._precise_reverse_periods(k, w, a, direct, adjoint, p, failed)
                if adjoint is not previous:
                    ids = failed.reshape(-1).nonzero(as_tuple=False).flatten()
                    repaired = reference._gather_periods(k, ids, p) @ reference._gather_periods(adjoint, ids, p)
                    projection = reference._replace_periods(projection, ids, repaired, p)
                gradients[4 * original:4 * original + 4] = reference._factor_gradients(k, w, v, state, adjoint, p, projected_adjoint=projection)
        result = []
        for i in range(len(ctx.periods)):
            g = gradients[4 * i:4 * i + 4]
            if g[0] is None:
                g = tuple((torch.zeros_like(x) for x in saved_all[5 * i:5 * i + 4]))
            cropped = (g[0][..., :c, :kdim], g[1][..., :c, :kdim], g[2][..., :c, :vdim], g[3])
            result.extend((x.to(t) for x, t in zip(cropped, ctx.dtypes[4 * i:4 * i + 4])))
        return (None, None, None, None, *result)

def grouped_refined_states(factors, periods, *, norm_floor=1e-06, refinements=1, launch_order='descending'):
    factors, periods = (tuple((tuple(f) for f in factors)), tuple(periods))
    if launch_order not in ('ascending', 'descending'):
        raise ValueError('Unknown launch order')
    if not factors:
        return ()
    if len(factors) != len(periods) or any((len(f) != 4 for f in factors)):
        raise ValueError('One four-tensor factor tuple is required per period')
    if refinements not in (1, 2) or not (math.isfinite(norm_floor) and norm_floor > 0):
        raise ValueError('Positive floor and one or two refinements required')
    c, k = factors[0][0].shape[-2:]
    v = factors[0][2].shape[-1]
    for (kk, ww, vv, aa), p in zip(factors, periods):
        if not isinstance(p, int) or p < 2 or kk.shape[1] == 0:
            raise ValueError('Period >=2 and nonzero history length required')
        if kk.shape != ww.shape or kk.shape[-2:] != (c, k) or vv.shape != (*kk.shape[:3], v) or (aa.shape != kk.shape[:2]):
            raise ValueError('Factor dimensions must agree; feature dimensions shared across levels')
    device = factors[0][0].device
    if any((x.device != device for f in factors for x in f)):
        raise ValueError('All grouped factors must share one device')
    if device.type != 'cuda':
        return tuple((BucketStates.apply(*f, p) for f, p in zip(factors, periods)))
    return _GroupedRefined.apply(periods, norm_floor, refinements, launch_order, *(x for f in factors for x in f))
