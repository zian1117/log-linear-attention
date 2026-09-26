"""Fused coarse routing using the state scan's precomputed P=z@H.

P is an independent autograd input here. Its backward contribution to z/H
belongs to the producing state scan and must not be repeated by this core.
"""
import torch
import triton
import triton.language as tl
from .fused_coarse_router import _finish_forward, _scalar_backward, _activation_backward


@triton.jit
def _project_forward(P, R, Q, H, U, Y, Z2, NUM, H2, QREAD,
                     C: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                     CB: tl.constexpr, BV: tl.constexpr, BK: tl.constexpr,
                     VT: tl.constexpr):
    batch, tile = tl.program_id(0), tl.program_id(1)
    c = tl.arange(0, CB)
    vv = tile * BV + tl.arange(0, BV)
    kk = tl.arange(0, BK)
    aq = tl.full((CB, BV), 0, tl.float32)
    ar = tl.full((CB, BV), 0, tl.float32)
    h2 = tl.full((), 0, tl.float32)
    for block in range(tl.cdiv(K, BK)):
        key = block * BK + kk
        h = tl.load(H + batch*K*V + key[:, None]*V + vv[None, :],
                    (key[:, None] < K) & (vv[None, :] < V), 0)
        r = tl.load(R + batch*C*K + c[:, None]*K + key[None, :],
                    (c[:, None] < C) & (key[None, :] < K), 0)
        ar = tl.dot(r, h, ar, input_precision='tf32x3')
        q = tl.load(Q + batch*C*K + c[:, None]*K + key[None, :],
                    (c[:, None] < C) & (key[None, :] < K), 0)
        aq = tl.dot(q, h, aq, input_precision='tf32x3')
        h2 += tl.sum(tl.sum(h*h, 0), 0)
    u = tl.load(U + batch*C*V + c[:, None]*V + vv[None, :],
                (c[:, None] < C) & (vv[None, :] < V), 0)
    tl.store(Y + batch*C*V + c[:, None]*V + vv[None, :], ar,
             (c[:, None] < C) & (vv[None, :] < V))
    tl.store(QREAD + batch*C*V+c[:, None]*V+vv[None, :], aq,
             (c[:, None] < C) & (vv[None, :] < V))
    az = tl.load(P + batch*C*V+c[:, None]*V+vv[None, :],
                 (c[:, None] < C) & (vv[None, :] < V), 0)
    tl.store(Z2 + (batch*VT+tile)*C + c, tl.sum(az*az, 1), c < C)
    tl.store(NUM + (batch*VT+tile)*C + c, tl.sum(aq*u, 1), c < C)
    tl.store(H2 + batch*VT+tile, h2)


@triton.jit
def _query_backward(DQ, DY, H, KP, DKCOEF, QG, RG, KG,
                    C: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                    BC: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    batch, ct, kt = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    cc, kk, vv = ct*BC+tl.arange(0, BC), kt*BK+tl.arange(0, BK), tl.arange(0, BV)
    dq = tl.full((BC, BK), 0, tl.float32)
    dr = tl.full((BC, BK), 0, tl.float32)
    for block in range(tl.cdiv(V, BV)):
        value = block*BV+vv
        h = tl.load(H+batch*K*V+kk[None, :]*V+value[:, None],
                    (kk[None, :] < K) & (value[:, None] < V), 0)
        off = batch*C*V+cc[:, None]*V+value[None, :]
        mask = (cc[:, None] < C) & (value[None, :] < V)
        aq = tl.load(DQ+off, mask, 0)
        dq = tl.dot(aq, h, dq, input_precision='tf32x3')
        ar = tl.load(DY+off, mask, 0).to(tl.float32)
        dr = tl.dot(ar, h, dr, input_precision='tf32x3')
    off = batch*C*K+cc[:, None]*K+kk[None, :]
    mask = (cc[:, None] < C) & (kk[None, :] < K)
    key = tl.load(KP+off, mask, 0)
    dkey = tl.load(DKCOEF+batch*C+cc, cc < C, 0)
    tl.store(QG+off, dq, mask)
    tl.store(RG+off, dr, mask)
    tl.store(KG+off, key*dkey[:, None], mask)


@triton.jit
def _state_backward(Q, R, DQ, DY, H, DI, DH,
                    C: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                    BC: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    batch, kt, vt = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    kk, vv, cc = kt*BK+tl.arange(0, BK), vt*BV+tl.arange(0, BV), tl.arange(0, BC)
    dh = tl.full((BK, BV), 0, tl.float32)
    for block in range(tl.cdiv(C, BC)):
        row = block*BC+cc
        qo = batch*C*K+row[None, :]*K+kk[:, None]
        qm = (row[None, :] < C) & (kk[:, None] < K)
        vo = batch*C*V+row[:, None]*V+vv[None, :]
        vm = (row[:, None] < C) & (vv[None, :] < V)
        r, dr = tl.load(R+qo, qm, 0), tl.load(DY+vo, vm, 0).to(tl.float32)
        dh = tl.dot(r, dr, dh, input_precision='tf32x3')
        q, dq = tl.load(Q+qo, qm, 0), tl.load(DQ+vo, vm, 0)
        dh = tl.dot(q, dq, dh, input_precision='tf32x3')
    off = batch*K*V+kk[:, None]*V+vv[None, :]
    mask = (kk[:, None] < K) & (vv[None, :] < V)
    h = tl.load(H+off, mask, 0)
    di = tl.load(DI+batch)
    tl.store(DH+off, dh+di*h, mask)


class _FusedProjectedCoarse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, projected, beta, k, gc, state, rn, qn, u, temperature, floor, output_dtype):
        bh, n, c, value_dim = projected.shape
        key_dim = state.shape[-2]
        tensors = tuple(x.contiguous() for x in (projected, beta, k, gc, state, rn, qn, u, temperature))
        projected, beta, k, gc, state, rn, qn, u, temperature = tensors
        matrices = bh*n
        cb, bv, bk = max(16, triton.next_power_of_2(c)), 32, 16
        vt = triton.cdiv(value_dim, bv)
        y = torch.empty((bh, n, c, value_dim), device=projected.device, dtype=output_dtype)
        z2 = torch.empty((matrices, vt, c), device=projected.device, dtype=torch.float32)
        numerator = torch.empty_like(z2)
        h2 = torch.empty((matrices, vt), device=projected.device, dtype=torch.float32)
        stats = torch.empty((matrices, 5, c), device=projected.device, dtype=torch.float32)
        scores = torch.empty((bh, n, c), device=projected.device, dtype=torch.float64)
        bound = torch.empty((bh, n, c), device=projected.device, dtype=torch.float32)
        qread = torch.empty_like(u)
        shared_temp = temperature.numel() == 1
        _project_forward[(matrices, vt)](projected, rn, qn, state, u, y, z2, numerator, h2, qread,
            c, key_dim, value_dim, cb, bv, bk, vt, num_warps=4, num_stages=1)
        _finish_forward[(matrices,)](k, beta, gc, temperature, z2, numerator, h2,
            scores, stats, bound, n, c, key_dim, cb, triton.next_power_of_2(key_dim),
            vt, triton.next_power_of_2(vt), floor*floor, shared_temp,
            num_warps=4, enable_fp_fusion=False)
        ctx.save_for_backward(*tensors, stats, qread)
        ctx.coarse_dimensions = bh, n, c, key_dim, value_dim, floor, shared_temp
        norm2 = stats[:, 0].reshape(bh, n, c)
        ctx.mark_non_differentiable(norm2, bound)
        return y, scores, norm2, bound

    @staticmethod
    def backward(ctx, dy, ds, _dnorm, _dbound):
        projected, beta, k, gc, state, rn, qn, u, temperature, stats, qread = ctx.saved_tensors
        bh, n, c, key_dim, value_dim, floor, shared_temp = ctx.coarse_dimensions
        matrices = bh*n
        dy, ds = dy.contiguous(), ds.contiguous()
        dn, dzcoef, dkcoef, dbeta = (torch.empty_like(beta) for _ in range(4))
        di = torch.empty((matrices,), device=projected.device, dtype=torch.float32)
        dgc, dt = torch.empty_like(gc), torch.empty_like(ds)
        _scalar_backward[(matrices,)](beta, gc, temperature, stats, ds,
            dn, dzcoef, dkcoef, di, dbeta, dgc, dt, n, c, triton.next_power_of_2(c),
            floor*floor, shared_temp, num_warps=4, enable_fp_fusion=False)
        dz, dq, du = (torch.empty_like(u) for _ in range(3))
        _activation_backward[(triton.cdiv(u.numel(), 256),)](
            projected, qread, u, dn, dzcoef, dz, dq, du, u.numel(), value_dim, 256,
            num_warps=4)
        qg, rg, kg = (torch.empty_like(k) for _ in range(3))
        _query_backward[(matrices, triton.cdiv(c, 64), triton.cdiv(key_dim, 64))](
            dq, dy, state, k, dkcoef, qg, rg, kg,
            c, key_dim, value_dim, 64, 64, 16, num_warps=4, num_stages=1)
        dh = torch.empty_like(state)
        _state_backward[(matrices, triton.cdiv(key_dim, 64), triton.cdiv(value_dim, 64))](
            qn, rn, dq, dy, state, di, dh, c, key_dim, value_dim,
            16, 64, 64, num_warps=4, num_stages=1)
        dtemperature = dt.sum_to_size(temperature.shape)
        return dz, dbeta, kg, dgc, dh, rg, qg, du, dtemperature, None, None


def fused_projected_coarse_core(projected, beta, k, gc, state, rn, qn, u, temperature, mass0,
                      floor, output_dtype, return_masks=False):
    """Experimental replacement for projected_coarse_router._core.

The projected input is P=z@state supplied by the state scan. Its gradient is
returned independently; the scan owns propagation through that product.

Matrix/scalar inputs use the same FP32/FP64 contract as that core. Returned
flags use its unchanged conditioning policy; the caller still owns fallback.
    """
    if not projected.is_cuda:
        raise ValueError('The fused coarse router requires CUDA tensors.')
    if any(x.dtype != torch.float32 for x in (projected, beta, k, state, rn, qn, u)):
        raise TypeError('The fused matrix core requires FP32 matrix inputs.')
    valid_temperature = (temperature.ndim == 0 or
                         (temperature.ndim == 3 and temperature.shape[1] == 1
                          and temperature.shape[2] == 1
                          and (temperature.shape[0] == 1 or temperature.shape[0] == projected.shape[0])))
    if not valid_temperature:
        raise ValueError('Temperature must be scalar or have shape [heads*batch,1,1].')
    y, scores, norm2, bound = _FusedProjectedCoarse.apply(
        projected, beta, k, gc, state, rn, qn, u, temperature, floor, output_dtype)
    # Import lazily: callers can later dispatch from the existing fast module.
    from .fast_matrix_gdn import _diagnostics
    from .decay_mass import high_mass
    bad, nonfinite = _diagnostics(norm2, bound, high_mass(mass0, gc), scores, y, floor)
    if return_masks:
        return y, scores, bad, nonfinite
    return y, scores, bad.reshape(bad.shape[0], -1).any(-1)
