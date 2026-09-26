"""Experimental fused FP32 coarse-bucket routing with analytical backward.

The matrix contractions use TF32x3 (three products, FP32 accuracy), not TF32.
Projected zH/qH values and scalar statistics are saved to avoid repeating
large matrix products in backward. Conditioning flags retain the experimental router's contract.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _project_forward(Z, R, Q, H, U, Y, Z2, NUM, H2, ZREAD, QREAD,
                     C: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                     CB: tl.constexpr, BV: tl.constexpr, BK: tl.constexpr,
                     VT: tl.constexpr):
    batch, tile = tl.program_id(0), tl.program_id(1)
    c = tl.arange(0, CB)
    vv = tile * BV + tl.arange(0, BV)
    kk = tl.arange(0, BK)
    az = tl.full((CB, BV), 0, tl.float32)
    aq = tl.full((CB, BV), 0, tl.float32)
    ar = tl.full((CB, BV), 0, tl.float32)
    h2 = tl.full((), 0, tl.float32)
    for block in range(tl.cdiv(K, BK)):
        key = block * BK + kk
        h = tl.load(H + batch*K*V + key[:, None]*V + vv[None, :],
                    (key[:, None] < K) & (vv[None, :] < V), 0)
        z = tl.load(Z + batch*C*K + c[:, None]*K + key[None, :],
                    (c[:, None] < C) & (key[None, :] < K), 0)
        az = tl.dot(z, h, az, input_precision='tf32x3')
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
    tl.store(ZREAD + batch*C*V+c[:, None]*V+vv[None, :], az,
             (c[:, None] < C) & (vv[None, :] < V))
    tl.store(QREAD + batch*C*V+c[:, None]*V+vv[None, :], aq,
             (c[:, None] < C) & (vv[None, :] < V))
    tl.store(Z2 + (batch*VT+tile)*C + c, tl.sum(az*az, 1), c < C)
    tl.store(NUM + (batch*VT+tile)*C + c, tl.sum(aq*u, 1), c < C)
    tl.store(H2 + batch*VT+tile, h2)


@triton.jit
def _finish_forward(KP, BETA, GC, TEMPERATURE, Z2, NUM, H2,
                    SCORE, STATS, BOUND,
                    N: tl.constexpr, C: tl.constexpr, K: tl.constexpr,
                    CB: tl.constexpr, KB: tl.constexpr, VT: tl.constexpr,
                    VTB: tl.constexpr, FLOOR2: tl.constexpr, SHARED_TEMP: tl.constexpr,
                     RAW_HALF: tl.constexpr=0, HEAD_GROUPS: tl.constexpr=1):
    batch = tl.program_id(0)
    raw_delta = 0
    if RAW_HALF:
        raw_delta = (batch // RAW_HALF) * RAW_HALF
    c, kk, tile = tl.arange(0, CB), tl.arange(0, KB), tl.arange(0, VTB)
    zp = tl.load(Z2 + (batch*VT+tile[:, None])*C+c[None, :],
                 (tile[:, None] < VT) & (c[None, :] < C), 0)
    np = tl.load(NUM + (batch*VT+tile[:, None])*C+c[None, :],
                 (tile[:, None] < VT) & (c[None, :] < C), 0)
    z2, numerator = tl.sum(zp, 0), tl.sum(np, 0)
    initial = tl.sum(tl.load(H2+batch*VT+tile, tile < VT, 0), 0)
    key = tl.load(KP+batch*C*K+c[:, None]*K+kk[None, :]+raw_delta*(C*K),
                  (c[:, None] < C) & (kk[None, :] < K), 0)
    key2 = tl.sum(key*key, 1)
    beta = tl.load(BETA+batch*C+c+raw_delta*C, c < C, 0)
    gc = tl.load(GC+batch*C+c+raw_delta*C, c < C, 0).to(tl.float64)
    coefficient = beta*(2.-beta*key2)
    energy = coefficient*z2
    raw = initial-tl.cumsum(energy, 0)
    gain = libdevice.exp(2.*gc).to(tl.float32)
    norm2 = gain*raw
    bound = gain*(initial+tl.cumsum(tl.abs(energy), 0))
    inverse = tl.rsqrt(tl.maximum(norm2, FLOOR2))
    ti = 0 if SHARED_TEMP else (batch//N)//HEAD_GROUPS
    temperature = tl.load(TEMPERATURE+ti).to(tl.float64)
    score = (temperature*numerator.to(tl.float64))*inverse.to(tl.float64)
    tl.store(SCORE+batch*C+c, score, c < C)
    tl.store(BOUND+batch*C+c, bound, c < C)
    # Scalar planes: norm², unscaled norm², numerator, ||zH||², ||k||².
    tl.store(STATS+(batch*5+0)*C+c, norm2, c < C)
    tl.store(STATS+(batch*5+1)*C+c, raw, c < C)
    tl.store(STATS+(batch*5+2)*C+c, numerator, c < C)
    tl.store(STATS+(batch*5+3)*C+c, z2, c < C)
    tl.store(STATS+(batch*5+4)*C+c, key2, c < C)


@triton.jit
def _scalar_backward(BETA, GC, TEMPERATURE, STATS, DS,
                     DN, DZCOEF, DKCOEF, DI, DBETA, DGC, DT,
                     N: tl.constexpr, C: tl.constexpr, CB: tl.constexpr,
                     FLOOR2: tl.constexpr, SHARED_TEMP: tl.constexpr,
                     RAW_HALF: tl.constexpr=0, HEAD_GROUPS: tl.constexpr=1):
    batch = tl.program_id(0)
    raw_delta = 0
    if RAW_HALF:
        raw_delta = (batch // RAW_HALF) * RAW_HALF
    c = tl.arange(0, CB)
    norm2 = tl.load(STATS+(batch*5+0)*C+c, c < C, 0)
    raw = tl.load(STATS+(batch*5+1)*C+c, c < C, 0)
    numerator = tl.load(STATS+(batch*5+2)*C+c, c < C, 0)
    z2 = tl.load(STATS+(batch*5+3)*C+c, c < C, 0)
    key2 = tl.load(STATS+(batch*5+4)*C+c, c < C, 0)
    beta = tl.load(BETA+batch*C+c+raw_delta*C, c < C, 0)
    gc = tl.load(GC+batch*C+c+raw_delta*C, c < C, 0).to(tl.float64)
    ds = tl.load(DS+batch*C+c, c < C, 0).to(tl.float64)
    ti = 0 if SHARED_TEMP else (batch//N)//HEAD_GROUPS
    temperature = tl.load(TEMPERATURE+ti).to(tl.float64)
    inverse = tl.rsqrt(tl.maximum(norm2, FLOOR2))
    intermediate = ds*inverse.to(tl.float64)
    dn = (intermediate*temperature).to(tl.float32)
    df = (ds*(temperature*numerator.to(tl.float64))).to(tl.float32)
    dnorm = (-.5*df)*(inverse*inverse*inverse)
    # torch.clamp_min differentiates the input at equality.
    dnorm = tl.where((norm2 >= FLOOR2) & (c < C), dnorm, 0.)
    gain64 = libdevice.exp(2.*gc)
    gain = gain64.to(tl.float32)
    draw = dnorm*gain
    de = -tl.cumsum(draw, 0, reverse=True)
    da = de*z2
    coefficient = beta*(2.-beta*key2)
    dzcoef = (2.*de)*coefficient
    dkcoef = (-2.*da)*(beta*beta)
    dbeta = da*(2.-beta*key2) - (da*beta)*key2
    dgc = (dnorm*raw).to(tl.float64)*gain64*2.
    dt = intermediate*numerator.to(tl.float64)
    tl.store(DN+batch*C+c, dn, c < C)
    tl.store(DZCOEF+batch*C+c, dzcoef, c < C)
    tl.store(DKCOEF+batch*C+c, dkcoef, c < C)
    tl.store(DI+batch, 2.*tl.sum(draw, 0))
    tl.store(DBETA+batch*C+c, dbeta, c < C)
    tl.store(DGC+batch*C+c, dgc, c < C)
    tl.store(DT+batch*C+c, dt, c < C)


@triton.jit
def _project_backward(Z, Q, H, U, DN, DZCOEF, DZ, DQ, DU,
                      C: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                      CB: tl.constexpr, BV: tl.constexpr, BK: tl.constexpr):
    batch, tile = tl.program_id(0), tl.program_id(1)
    c, kk = tl.arange(0, CB), tl.arange(0, BK)
    vv = tile*BV + tl.arange(0, BV)
    az = tl.full((CB, BV), 0, tl.float32)
    aq = tl.full((CB, BV), 0, tl.float32)
    for block in range(tl.cdiv(K, BK)):
        key = block*BK+kk
        h = tl.load(H+batch*K*V+key[:, None]*V+vv[None, :],
                    (key[:, None] < K) & (vv[None, :] < V), 0)
        z = tl.load(Z+batch*C*K+c[:, None]*K+key[None, :],
                    (c[:, None] < C) & (key[None, :] < K), 0)
        az = tl.dot(z, h, az, input_precision='tf32x3')
        q = tl.load(Q+batch*C*K+c[:, None]*K+key[None, :],
                    (c[:, None] < C) & (key[None, :] < K), 0)
        aq = tl.dot(q, h, aq, input_precision='tf32x3')
    mask = (c[:, None] < C) & (vv[None, :] < V)
    offset = batch*C*V+c[:, None]*V+vv[None, :]
    u = tl.load(U+offset, mask, 0)
    dn = tl.load(DN+batch*C+c, c < C, 0)
    dz = tl.load(DZCOEF+batch*C+c, c < C, 0)
    tl.store(DZ+offset, dz[:, None]*az, mask)
    tl.store(DQ+offset, dn[:, None]*u, mask)
    tl.store(DU+offset, dn[:, None]*aq, mask)


@triton.jit
def _activation_backward(ZREAD, QREAD, U, DN, DZCOEF, DZ, DQ, DU,
                         SIZE: tl.constexpr, V: tl.constexpr, BLOCK: tl.constexpr,
                         RAW_HALF: tl.constexpr=0, RAW_C: tl.constexpr=1):
    offset = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    raw_delta = 0
    if RAW_HALF:
        raw_delta = (offset // (RAW_C*V*RAW_HALF)) * (RAW_C*V*RAW_HALF)
    mask = offset < SIZE
    row = offset//V
    dn = tl.load(DN+row, mask, 0)
    coefficient = tl.load(DZCOEF+row, mask, 0)
    z = tl.load(ZREAD+offset, mask, 0)
    q = tl.load(QREAD+offset, mask, 0)
    u = tl.load(U+offset+raw_delta, mask, 0)
    tl.store(DZ+offset, coefficient*z, mask)
    tl.store(DQ+offset, dn*u, mask)
    tl.store(DU+offset, dn*q, mask)


@triton.jit
def _query_backward(DZ, DQ, DY, H, KP, DKCOEF, ZG, QG, RG, KG,
                    C: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                    BC: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    batch, ct, kt = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    cc, kk, vv = ct*BC+tl.arange(0, BC), kt*BK+tl.arange(0, BK), tl.arange(0, BV)
    dz = tl.full((BC, BK), 0, tl.float32)
    dq = tl.full((BC, BK), 0, tl.float32)
    dr = tl.full((BC, BK), 0, tl.float32)
    for block in range(tl.cdiv(V, BV)):
        value = block*BV+vv
        h = tl.load(H+batch*K*V+kk[None, :]*V+value[:, None],
                    (kk[None, :] < K) & (value[:, None] < V), 0)
        off = batch*C*V+cc[:, None]*V+value[None, :]
        mask = (cc[:, None] < C) & (value[None, :] < V)
        az = tl.load(DZ+off, mask, 0)
        dz = tl.dot(az, h, dz, input_precision='tf32x3')
        aq = tl.load(DQ+off, mask, 0)
        dq = tl.dot(aq, h, dq, input_precision='tf32x3')
        ar = tl.load(DY+off, mask, 0).to(tl.float32)
        dr = tl.dot(ar, h, dr, input_precision='tf32x3')
    off = batch*C*K+cc[:, None]*K+kk[None, :]
    mask = (cc[:, None] < C) & (kk[None, :] < K)
    key = tl.load(KP+off, mask, 0)
    dkey = tl.load(DKCOEF+batch*C+cc, cc < C, 0)
    tl.store(ZG+off, dz, mask)
    tl.store(QG+off, dq, mask)
    tl.store(RG+off, dr, mask)
    tl.store(KG+off, key*dkey[:, None], mask)


@triton.jit
def _state_backward(Z, Q, R, DZ, DQ, DY, H, DI, DH,
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
        z, dz = tl.load(Z+qo, qm, 0), tl.load(DZ+vo, vm, 0)
        dh = tl.dot(z, dz, dh, input_precision='tf32x3')
    off = batch*K*V+kk[:, None]*V+vv[None, :]
    mask = (kk[:, None] < K) & (vv[None, :] < V)
    h = tl.load(H+off, mask, 0)
    di = tl.load(DI+batch)
    tl.store(DH+off, dh+di*h, mask)


class _FusedCoarse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, beta, k, gc, state, rn, qn, u, temperature, floor, output_dtype):
        bh, n, c, key_dim = z.shape
        value_dim = state.shape[-1]
        tensors = tuple(x.contiguous() for x in (z, beta, k, gc, state, rn, qn, u, temperature))
        z, beta, k, gc, state, rn, qn, u, temperature = tensors
        matrices = bh*n
        cb, bv, bk = max(16, triton.next_power_of_2(c)), 32, 16
        vt = triton.cdiv(value_dim, bv)
        y = torch.empty((bh, n, c, value_dim), device=z.device, dtype=output_dtype)
        z2 = torch.empty((matrices, vt, c), device=z.device, dtype=torch.float32)
        numerator = torch.empty_like(z2)
        h2 = torch.empty((matrices, vt), device=z.device, dtype=torch.float32)
        stats = torch.empty((matrices, 5, c), device=z.device, dtype=torch.float32)
        scores = torch.empty((bh, n, c), device=z.device, dtype=torch.float64)
        bound = torch.empty((bh, n, c), device=z.device, dtype=torch.float32)
        zread, qread = (torch.empty_like(u) for _ in range(2))
        shared_temp = temperature.numel() == 1
        _project_forward[(matrices, vt)](z, rn, qn, state, u, y, z2, numerator, h2, zread, qread,
            c, key_dim, value_dim, cb, bv, bk, vt, num_warps=4, num_stages=1)
        _finish_forward[(matrices,)](k, beta, gc, temperature, z2, numerator, h2,
            scores, stats, bound, n, c, key_dim, cb, triton.next_power_of_2(key_dim),
            vt, triton.next_power_of_2(vt), floor*floor, shared_temp,
            num_warps=4, enable_fp_fusion=False)
        ctx.save_for_backward(*tensors, stats, zread, qread)
        ctx.coarse_dimensions = bh, n, c, key_dim, value_dim, floor, shared_temp
        norm2 = stats[:, 0].reshape(bh, n, c)
        ctx.mark_non_differentiable(norm2, bound)
        return y, scores, norm2, bound

    @staticmethod
    def backward(ctx, dy, ds, _dnorm, _dbound):
        z, beta, k, gc, state, rn, qn, u, temperature, stats, zread, qread = ctx.saved_tensors
        bh, n, c, key_dim, value_dim, floor, shared_temp = ctx.coarse_dimensions
        matrices = bh*n
        dy, ds = dy.contiguous(), ds.contiguous()
        dn, dzcoef, dkcoef, dbeta = (torch.empty_like(beta) for _ in range(4))
        di = torch.empty((matrices,), device=z.device, dtype=torch.float32)
        dgc, dt = torch.empty_like(gc), torch.empty_like(ds)
        _scalar_backward[(matrices,)](beta, gc, temperature, stats, ds,
            dn, dzcoef, dkcoef, di, dbeta, dgc, dt, n, c, triton.next_power_of_2(c),
            floor*floor, shared_temp, num_warps=4, enable_fp_fusion=False)
        dz, dq, du = (torch.empty_like(u) for _ in range(3))
        _activation_backward[(triton.cdiv(u.numel(), 256),)](
            zread, qread, u, dn, dzcoef, dz, dq, du, u.numel(), value_dim, 256,
            num_warps=4)
        zg, qg, rg, kg = (torch.empty_like(z) for _ in range(4))
        _query_backward[(matrices, triton.cdiv(c, 64), triton.cdiv(key_dim, 64))](
            dz, dq, dy, state, k, dkcoef, zg, qg, rg, kg,
            c, key_dim, value_dim, 64, 64, 16, num_warps=4, num_stages=1)
        dh = torch.empty_like(state)
        _state_backward[(matrices, triton.cdiv(key_dim, 64), triton.cdiv(value_dim, 64))](
            z, qn, rn, dz, dq, dy, state, di, dh, c, key_dim, value_dim,
            16, 64, 64, num_warps=4, num_stages=1)
        dtemperature = dt.sum_to_size(temperature.shape)
        return zg, dbeta, kg, dgc, dh, rg, qg, du, dtemperature, None, None


def fused_coarse_core(z, beta, k, gc, state, rn, qn, u, temperature, mass0,
                      floor, output_dtype, return_masks=False):
    """Drop-in experimental replacement for fast_matrix_gdn._coarse_core.

Matrix/scalar inputs use the same FP32/FP64 contract as that core. Returned
flags use its unchanged conditioning policy; the caller still owns fallback.
    """
    if not z.is_cuda:
        raise ValueError('The fused coarse router requires CUDA tensors.')
    if any(x.dtype != torch.float32 for x in (z, beta, k, state, rn, qn, u)):
        raise TypeError('The fused matrix core requires FP32 matrix inputs.')
    valid_temperature = (temperature.ndim == 0 or
                         (temperature.ndim == 3 and temperature.shape[1] == 1
                          and temperature.shape[2] == 1
                          and (temperature.shape[0] == 1 or temperature.shape[0] == z.shape[0])))
    if not valid_temperature:
        raise ValueError('Temperature must be scalar or have shape [heads*batch,1,1].')
    y, scores, norm2, bound = _FusedCoarse.apply(
        z, beta, k, gc, state, rn, qn, u, temperature, floor, output_dtype)
    # Import lazily: callers can later dispatch from the existing fast module.
    from .fast_matrix_gdn import _diagnostics
    from .decay_mass import high_mass
    bad, nonfinite = _diagnostics(norm2, bound, high_mass(mass0, gc), scores, y, floor)
    if return_masks:
        return y, scores, bad, nonfinite
    return y, scores, bad.reshape(bad.shape[0], -1).any(-1)
