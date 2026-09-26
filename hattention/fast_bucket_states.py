"""Experimental FP32 boundary scan, tiled over independent value columns."""

import torch
import triton
import triton.language as tl


@triton.jit
def _forward_one(bh, seg, tile, KP, WP, UP, AP, HP, N: tl.constexpr, C: tl.constexpr,
             K: tl.constexpr, V: tl.constexpr, P: tl.constexpr, BV: tl.constexpr):
    kk, vv, tt = tl.arange(0, K), tile * BV + tl.arange(0, BV), tl.arange(0, C)
    state = tl.full((K, BV), 0, tl.float32)
    for i in range(P):
        n = seg * P + i
        if n < N:
            off = bh * N + n
            tl.store(HP + off*K*V + kk[:, None]*V + vv[None, :], state, vv[None, :] < V)
            if i < P-1:
                key = tl.load(KP + off*C*K + tt[:, None]*K + kk[None, :])
                w = tl.load(WP + off*C*K + tt[:, None]*K + kk[None, :])
                u = tl.load(UP + off*C*V + tt[:, None]*V + vv[None, :], vv[None, :] < V, 0)
                if i >= P//2:
                    u = tl.full((C, BV), 0, tl.float32)
                residual = u - tl.dot(w, state, input_precision='tf32x3')
                a = tl.load(AP + off)
                state = a * state + tl.dot(tl.trans(key), residual, input_precision='tf32x3')


@triton.jit
def _forward(KP, WP, UP, AP, HP, N: tl.constexpr, C: tl.constexpr,
             K: tl.constexpr, V: tl.constexpr, P: tl.constexpr, BV: tl.constexpr):
    _forward_one(tl.program_id(0), tl.program_id(1), tl.program_id(2), KP, WP, UP, AP, HP, N, C, K, V, P, BV)


@triton.jit
def _backward(KP, WP, UP, AP, HP, DHP, DKP, DWP, DUP, DAP,
              N: tl.constexpr, C: tl.constexpr, K: tl.constexpr,
              V: tl.constexpr, P: tl.constexpr, BV: tl.constexpr):
    bh, seg, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    tiles: tl.constexpr = triton.cdiv(V, BV)
    kk, vv, tt = tl.arange(0, K), tile*BV + tl.arange(0, BV), tl.arange(0, C)
    dh = tl.full((K, BV), 0, tl.float32)
    for reverse in range(P):
        i = P-1-reverse
        n = seg*P+i
        if n < N:
            off = bh*N+n
            key = tl.load(KP + off*C*K + tt[:, None]*K + kk[None, :])
            w = tl.load(WP + off*C*K + tt[:, None]*K + kk[None, :])
            u = tl.load(UP + off*C*V + tt[:, None]*V + vv[None, :], vv[None, :] < V, 0)
            state = tl.load(HP + off*K*V + kk[:, None]*V + vv[None, :], vv[None, :] < V, 0)
            a = tl.load(AP+off)
            if i >= P//2:
                u = tl.full((C, BV), 0, tl.float32)
            residual = u - tl.dot(w, state, input_precision='tf32x3')
            dk = tl.dot(residual, tl.trans(dh), input_precision='tf32x3')
            partial = (off*tiles+tile)*C*K + tt[:, None]*K + kk[None, :]
            tl.store(DKP+partial, dk)
            du = tl.dot(key, dh, input_precision='tf32x3')
            dw = -tl.dot(du, tl.trans(state), input_precision='tf32x3')
            tl.store(DWP+partial, dw)
            tl.store(DUP+off*C*V + tt[:, None]*V + vv[None, :],
                     tl.where(i < P//2, du, 0.), vv[None, :] < V)
            da = tl.sum(tl.sum(state * dh, 1), 0)
            tl.store(DAP+off*tiles+tile, da)
            direct = tl.load(DHP+off*K*V + kk[:, None]*V + vv[None, :], vv[None, :] < V, 0)
            dh = a*dh - tl.dot(tl.trans(w), du, input_precision='tf32x3') + direct


def states(k, w, value, decay, period, value_tile=32, warps=4):
    bh, n, chunk, key_dim = k.shape
    value_dim = value.shape[-1]
    out = k.new_empty((bh, n, key_dim, value_dim))
    _forward[(bh, triton.cdiv(n, period), triton.cdiv(value_dim, value_tile))](
        k, w, value, decay, out, n, chunk, key_dim, value_dim, period,
        value_tile, num_warps=warps, num_stages=1)
    return out


def state_gradients(k, w, value, decay, state, direct, period, value_tile=32, warps=4):
    bh, n, chunk, key_dim = k.shape
    value_dim = value.shape[-1]
    tiles = triton.cdiv(value_dim, value_tile)
    dk, dw = (k.new_empty((bh, n, tiles, chunk, key_dim)) for _ in range(2))
    dv = torch.empty_like(value)
    da = decay.new_empty((bh, n, tiles))
    _backward[(bh, triton.cdiv(n, period), tiles)](
        k, w, value, decay, state, direct, dk, dw, dv, da,
        n, chunk, key_dim, value_dim, period, value_tile,
        num_warps=warps, num_stages=1)
    return dk.sum(2), dw.sum(2), dv, da.sum(2)


@triton.jit
def _adjoints_one(bh, seg, tile, KP, WP, AP, DHP, AHP, DUP,
              N: tl.constexpr, C: tl.constexpr, K: tl.constexpr,
              V: tl.constexpr, P: tl.constexpr, BV: tl.constexpr,
              STORE_DU: tl.constexpr = True):
    kk, vv, tt = tl.arange(0, K), tile*BV + tl.arange(0, BV), tl.arange(0, C)
    dh = tl.full((K, BV), 0, tl.float32)
    for reverse in range(P):
        i = P-1-reverse
        n = seg*P+i
        if n < N:
            off = bh*N+n
            tl.store(AHP+off*K*V + kk[:, None]*V + vv[None, :], dh, vv[None, :] < V)
            key = tl.load(KP+off*C*K + tt[:, None]*K + kk[None, :])
            du = tl.dot(key, dh, input_precision='tf32x3')
            if STORE_DU:
                tl.store(DUP+off*C*V + tt[:, None]*V + vv[None, :], du, vv[None, :] < V)
            w = tl.load(WP+off*C*K + tt[:, None]*K + kk[None, :])
            a = tl.load(AP+off)
            direct = tl.load(DHP+off*K*V + kk[:, None]*V + vv[None, :], vv[None, :] < V, 0)
            dh = a*dh - tl.dot(tl.trans(w), du, input_precision='tf32x3') + direct


@triton.jit
def _adjoints(KP, WP, AP, DHP, AHP, DUP,
              N: tl.constexpr, C: tl.constexpr, K: tl.constexpr,
              V: tl.constexpr, P: tl.constexpr, BV: tl.constexpr,
              STORE_DU: tl.constexpr = True):
    _adjoints_one(tl.program_id(0), tl.program_id(1), tl.program_id(2), KP, WP, AP, DHP, AHP, DUP, N, C, K, V, P, BV, STORE_DU)


@triton.jit
def _correction_one(bh, seg, tile, KP, WP, AP, DP, HP, N: tl.constexpr, C: tl.constexpr,
                K: tl.constexpr, V: tl.constexpr, P: tl.constexpr, BV: tl.constexpr):
    kk, vv, tt = tl.arange(0, K), tile*BV + tl.arange(0, BV), tl.arange(0, C)
    state = tl.full((K, BV), 0, tl.float32)
    for i in range(P):
        n = seg*P+i
        if n < N:
            off = bh*N+n
            tl.store(HP+off*K*V + kk[:, None]*V + vv[None, :], state, vv[None, :] < V)
            if i < P-1:
                w = tl.load(WP+off*C*K + tt[:, None]*K + kk[None, :])
                projection = tl.dot(w, state, input_precision='tf32x3')
                key = tl.load(KP+off*C*K + tt[:, None]*K + kk[None, :])
                a = tl.load(AP+off)
                defect = tl.load(DP+off*K*V + kk[:, None]*V + vv[None, :], vv[None, :] < V, 0)
                state = a*state - tl.dot(tl.trans(key), projection, input_precision='tf32x3') + defect


@triton.jit
def _correction(KP, WP, AP, DP, HP, N: tl.constexpr, C: tl.constexpr,
                K: tl.constexpr, V: tl.constexpr, P: tl.constexpr, BV: tl.constexpr):
    _correction_one(tl.program_id(0), tl.program_id(1), tl.program_id(2), KP, WP, AP, DP, HP, N, C, K, V, P, BV)


@torch.compile
def _parameter_gradients(w, value, state, adjoint, du, period):
    write = torch.arange(w.shape[1], device=w.device) % period < period//2
    mask = write[None, :, None, None]
    residual = torch.where(mask, value, 0.) - w @ state
    dk = residual @ adjoint.transpose(-1, -2)
    dw = -(du @ state.transpose(-1, -2))
    dv = torch.where(mask, du, 0.)
    da = (state * adjoint).sum((-2, -1))
    return dk, dw, dv, da


def state_gradients_split(k, w, value, decay, state, direct, period, value_tile=32, warps=4):
    """Keep only the adjoint recurrence serial; batch parameter gradients."""
    bh, n, chunk, key_dim = k.shape
    value_dim = value.shape[-1]
    adjoint = torch.empty_like(state)
    du = torch.empty_like(value)
    _adjoints[(bh, triton.cdiv(n, period), triton.cdiv(value_dim, value_tile))](
        k, w, decay, direct, adjoint, du, n, chunk, key_dim, value_dim,
        period, value_tile, num_warps=warps, num_stages=1)
    return _parameter_gradients(w, value, state, adjoint, du, period)


class TiledStates(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k, w, value, decay, period):
        k, w, value, decay = (x.contiguous() for x in (k, w, value, decay))
        state = states(k, w, value, decay, period)
        ctx.save_for_backward(k, w, value, decay, state)
        ctx.period = period
        return state

    @staticmethod
    def backward(ctx, direct):
        k, w, value, decay, state = ctx.saved_tensors
        return (*state_gradients_split(k, w, value, decay, state, direct.contiguous(), ctx.period), None)
