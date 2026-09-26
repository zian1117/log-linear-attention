"""Experimental shared state scan and norm projections.

The recurrence already needs Z @ H. Return that product for the Frobenius
calculation, avoiding a second evaluation and combining its gradient with
the recurrence gradient. H is key-by-value in this implementation.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _forward(KP, ZP, FP, UP, AP, HP, PP, N: tl.constexpr,
             C: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
             PERIOD: tl.constexpr, BV: tl.constexpr):
    bh, segment, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    kk, vv, tt = tl.arange(0, K), tile*BV + tl.arange(0, BV), tl.arange(0, C)
    state = tl.full((K, BV), 0, tl.float32)
    for i in range(PERIOD):
        n = segment*PERIOD+i
        if n < N:
            off = bh*N+n
            tl.store(HP+off*K*V+kk[:, None]*V+vv[None, :], state, vv[None, :] < V)
            z = tl.load(ZP+off*C*K+tt[:, None]*K+kk[None, :])
            projected = tl.dot(z, state, input_precision='tf32x3')
            tl.store(PP+off*C*V+tt[:, None]*V+vv[None, :], projected, vv[None, :] < V)
            if i < PERIOD-1:
                key = tl.load(KP+off*C*K+tt[:, None]*K+kk[None, :])
                factor = tl.load(FP+off*C+tt)
                value = tl.load(UP+off*C*V+tt[:, None]*V+vv[None, :], vv[None, :] < V, 0)
                if i >= PERIOD//2:
                    value = tl.full((C, BV), 0, tl.float32)
                residual = value-factor[:, None]*projected
                decay = tl.load(AP+off)
                state = decay*state+tl.dot(tl.trans(key), residual, input_precision='tf32x3')


@triton.jit
def _adjoints(KP, ZP, FP, AP, DHP, DPP, AHP, DUP, DZHP,
              N: tl.constexpr, C: tl.constexpr, K: tl.constexpr,
              V: tl.constexpr, PERIOD: tl.constexpr, BV: tl.constexpr,
              COMPACT: tl.constexpr = False, ACTIVE: tl.constexpr = 0):
    bh, segment, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    kk, vv, tt = tl.arange(0, K), tile*BV + tl.arange(0, BV), tl.arange(0, C)
    adjoint = tl.full((K, BV), 0, tl.float32)
    for reverse in range(PERIOD):
        i = PERIOD-1-reverse
        n = segment*PERIOD+i
        if n < N:
            off = bh*N+n
            tl.store(AHP+off*K*V+kk[:, None]*V+vv[None, :], adjoint, vv[None, :] < V)
            key = tl.load(KP+off*C*K+tt[:, None]*K+kk[None, :])
            du = tl.dot(key, adjoint, input_precision='tf32x3')
            tl.store(DUP+off*C*V+tt[:, None]*V+vv[None, :], du, vv[None, :] < V)
            factor = tl.load(FP+off*C+tt)
            if COMPACT:
                grad_off = bh*ACTIVE+segment*(PERIOD-PERIOD//2)+i-PERIOD//2
                has_direct = i >= PERIOD//2
            else:
                grad_off = off
                has_direct = True
            dp = tl.load(DPP+grad_off*C*V+tt[:, None]*V+vv[None, :],
                         has_direct & (vv[None, :] < V), 0)
            combined = dp-factor[:, None]*du
            tl.store(DZHP+off*C*V+tt[:, None]*V+vv[None, :], combined, vv[None, :] < V)
            z = tl.load(ZP+off*C*K+tt[:, None]*K+kk[None, :])
            decay = tl.load(AP+off)
            direct = tl.load(DHP+grad_off*K*V+kk[:, None]*V+vv[None, :],
                             has_direct & (vv[None, :] < V), 0)
            adjoint = decay*adjoint+tl.dot(tl.trans(z), combined, input_precision='tf32x3')+direct


@torch.compile
def _factor_gradients(factor, value, state, projected, adjoint, du, combined, period):
    write = (torch.arange(value.shape[1], device=value.device) % period < period//2)[None, :, None, None]
    residual = torch.where(write, value, 0.)-factor.unsqueeze(-1)*projected
    dk = residual @ adjoint.transpose(-1, -2)
    dz = combined @ state.transpose(-1, -2)
    df = -(du*projected).sum(-1)
    dv = torch.where(write, du, 0.)
    da = (state*adjoint).sum((-2, -1))
    return dk, dz, df, dv, da


class ProjectedStates(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k, z, factor, value, decay, period):
        k, z, factor, value, decay = (x.contiguous() for x in (k,z,factor,value,decay))
        batch, chunks, chunk, key_dim = k.shape
        value_dim = value.shape[-1]
        state = k.new_empty((batch, chunks, key_dim, value_dim))
        projected = torch.empty_like(value)
        tile = 32
        _forward[(batch, triton.cdiv(chunks, period), triton.cdiv(value_dim, tile))](
            k,z,factor,value,decay,state,projected,chunks,chunk,key_dim,value_dim,
            period,tile,num_warps=4,num_stages=1)
        ctx.save_for_backward(k,z,factor,value,decay,state,projected)
        ctx.period = period
        return state, projected

    @staticmethod
    def backward(ctx, direct, projected_gradient):
        k,z,factor,value,decay,state,projected = ctx.saved_tensors
        batch,chunks,chunk,key_dim = k.shape
        value_dim,tile = value.shape[-1],32
        adjoint = torch.empty_like(state)
        du,combined = torch.empty_like(value),torch.empty_like(value)
        _adjoints[(batch,triton.cdiv(chunks,ctx.period),triton.cdiv(value_dim,tile))](
            k,z,factor,decay,direct.contiguous(),projected_gradient.contiguous(),
            adjoint,du,combined,chunks,chunk,key_dim,value_dim,ctx.period,tile,
            num_warps=4,num_stages=1)
        return (*_factor_gradients(factor,value,state,projected,adjoint,du,combined,ctx.period),None)
