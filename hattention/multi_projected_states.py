"""Experimental joint backward for multiple projected Fenwick state scans.

The scans themselves are unchanged. Their matrix gradients accumulate in
the existing FP32 cuBLAS products' epilogues, avoiding a dense temporary
and separate addition for each period. All periods' incoming gradients
must be available together, which can increase peak memory.
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .projected_bucket_states import _forward, _adjoints
from .fenwick_gather import _gather, _count


@triton.jit
def _residual_and_auxiliary(VALUE, PROJECTED, FACTOR, DU, RESIDUAL, DF, DV,
                            N: tl.constexpr, C: tl.constexpr, V: tl.constexpr,
                            ROWS: tl.constexpr, PERIOD: tl.constexpr,
                            INITIAL: tl.constexpr, BR: tl.constexpr,
                            BV: tl.constexpr):
    row = tl.program_id(0)*BR+tl.arange(0, BR)
    value = tl.arange(0, BV)
    mask = (row[:, None] < ROWS) & (value[None, :] < V)
    off = row[:, None]*V+value[None, :]
    write = ((row//C) % N) % PERIOD < PERIOD//2
    original = tl.load(VALUE+off, mask, 0)
    projected = tl.load(PROJECTED+off, mask, 0)
    du = tl.load(DU+off, mask, 0)
    factor = tl.load(FACTOR+row, row < ROWS, 0)
    residual = tl.where(write[:, None], original, 0.)-factor[:, None]*projected
    df = -tl.sum(du*projected, 1)
    dv = tl.where(write[:, None], du, 0.)
    if not INITIAL:
        df += tl.load(DF+row, row < ROWS, 0)
        dv += tl.load(DV+off, mask, 0)
    tl.store(RESIDUAL+off, residual, mask)
    tl.store(DF+row, df, row < ROWS)
    tl.store(DV+off, dv, mask)


@triton.jit
def _decay_gradient(STATE, ADJOINT, DA, WIDTH: tl.constexpr,
                    INITIAL: tl.constexpr, BLOCK: tl.constexpr):
    matrix = tl.program_id(0)
    inner = tl.arange(0, BLOCK)
    state = tl.load(STATE+matrix*WIDTH+inner, inner < WIDTH, 0)
    adjoint = tl.load(ADJOINT+matrix*WIDTH+inner, inner < WIDTH, 0)
    result = tl.sum(state*adjoint, 0)
    if not INITIAL:
        result += tl.load(DA+matrix)
    tl.store(DA+matrix, result)


class _MultiProjectedStates(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k, z, factor, value, decay, periods, active_outputs):
        k, z, factor, value, decay = (x.contiguous() for x in (k,z,factor,value,decay))
        batch, chunks, chunk, key_dim = k.shape
        value_dim = value.shape[-1]
        states, projections, outputs = [], [], []
        for period in periods:
            state = k.new_empty((batch,chunks,key_dim,value_dim))
            projected = torch.empty_like(value)
            _forward[(batch,triton.cdiv(chunks,period),triton.cdiv(value_dim,32))](
                k,z,factor,value,decay,state,projected,chunks,chunk,key_dim,value_dim,
                period,32,num_warps=4,num_stages=1)
            states.append(state)
            projections.append(projected)
            outputs.extend((_gather(state,period),_gather(projected,period))
                           if active_outputs else (state,projected))
        ctx.save_for_backward(k,z,factor,value,decay,*states,*projections)
        ctx.periods = periods
        ctx.active_outputs = active_outputs
        ctx.set_materialize_grads(False)
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *gradients):
        k,z,factor,value,decay,*saved = ctx.saved_tensors
        count = len(ctx.periods)
        states, projections = saved[:count], saved[count:]
        batch,chunks,chunk,key_dim = k.shape
        value_dim = value.shape[-1]
        dk,dz,df,dv,da = (torch.empty_like(x) for x in (k,z,factor,value,decay))
        adjoint = torch.empty_like(states[0])
        du,combined,residual = (torch.empty_like(value) for _ in range(3))
        zero_state, zero_projection = None, None
        initial = True
        for index,period in enumerate(ctx.periods):
            direct, projection_gradient = gradients[2*index:2*index+2]
            if ((direct is None or not direct.numel()) and
                    (projection_gradient is None or not projection_gradient.numel())):
                continue
            active = _count(chunks,period) if ctx.active_outputs else chunks
            if direct is None:
                if zero_state is None or zero_state.shape[1] != active:
                    zero_state = k.new_zeros((batch,active,key_dim,value_dim))
                direct = zero_state
            if projection_gradient is None:
                if zero_projection is None or zero_projection.shape[1] != active:
                    zero_projection = value.new_zeros((batch,active,chunk,value_dim))
                projection_gradient = zero_projection
            _adjoints[(batch,triton.cdiv(chunks,period),triton.cdiv(value_dim,32))](
                k,z,factor,decay,direct.contiguous(),projection_gradient.contiguous(),
                adjoint,du,combined,chunks,chunk,key_dim,value_dim,period,32,
                ctx.active_outputs,active,
                num_warps=4,num_stages=1)
            _residual_and_auxiliary[(triton.cdiv(factor.numel(),16),)](
                value,projections[index],factor,du,residual,df,dv,chunks,chunk,value_dim,
                factor.numel(),period,initial,16,triton.next_power_of_2(value_dim),
                num_warps=4,enable_fp_fusion=False)
            _decay_gradient[(batch*chunks,)](
                states[index],adjoint,da,key_dim*value_dim,initial,
                triton.next_power_of_2(key_dim*value_dim),num_warps=8)
            dk_matrix = dk.view(batch*chunks,chunk,key_dim)
            dz_matrix = dz.view(batch*chunks,chunk,key_dim)
            # beta=0 on the first used period ignores the uninitialized
            # accumulators. Subsequent products add in the GEMM epilogue.
            torch.baddbmm(dk_matrix,residual.view(batch*chunks,chunk,value_dim),
                adjoint.view(batch*chunks,key_dim,value_dim).transpose(-1,-2),
                beta=0 if initial else 1,out=dk_matrix)
            torch.baddbmm(dz_matrix,combined.view(batch*chunks,chunk,value_dim),
                states[index].view(batch*chunks,key_dim,value_dim).transpose(-1,-2),
                beta=0 if initial else 1,out=dz_matrix)
            initial = False
        if initial:
            for gradient in (dk,dz,df,dv,da):
                gradient.zero_()
        return dk,dz,df,dv,da,None,None


def multi_projected_states(k,z,factor,value,decay,periods, *, active_outputs=False):
    """Return a tuple of (boundary states, z@state) pairs, one per period.

Inputs match ProjectedStates: k/z [B,N,C,K], factor [B,N,C],
value [B,N,C,V], decay [B,N]. Odd feature/chunk dimensions are padded
internally for the unchanged tensor-core scan kernels.
With active_outputs=True, return only second-half chunks of each period;
backward reads their compact direct gradients without dense expansion. The
complete prefix states and recurrent gradients are retained internally.
"""
    periods = tuple(periods)
    if not periods:
        return ()
    if any(not isinstance(p,int) or isinstance(p,bool) or p < 2 for p in periods):
        raise ValueError('Each Fenwick period must be an integer >= 2.')
    if not k.is_cuda or any(x.dtype != torch.float32 for x in (k,z,factor,value,decay)):
        raise ValueError('Joint projected scans require CUDA FP32 inputs.')
    if k.ndim != 4 or min(k.shape) <= 0 or value.shape[-1] <= 0:
        raise ValueError('Projected scan inputs must have nonempty [B,N,C,K/V] dimensions.')
    if (z.shape != k.shape or factor.shape != k.shape[:-1]
            or value.shape[:-1] != k.shape[:-1] or decay.shape != k.shape[:2]):
        raise ValueError('Incompatible projected scan input shapes.')
    chunk,key_dim,value_dim = k.shape[-2],k.shape[-1],value.shape[-1]
    cp = max(16,triton.next_power_of_2(chunk))-chunk
    kp = max(16,triton.next_power_of_2(key_dim))-key_dim
    vp = max(16,triton.next_power_of_2(value_dim))-value_dim
    inputs = (F.pad(k,(0,kp,0,cp)) if kp or cp else k,
              F.pad(z,(0,kp,0,cp)) if kp or cp else z,
              F.pad(factor,(0,cp)) if cp else factor,
              F.pad(value,(0,vp,0,cp)) if vp or cp else value,decay)
    flat = _MultiProjectedStates.apply(*inputs,periods,active_outputs)
    return tuple((flat[2*i][...,:key_dim,:value_dim],flat[2*i+1][...,:chunk,:value_dim])
                 for i in range(len(periods)))
