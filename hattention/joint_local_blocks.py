"""Joint adjoint for diagonal block views at multiple Fenwick levels."""
import torch
import triton
import triton.language as tl


from .bucket_frobenius import _segment_blocks as segment_blocks


@triton.jit
def _sum_blocks(SOURCES, OUTPUT, PERIODS, C: tl.constexpr, TOTAL: tl.constexpr,
                DOUBLE: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    batch = offset//(C*C)
    row, column = (offset//C)%C, offset%C
    value = tl.full((BLOCK,), 0, tl.float64 if DOUBLE else tl.float32)
    for index in tl.static_range(len(PERIODS)):
        p = PERIODS[index]
        group = row//p
        index_in = ((batch*(C//p)+group)*p+row%p)*p+column%p
        add = tl.load(SOURCES[index]+index_in,
                      (offset<TOTAL)&(group==column//p),other=0)
        value += add
    tl.store(OUTPUT+offset,value,offset<TOTAL)


def _adjoint_impl(sources:list[torch.Tensor], periods:list[int], shape:list[int])->torch.Tensor:
    sources=tuple(x.contiguous() for x in sources)
    result=sources[0].new_empty(shape)
    _sum_blocks[(triton.cdiv(result.numel(),1024),)](
        sources,result,tuple(periods),shape[-1],result.numel(),
        result.dtype==torch.float64,1024,num_warps=4)
    return result

try:
    _adjoint=torch.ops.hattention.joint_local_blocks_backward.default
except AttributeError:
    _adjoint=torch.library.custom_op('hattention::joint_local_blocks_backward',_adjoint_impl,mutates_args=())
    @_adjoint.register_fake
    def _adjoint_fake(sources,periods,shape):
        return sources[0].new_empty(shape)


class _JointBlocks(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,periods):
        ctx.shape,ctx.dtype,ctx.device=x.shape,x.dtype,x.device
        ctx.periods=periods
        ctx.set_materialize_grads(False)
        return tuple(segment_blocks(x,p) for p in periods)
    @staticmethod
    def backward(ctx,*gradients):
        supplied=[(p,g) for p,g in zip(ctx.periods,gradients) if g is not None]
        # Match reverse graph traversal's usual level-addition order.
        supplied.reverse()
        if not supplied:
            return torch.zeros(ctx.shape,dtype=ctx.dtype,device=ctx.device),None
        periods,sources=zip(*supplied)
        if ctx.device.type=='cuda':
            return _adjoint(list(sources),list(periods),list(ctx.shape)),None
        result=torch.zeros(ctx.shape,dtype=ctx.dtype,device=ctx.device)
        for p,g in supplied:
            segment_blocks(result,p).add_(g)
        return result,None


def joint_blocks(x,periods):
    periods=tuple(periods)
    if x.shape[-2]!=x.shape[-1] or any(p<=0 or x.shape[-1]%p for p in periods):
        raise ValueError('Expected square matrix and periods dividing its width')
    return _JointBlocks.apply(x,periods) if periods else ()
