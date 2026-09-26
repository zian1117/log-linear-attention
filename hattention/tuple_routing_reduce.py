"""Reduce separate Fenwick reads without packing the hierarchy dimension.

Scores retain FP64 precision until maximum subtraction. Softmax and weighted
value sums use FP32, matching the retained stacked reducer. Backward returns
contiguous gradients for each level instead of strided views of a packed tensor.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _tuple_reduce(YS, LOGITS, O, DO, DYS, DLOGITS,
                  T: tl.constexpr, L: tl.constexpr, V: tl.constexpr,
                  LB: tl.constexpr, VB: tl.constexpr,
                  READ_SCALE: tl.constexpr, BACKWARD: tl.constexpr):
    row = tl.program_id(0)
    t = row % T
    ll = tl.arange(0, LB)
    vv = tl.arange(0, VB)
    value_bases = tl.full((LB,), 0, tl.int64)
    score_bases = tl.full((LB,), 0, tl.int64)
    for level in tl.static_range(L):
        value_bases = tl.where(ll == level, YS[level].to(tl.int64), value_bases)
        score_bases = tl.where(ll == level, LOGITS[level].to(tl.int64), score_bases)
    value_ptrs = value_bases.to(tl.pointer_type(YS[0].dtype.element_ty))
    score_ptrs = score_bases.to(tl.pointer_type(LOGITS[0].dtype.element_ty))
    valid = (ll[:, None] < L) & (vv[None, :] < V)
    y = tl.load(value_ptrs[:, None] + row*V + vv[None, :], valid, 0).to(tl.float32)
    logits = tl.load(score_ptrs + row, ll < L, 0)
    active = (ll < L) & ((ll == 0) | (((t >> tl.maximum(ll - 1, 0)) & 1) != 0))
    masked = tl.where(active, logits, -float('inf'))
    shifted = (masked - tl.max(masked, 0)).to(tl.float32)
    weight = tl.exp(shifted)
    weight = weight / tl.sum(weight, 0)
    if not BACKWARD:
        output = tl.sum(weight[:, None] * y, 0) * READ_SCALE
        tl.store(O + row*V + vv, output, vv < V)
    else:
        do = tl.load(DO + row*V + vv, vv < V, 0).to(tl.float32) * READ_SCALE
        da = tl.sum(y * do[None, :], 1)
        dl = weight * (da - tl.sum(weight * da, 0))
        dy = weight[:, None] * do[None, :]
        dy_bases = tl.full((LB,), 0, tl.int64)
        dl_bases = tl.full((LB,), 0, tl.int64)
        for level in tl.static_range(L):
            dy_bases = tl.where(ll == level, DYS[level].to(tl.int64), dy_bases)
            dl_bases = tl.where(ll == level, DLOGITS[level].to(tl.int64), dl_bases)
        dy_ptrs = dy_bases.to(tl.pointer_type(DYS[0].dtype.element_ty))
        dl_ptrs = dl_bases.to(tl.pointer_type(DLOGITS[0].dtype.element_ty))
        tl.store(dy_ptrs[:, None] + row*V + vv[None, :], dy, valid)
        tl.store(dl_ptrs + row, dl, ll < L)


class _TupleRoutingReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, read_scale, levels, *inputs):
        ys = tuple(x.contiguous() for x in inputs[:levels])
        scores = tuple(x.contiguous() for x in inputs[levels:])
        bh, n, c, value_dim = ys[0].shape
        output = torch.empty_like(ys[0])
        _tuple_reduce[(bh*n*c,)](ys, scores, output, None, None, None,
            n*c, levels, value_dim, triton.next_power_of_2(levels),
            triton.next_power_of_2(value_dim), read_scale, False, num_warps=4)
        ctx.save_for_backward(*ys, *scores)
        ctx.levels, ctx.read_scale = levels, read_scale
        return output

    @staticmethod
    def backward(ctx, gradient):
        levels = ctx.levels
        # Non-reentrant checkpointing permits each saved tensor to be
        # unpacked once during a backward invocation.
        saved = ctx.saved_tensors
        ys, scores = saved[:levels], saved[levels:]
        dys, dls = tuple(torch.empty_like(x) for x in ys), tuple(torch.empty_like(x) for x in scores)
        bh, n, c, value_dim = ys[0].shape
        _tuple_reduce[(bh*n*c,)](ys, scores, None, gradient.contiguous(), dys, dls,
            n*c, levels, value_dim, triton.next_power_of_2(levels),
            triton.next_power_of_2(value_dim), ctx.read_scale, True, num_warps=4)
        return None, None, *dys, *dls


def tuple_routing_reduce(ys, scores, read_scale):
    """Reduce [BH,N,C,V] reads with matching FP64 [BH,N,C] level scores.

    Level zero is always active; higher levels use the token-position Fenwick
    mask. ``read_scale`` is the caller's inverse square root of key dimension.
    Inputs may be strided, but contiguous inputs avoid their conversion copies.
    """
    ys, scores = tuple(ys), tuple(scores)
    if not ys or len(ys) != len(scores):
        raise ValueError('Reads and scores must contain the same nonzero number of levels')
    first = ys[0]
    if first.ndim != 4 or not first.is_floating_point() or min(first.shape) == 0:
        raise ValueError('Reads must be nonempty floating-point [BH,N,C,V] tensors')
    if any(x.shape != first.shape or x.dtype != first.dtype or x.device != first.device for x in ys):
        raise ValueError('All level reads must share shape, dtype, and device')
    if any(x.shape != first.shape[:-1] or x.dtype != torch.float64 or x.device != first.device for x in scores):
        raise ValueError('Level scores must be FP64 [BH,N,C] tensors on the read device')
    return _TupleRoutingReduce.apply(read_scale, len(ys), *ys, *scores)
