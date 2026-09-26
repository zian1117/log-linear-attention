"""Experimental three local-bucket projections sharing their value matrix.

The inputs are three [..., H, H] coefficient matrices and [..., H, V]
values. This primitive does not construct coefficients or bucket norms.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _small_forward(A, B, D, X, Y, R, P, SIZE: tl.constexpr,
                   H: tl.constexpr, V: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    mask = i < SIZE
    batch, row, col = i//(H*V), (i//V)%H, i%V
    y, r, p = tl.full((BLOCK,), 0, tl.float32), tl.full((BLOCK,), 0, tl.float32), tl.full((BLOCK,), 0, tl.float32)
    for j in tl.static_range(H):
        x = tl.load(X+batch*H*V+j*V+col, mask, 0)
        off = batch*H*H+row*H+j
        y += tl.load(A+off, mask, 0)*x
        r += tl.load(B+off, mask, 0)*x
        p += tl.load(D+off, mask, 0)*x
    tl.store(Y+i, y, mask)
    tl.store(R+i, r, mask)
    tl.store(P+i, p, mask)


@triton.jit
def _small_matrix_backward(X, DY, DR, DP, DA, DB, DD, SIZE: tl.constexpr,
                           H: tl.constexpr, V: tl.constexpr,
                           BLOCK: tl.constexpr, VB: tl.constexpr):
    i = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    v = tl.arange(0, VB)
    batch, row, col = i//(H*H), (i//H)%H, i%H
    mask = (i[:, None] < SIZE) & (v[None, :] < V)
    x = tl.load(X+batch[:, None]*H*V+col[:, None]*V+v[None, :], mask, 0)
    off = batch[:, None]*H*V+row[:, None]*V+v[None, :]
    a = tl.sum(tl.load(DY+off, mask, 0)*x, 1)
    b = tl.sum(tl.load(DR+off, mask, 0)*x, 1)
    d = tl.sum(tl.load(DP+off, mask, 0)*x, 1)
    tl.store(DA+i, a, i < SIZE)
    tl.store(DB+i, b, i < SIZE)
    tl.store(DD+i, d, i < SIZE)


@triton.jit
def _small_value_backward(A, B, D, DY, DR, DP, DX, SIZE: tl.constexpr,
                          H: tl.constexpr, V: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    mask = i < SIZE
    batch, row, col = i//(H*V), (i//V)%H, i%V
    dx = tl.full((BLOCK,), 0, tl.float32)
    for j in tl.static_range(H):
        off = batch*H*H+j*H+row
        grad = batch*H*V+j*V+col
        dx += tl.load(A+off, mask, 0)*tl.load(DY+grad, mask, 0)
        dx += tl.load(B+off, mask, 0)*tl.load(DR+grad, mask, 0)
        dx += tl.load(D+off, mask, 0)*tl.load(DP+grad, mask, 0)
    tl.store(DX+i, dx, mask)


@triton.jit
def _forward(A, B, D, X, Y, R, P,
             H: tl.constexpr, V: tl.constexpr, HB: tl.constexpr,
             VB: tl.constexpr):
    batch, tile = tl.program_id(0), tl.program_id(1)
    h, j = tl.arange(0, HB), tl.arange(0, HB)
    v = tile*VB+tl.arange(0, VB)
    aoff = batch*H*H+h[:, None]*H+j[None, :]
    amask = (h[:, None] < H) & (j[None, :] < H)
    x = tl.load(X+batch*H*V+j[:, None]*V+v[None, :],
                (j[:, None] < H) & (v[None, :] < V), 0)
    a, b, d = tl.load(A+aoff, amask, 0), tl.load(B+aoff, amask, 0), tl.load(D+aoff, amask, 0)
    y = tl.dot(a, x, input_precision='tf32x3')
    r = tl.dot(b, x, input_precision='tf32x3')
    p = tl.dot(d, x, input_precision='tf32x3')
    off = batch*H*V+h[:, None]*V+v[None, :]
    mask = (h[:, None] < H) & (v[None, :] < V)
    tl.store(Y+off, y, mask)
    tl.store(R+off, r, mask)
    tl.store(P+off, p, mask)


@triton.jit
def _matrix_backward(X, DY, DR, DP, DA, DB, DD,
                     H: tl.constexpr, V: tl.constexpr,
                     HB: tl.constexpr, VB: tl.constexpr):
    batch = tl.program_id(0)
    h, j, v = tl.arange(0, HB), tl.arange(0, HB), tl.arange(0, VB)
    a, b, d = tl.full((HB, HB), 0, tl.float32), tl.full((HB, HB), 0, tl.float32), tl.full((HB, HB), 0, tl.float32)
    for tile in range(tl.cdiv(V, VB)):
        vv = tile*VB+v
        x = tl.load(X+batch*H*V+j[None, :]*V+vv[:, None],
                    (j[None, :] < H) & (vv[:, None] < V), 0)
        off = batch*H*V+h[:, None]*V+vv[None, :]
        mask = (h[:, None] < H) & (vv[None, :] < V)
        y, r, p = tl.load(DY+off, mask, 0), tl.load(DR+off, mask, 0), tl.load(DP+off, mask, 0)
        a = tl.dot(y, x, a, input_precision='tf32x3')
        b = tl.dot(r, x, b, input_precision='tf32x3')
        d = tl.dot(p, x, d, input_precision='tf32x3')
    off = batch*H*H+h[:, None]*H+j[None, :]
    mask = (h[:, None] < H) & (j[None, :] < H)
    tl.store(DA+off, a, mask)
    tl.store(DB+off, b, mask)
    tl.store(DD+off, d, mask)


@triton.jit
def _value_backward(A, B, D, DY, DR, DP, DX,
                    H: tl.constexpr, V: tl.constexpr,
                    HB: tl.constexpr, VB: tl.constexpr):
    batch, tile = tl.program_id(0), tl.program_id(1)
    h, j = tl.arange(0, HB), tl.arange(0, HB)
    v = tile*VB+tl.arange(0, VB)
    off = batch*H*H+j[None, :]*H+h[:, None]
    mask = (j[None, :] < H) & (h[:, None] < H)
    a, b, d = tl.load(A+off, mask, 0), tl.load(B+off, mask, 0), tl.load(D+off, mask, 0)
    off = batch*H*V+j[:, None]*V+v[None, :]
    mask = (j[:, None] < H) & (v[None, :] < V)
    dy, dr, dp = tl.load(DY+off, mask, 0), tl.load(DR+off, mask, 0), tl.load(DP+off, mask, 0)
    dx = tl.dot(a, dy, input_precision='tf32x3')
    dx = tl.dot(b, dr, dx, input_precision='tf32x3')
    dx = tl.dot(d, dp, dx, input_precision='tf32x3')
    off = batch*H*V+h[:, None]*V+v[None, :]
    mask = (h[:, None] < H) & (v[None, :] < V)
    tl.store(DX+off, dx, mask)


class _LocalProjections(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, d, value):
        a, b, d, value = (x.contiguous() for x in (a, b, d, value))
        h, v = value.shape[-2:]
        batches = value.numel()//(h*v)
        outputs = tuple(torch.empty_like(value) for _ in range(3))
        if h <= 4:
            _small_forward[(triton.cdiv(value.numel(), 256),)](
                a, b, d, value, *outputs, value.numel(), h, v, 256, enable_fp_fusion=False)
        else:
            _forward[(batches, triton.cdiv(v, 32))](
                a, b, d, value, *outputs, h, v, max(16, triton.next_power_of_2(h)), 32,
                num_warps=4, num_stages=1)
        ctx.save_for_backward(a, b, d, value)
        return outputs

    @staticmethod
    def backward(ctx, dy, dr, dp):
        a, b, d, value = ctx.saved_tensors
        dy, dr, dp = (x.contiguous() for x in (dy, dr, dp))
        h, v = value.shape[-2:]
        batches = value.numel()//(h*v)
        da, db, dd, dx = (torch.empty_like(x) for x in (a, b, d, value))
        if h <= 4:
            _small_matrix_backward[(triton.cdiv(a.numel(), 32),)](
                value, dy, dr, dp, da, db, dd, a.numel(), h, v, 32,
                triton.next_power_of_2(v), num_warps=4, enable_fp_fusion=False)
            _small_value_backward[(triton.cdiv(value.numel(), 256),)](
                a, b, d, dy, dr, dp, dx, value.numel(), h, v, 256, enable_fp_fusion=False)
        else:
            hb = max(16, triton.next_power_of_2(h))
            _matrix_backward[(batches,)](value, dy, dr, dp, da, db, dd,
                h, v, hb, 16, num_warps=4, num_stages=1)
            _value_backward[(batches, triton.cdiv(v, 32))](a, b, d, dy, dr, dp, dx,
                h, v, hb, 32, num_warps=4, num_stages=1)
        return da, db, dd, dx


def local_bucket_projections(a, b, d, value):
    """Return (a @ value, b @ value, d @ value), with shared-value backward."""
    if not value.is_cuda:
        raise ValueError('Local fused projections require CUDA.')
    if any(x.dtype != torch.float32 for x in (a, b, d, value)):
        raise TypeError('Local fused projections require FP32 inputs.')
    expected = (*value.shape[:-1], value.shape[-2])
    if any(x.shape != expected for x in (a, b, d)):
        raise ValueError('Coefficient matrices must be [..., H, H] and values [..., H, V].')
    return _LocalProjections.apply(a, b, d, value)
