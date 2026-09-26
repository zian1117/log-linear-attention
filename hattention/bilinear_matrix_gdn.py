"""Independent token-conditioned routing and retrieval from Fenwick GDN states.

The ordinary path represents matrices at chunk boundaries. Within chunks,
reads and squared Frobenius norms use triangular solves and Gram products;
ill-conditioned norm calculations fall back to direct matrix recurrence.
"""
import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.utils.checkpoint import checkpoint

from .bucket_states import BucketStates
from .bucket_frobenius import local_bucket_norm2, high_bucket_norm2, _segment_blocks


# A computational tile, independent of the model's key/value dimensions.
_CHUNK_SIZE = 64


def _gdn_normalize(x):
    # Share one promotion: numerator and denominator gradients must cancel
    # in FP32 before the single conversion back to a BF16 input.
    x = x.float()
    return x * torch.rsqrt(x.square().sum(-1, keepdim=True) + 1e-6)


@torch.compile
def _prepare(r, k, v, beta, gc, q):
    c = k.shape[-2]
    gram = k @ k.transpose(-1, -2)
    eye = torch.eye(c, device=k.device, dtype=k.dtype)
    system = eye + (beta.unsqueeze(-1) * gram).tril(-1)
    inv = torch.linalg.solve_triangular(
        system, eye.expand_as(system), upper=False, unitriangular=True,
    )
    causal = torch.ones((c, c), device=k.device, dtype=torch.bool).tril()
    decay = (gc.unsqueeze(-1) - gc.unsqueeze(-2)).masked_fill(~causal, -torch.inf).exp()
    w = inv @ (beta.unsqueeze(-1) * k)
    writes = (inv * decay) @ (beta.unsqueeze(-1) * v)
    rk = (r @ k.transpose(-1, -2)).tril()
    qk = (q @ k.transpose(-1, -2)).tril()
    ar = (rk @ inv) * decay * beta.unsqueeze(-2)
    aq = (qk @ inv) * decay * beta.unsqueeze(-2)
    ge = gc.exp().unsqueeze(-1)
    rn = ge * (r - rk @ w)
    qn = ge * (q - qk @ w)
    kn = k * (gc[..., -1:] - gc).exp().unsqueeze(-1)
    wn = w * ge
    end = gc[..., -1].exp().contiguous()
    return kn, wn, writes, end, rn, qn, ar, aq, (gram, inv, decay, w)


@triton.jit
def _reduce(Y, LOGITS, O, DO, DY, DLOGITS,
            T: tl.constexpr, L: tl.constexpr, V: tl.constexpr,
            LB: tl.constexpr, VB: tl.constexpr,
            READ_SCALE: tl.constexpr, BACKWARD: tl.constexpr):
    row = tl.program_id(0)
    t = row % T
    ll = tl.arange(0, LB)
    vv = tl.arange(0, VB)
    offsets = (row * L + ll[:, None]) * V + vv[None, :]
    valid = (ll[:, None] < L) & (vv[None, :] < V)
    y = tl.load(Y + offsets, valid, 0).to(tl.float32)
    logits = tl.load(LOGITS + row * L + ll, ll < L, 0)
    active = (ll < L) & ((ll == 0) | (((t >> tl.maximum(ll - 1, 0)) & 1) != 0))
    masked = tl.where(active, logits, -float('inf'))
    # Scores are FP64. Subtract their maximum before converting to FP32 so
    # finite large scores neither overflow nor lose meaningful differences.
    shifted = (masked - tl.max(masked, 0)).to(tl.float32)
    weight = tl.exp(shifted)
    weight = weight / tl.sum(weight, 0)
    if not BACKWARD:
        out = tl.sum(weight[:, None] * y, 0) * READ_SCALE
        tl.store(O + row * V + vv, out, vv < V)
    else:
        do = tl.load(DO + row * V + vv, vv < V, 0).to(tl.float32) * READ_SCALE
        da = tl.sum(y * do[None, :], 1)
        dl = weight * (da - tl.sum(weight * da, 0))
        dy = weight[:, None] * do[None, :]
        tl.store(DY + offsets, dy, valid)
        tl.store(DLOGITS + row * L + ll, dl, ll < L)


class _RoutingReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, logits, read_scale):
        values, logits = values.contiguous(), logits.contiguous()
        bh, n, c, levels, value_dim = values.shape
        out = torch.empty((bh, n, c, value_dim), device=values.device, dtype=values.dtype)
        _reduce[(bh * n * c,)](
            values, logits, out, None, None, None,
            n * c, levels, value_dim, triton.next_power_of_2(levels),
            triton.next_power_of_2(value_dim), read_scale, False,
            num_warps=4,
        )
        ctx.save_for_backward(values, logits)
        ctx.read_scale = read_scale
        return out

    @staticmethod
    def backward(ctx, do):
        values, logits = ctx.saved_tensors
        bh, n, c, levels, value_dim = values.shape
        dy, dl = torch.empty_like(values), torch.empty_like(logits)
        _reduce[(bh * n * c,)](
            values, logits, None, do.contiguous(), dy, dl,
            n * c, levels, value_dim, triton.next_power_of_2(levels),
            triton.next_power_of_2(value_dim), ctx.read_scale, True,
            num_warps=4,
        )
        return dy, dl, None


@torch.compile
def _score(read, u, norm2, temperature, floor):
    # Preserve cancellation between numerator and denominator derivatives.
    # Only the final scalar score is rounded for FP32 softmax.
    return temperature * (read * u.double()).sum(-1) * norm2.clamp_min(floor ** 2).rsqrt()


def _local(ar, aq, k, v, beta, gc, u, temperature, level, terms, floor, output_dtype):
    chunk = k.shape[-2]
    norm2 = local_bucket_norm2(k, v, beta, gc, level, terms=terms, norm_floor=floor)
    if level == 0:
        y = ar.diagonal(dim1=-2, dim2=-1).unsqueeze(-1) * v.double()
        read = aq.diagonal(dim1=-2, dim2=-1).unsqueeze(-1) * v.double()
        return y.to(output_dtype), _score(read, u, norm2, temperature, floor)

    # A local bucket reads only the first half of its period, and only during
    # the second half. Dense C-by-C products would mostly multiply zeros.
    period = 1 << level
    half = period // 2
    groups = chunk // period
    value = v.double().reshape(*v.shape[:-2], groups, period, v.shape[-1])[..., :half, :]
    retrieval = _segment_blocks(ar, period)[..., half:, :half]
    routing = _segment_blocks(aq, period)[..., half:, :half]
    y = retrieval @ value
    read = routing @ value
    u = u.reshape(*u.shape[:-2], groups, period, u.shape[-1])[..., half:, :]
    norm2 = norm2.reshape(*norm2.shape[:-1], groups, period)[..., half:]
    score = _score(read, u, norm2, temperature.unsqueeze(-1), floor)
    y = torch.cat((torch.zeros_like(y), y), dim=-2).flatten(-3, -2)
    score = torch.cat((torch.zeros_like(score), score), dim=-1).flatten(-2, -1)
    return y.to(output_dtype), score


def _coarse(kn, wn, writes, end, rn, qn, k, beta, gc, u, temperature,
            period, terms, floor, output_dtype):
    state = BucketStates.apply(kn, wn, writes, end, period)
    # This Fenwick level is used only in each period's second half. Construct
    # the index from static shape information, avoiding a GPU nonzero()/sync.
    chunks = state.shape[1]
    active = torch.tensor(
        [chunk for chunk in range(chunks) if chunk % period >= period // 2],
        dtype=torch.long, device=state.device,
    )
    state = state.index_select(1, active)
    rn, qn, k, beta, gc, u = (
        x.index_select(1, active) for x in (rn, qn, k, beta, gc, u)
    )
    terms = tuple(x.index_select(1, active) for x in terms)
    norm2 = high_bucket_norm2(k, beta, gc, state, terms=terms, norm_floor=floor)
    y = (rn @ state).to(output_dtype)
    read = qn @ state
    score = _score(read, u, norm2, temperature, floor)
    # Inactive positions are masked by the final reducer. Their zero entries
    # carry no gradient, while the boundary recurrence still spans all chunks.
    output = y.new_zeros((y.shape[0], chunks, *y.shape[2:])).index_copy(1, active, y)
    logits = score.new_zeros((score.shape[0], chunks, *score.shape[2:])).index_copy(1, active, score)
    return output, logits


def precise_bilinear_matrix_gdn(r, k, v, g, beta, u, q, log_temperature,
                        norm_floor=1e-6, vector_eps=1e-6):
    """Return [batch,time,heads,value_dim], keeping tiny buckets in softmax.

    The routing probe q is independent of the retrieval query r. Routing vectors
    use floored L2 normalization; r/k retain GDN's additive-epsilon convention.
    """
    _validate_inputs(r, k, v, g, beta, u, q, log_temperature, norm_floor, vector_eps)
    bsz, length, heads, key_dim = k.shape
    value_dim = v.shape[-1]
    output_dtype = v.dtype
    levels = (length - 1).bit_length() + 1
    chunk = _CHUNK_SIZE
    padded = triton.cdiv(length, chunk) * chunk
    nchunks = padded // chunk

    def chunks(x):
        x = x.float()
        if padded != length:
            x = F.pad(x, (0, 0) * (x.ndim - 2) + (0, padded - length))
        return x.reshape(bsz, nchunks, chunk, heads, *x.shape[3:]).movedim(3, 1).reshape(
            bsz * heads, nchunks, chunk, *x.shape[3:]).contiguous()

    with torch.autocast('cuda', enabled=False):
        r, k = _gdn_normalize(r), _gdn_normalize(k)
        q = F.normalize(q.float(), dim=-1, eps=vector_eps)
        u = F.normalize(u.float(), dim=-1, eps=vector_eps)
        r, k, v, q, u = map(chunks, (r, k, v, q, u))
        beta = chunks(beta)
        gc = chunks(g).double().cumsum(-1)
        # Share each promotion across preparation, reads, and norms. Their
        # large cancelling gradient contributions must meet in FP64 before
        # the single cast back to the original FP32 normalized input.
        r, k, v, q, u, beta = (x.double() for x in (r, k, v, q, u, beta))
        # Erasure can make a bucket tiny through cancellation. Numerators and
        # norms must use the same precision before the denominator floor.
        kn, wn, writes, end, rn, qn, ar, aq, terms = _prepare(
            r, k, v, beta, gc, q)
        temperature = log_temperature.double().unsqueeze(0).expand(bsz, -1).reshape(bsz * heads, 1, 1).exp()
        ys, scores = [], []
        local_levels = min(chunk.bit_length(), levels)
        for level in range(local_levels):
            args = (ar, aq, k, v, beta, gc, u, temperature, level, terms, norm_floor, output_dtype)
            y, score = checkpoint(_local, *args, use_reentrant=False) if torch.is_grad_enabled() else _local(*args)
            ys.append(y)
            scores.append(score)
        for level in range(local_levels, levels):
            period = 1 << (level - (chunk.bit_length() - 1))
            args = (kn, wn, writes, end, rn, qn, k, beta, gc, u, temperature,
                    period, terms, norm_floor, output_dtype)
            y, score = checkpoint(_coarse, *args, use_reentrant=False) if torch.is_grad_enabled() else _coarse(*args)
            ys.append(y)
            scores.append(score)
        out = _RoutingReduce.apply(torch.stack(ys, -2), torch.stack(scores, -1), key_dim ** -.5)
    return out.reshape(bsz, heads, nchunks, chunk, value_dim).permute(0, 2, 3, 1, 4).reshape(
        bsz, padded, heads, value_dim)[:, :length]


def _validate_inputs(r, k, v, g, beta, u, q, log_temperature, norm_floor, vector_eps):
    if not (math.isfinite(norm_floor) and norm_floor > 0 and
            math.isfinite(vector_eps) and vector_eps > 0):
        raise ValueError('Normalization floors must be finite and positive.')
    if k.ndim != 4 or k.shape[1] == 0:
        raise ValueError('Expected nonempty [batch,time,heads,key_dim] inputs.')
    bsz, length, heads, key_dim = k.shape
    value_dim = v.shape[-1]
    if (r.shape != k.shape or q.shape != k.shape or
            v.shape != (bsz, length, heads, value_dim) or u.shape != v.shape or
            g.shape != k.shape[:3] or beta.shape != k.shape[:3] or
            log_temperature.shape != (heads,)):
        raise ValueError('Incompatible bilinear matrix router input shapes.')
    if not k.is_cuda:
        raise ValueError('The optimized bilinear matrix router requires CUDA.')


def bilinear_matrix_gdn(r, k, v, g, beta, u, q, log_temperature,
                        norm_floor=1e-6, vector_eps=1e-6):
    """Route over Fenwick matrices with separate normalized u/q and retrieval r.

    Matrix arithmetic ordinarily uses FP32. Ill-conditioned portions retain
    their full history and are recomputed precisely; no tiny bucket is masked.
    The explicit precise implementation remains available as a test reference.
    """
    _validate_inputs(r, k, v, g, beta, u, q, log_temperature, norm_floor, vector_eps)
    from .adaptive_matrix_gdn import adaptive_matrix_gdn
    return adaptive_matrix_gdn(
        r, k, v, g, beta, u, q, log_temperature,
        norm_floor=norm_floor, vector_eps=vector_eps,
        repair_periods=True, shared_local=True, fused_local=True,
        repair_chunks=True, shared_gathers=True, compact_cache_reads=True,
        shared_states=True,
    )
