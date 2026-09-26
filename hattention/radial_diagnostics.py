"""Detached indicators for cancellation in normalized matrix-score gradients.

These indicators supplement existing forward-error guards. They are not a
certificate for arbitrary gradient accuracy. All returned decisions are
nondifferentiable, and do not change the attention equations or norm floor.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@torch.no_grad()
def mass_decision(norm2, mass, *, level, floor=1e-6):
    scale = torch.maximum(norm2.abs(), mass.square()).clamp_min(floor * floor)
    return ((level >= 2) & (norm2 >= floor * floor) & (mass > 0)
            & (mass.square() - norm2 <= torch.finfo(norm2.dtype).eps ** .5 * scale))


@torch.no_grad()
def erase_decision(post_norm2, pre_norm2, projection2, key2, *, level, floor=1e-6):
    product = key2 * pre_norm2
    scale = torch.maximum(product.abs(), projection2.abs())
    return ((level >= 1) & (post_norm2 >= floor * floor) & (scale > 0)
            & (product - projection2 <= torch.finfo(product.dtype).eps ** .5 * scale))


@torch.no_grad()
def dominance_decision(norm2, rest, floor=1e-6):
    return ((norm2 >= floor * floor)
            & (rest <= torch.finfo(norm2.dtype).eps ** .5
               * norm2.clamp_min(floor * floor).sqrt()))


@torch.no_grad()
def write_derivative_decision(pre_norm2, p, e, key2):
    """Flag near-radial beta changes before adding a rank-one write.

    A is the decayed previous state, p=k^T A, e=v-p. For S=A+beta*k*e^T,
    the squared derivative of S/||S|| with respect to beta is
    (||k||² ||e||² ||A||² - (p·e)²) / ||S||⁴. Its numerator is independent
    of beta. No floor exclusion is valid here: later writes may increase the
    state above the normalization floor.
    """
    return write_derivative_scalar(pre_norm2, (p * e).sum(-1),
                                   e.square().sum(-1), key2)


@torch.no_grad()
def write_derivative_scalar(pre_norm2, p_dot_e, residual_norm2, key2):
    """Scalar form of ``write_derivative_decision`` for shared reductions."""
    product = key2 * residual_norm2 * pre_norm2
    other = p_dot_e.square()
    scale = torch.maximum(product.abs(), other)
    return (scale > 0) & (product - other <= torch.finfo(product.dtype).eps ** .5 * scale)



@torch.no_grad()
def propagate_read_chunks(token_flags, period):
    """Carry events forward inside each period; return only read-half chunks.

    Accept [BH,N,C] token flags or [BH,N] chunk flags. Partial final periods
    are padded with false solely for grouping, then cropped to the input N.
    """
    if not isinstance(period, int) or period < 2:
        raise ValueError('period must be an integer >= 2')
    if token_flags.ndim == 3:
        flags = token_flags.any(-1)
    elif token_flags.ndim == 2:
        flags = token_flags
    else:
        raise ValueError('Expected [BH,N] or [BH,N,C] flags')
    chunks = flags.shape[-1]
    padded = torch.nn.functional.pad(flags, (0, (-chunks) % period))
    carried = padded.reshape(flags.shape[0], -1, period).long().cumsum(-1) > 0
    active = torch.arange(period, device=flags.device) >= period // 2
    return (carried & active).flatten(1)[:, :chunks]


@triton.jit
def _affine_combine(a_left, b_left, a_right, b_right):
    return a_right * a_left, a_right * b_left + b_right


@triton.jit
def _within_energy(INCREMENT, GC, PRE, END, N: tl.constexpr, C: tl.constexpr,
                   CB: tl.constexpr):
    index = tl.program_id(0)
    t = tl.arange(0, CB)
    g = tl.load(GC + index*C+t, t < C, 0).to(tl.float64)
    previous_g = tl.load(GC + index*C+t-1, (t > 0) & (t < C), 0).to(tl.float64)
    gain = libdevice.exp(2. * (g-previous_g)).to(tl.float32)
    gain = tl.where(t < C, gain, 1.)
    addition = tl.load(INCREMENT + index*C+t, t < C, 0).to(tl.float32)
    # Exclusive prefix of the state recurrence, followed by this token's decay.
    _, after = tl.associative_scan((gain, addition), 0, _affine_combine)
    before = tl.gather(after, tl.maximum(t-1, 0), 0)
    before = tl.where(t > 0, before, 0.) * gain
    tl.store(PRE + index*C+t, before, t < C)
    last = tl.sum(tl.where(t == C-1, after, 0.), 0)
    tl.store(END + index, last)


@triton.jit
def _energy_boundaries(END, GC, BOUNDARY, N: tl.constexpr, C: tl.constexpr,
                       PERIOD: tl.constexpr):
    head, group = tl.program_id(0), tl.program_id(1)
    state = tl.full((), 0., tl.float32)
    for offset in range(PERIOD):
        chunk = group*PERIOD+offset
        if chunk < N:
            index = head*N+chunk
            tl.store(BOUNDARY+index, state)
            if offset < PERIOD//2:
                g = tl.load(GC+index*C+C-1).to(tl.float64)
                state = libdevice.exp(2.*g).to(tl.float32)*state+tl.load(END+index)


@torch.no_grad()
def write_pre_norms(increments, gc, period):
    """Return ||alpha_t S_previous||² for each write-half token.

    Inputs are [BH,N,C], with cumulative log decay reset in every chunk.
    `increments` equals 2 beta p·e + beta² ||k||² ||e||². Boundary states
    reset at each period, and only its first floor(period/2) chunks write.
    Values in read-half chunks are zero and must not be used as state norms.
    No C×C products or per-token matrix states are constructed.
    """
    if increments.ndim != 3 or increments.shape != gc.shape:
        raise ValueError('increments and gc must share [BH,N,C] shape')
    if not isinstance(period, int) or period < 2:
        raise ValueError('period must be an integer >= 2')
    bh, n, c = increments.shape
    if n == 0 or c == 0:
        return torch.zeros_like(increments, dtype=torch.float32)
    increments, gc = increments.float().contiguous(), gc.contiguous()
    if increments.is_cuda:
        within = torch.empty_like(increments)
        end = increments.new_empty((bh, n))
        boundary = torch.empty_like(end)
        _within_energy[(bh*n,)](increments, gc, within, end, n, c,
                                triton.next_power_of_2(c), num_warps=4,
                                enable_fp_fusion=False)
        _energy_boundaries[(bh, triton.cdiv(n, period))](
            end, gc, boundary, n, c, period, num_warps=1, enable_fp_fusion=False)
        pre = within + (2*gc).exp().float()*boundary.unsqueeze(-1)
    else:
        state = increments.new_zeros((bh,))
        rows = []
        for chunk in range(n):
            if chunk % period == 0:
                state = torch.zeros_like(state)
            tokens = []
            for token in range(c):
                step = gc[:, chunk, token]
                if token:
                    step = step-gc[:, chunk, token-1]
                before = (2*step).exp().float()*state
                tokens.append(before)
                if chunk % period < period//2:
                    state = before+increments[:, chunk, token]
            rows.append(torch.stack(tokens, -1))
        pre = torch.stack(rows, 1)
    active = torch.arange(n, device=increments.device) % period < period//2
    return torch.where(active[None, :, None], pre, 0.)


@torch.compile(fullgraph=True)
def _write_statistics(prefix_projected, local_delta, k, v, beta, gc):
    # Fuse the vector reductions; only scalar arrays leave this computation.
    p = gc.exp().float().unsqueeze(-1)*prefix_projected.float()-local_delta.float()
    e = v.float()-p
    key2 = k.float().square().sum(-1)
    p_dot_e = (p*e).sum(-1)
    e2 = e.square().sum(-1)
    b = beta.float()
    increments = 2*b*p_dot_e+b.square()*key2*e2
    return increments, p_dot_e, e2, key2


@torch.no_grad()
def coarse_write_history(prefix_projected, local_delta, k, v, beta, gc, period):
    """Detect write-half radial derivatives and propagate to that bucket's reads.

    `prefix_projected` [BH,N,C,V] sums existing lower-level z@H values at each
    chunk boundary. Within a period's write half these lower buckets partition
    every earlier chunk. `local_delta` is the final sum of ALL active local
    residuals, including the highest local level: its sign is minus the local
    old-state projection. Thus p=exp(gc)*prefix_projected-local_delta.

    Return (read_chunk_flags [BH,N], write_token_flags [BH,N,C]). The caller
    retains the complete differentiable state history for any precise repair.
    """
    increments, p_dot_e, e2, key2 = _write_statistics(
        prefix_projected, local_delta, k, v, beta, gc)
    pre = write_pre_norms(increments, gc, period)
    flags = write_derivative_scalar(pre, p_dot_e, e2, key2)
    write_half = torch.arange(beta.shape[1], device=beta.device) % period < period//2
    flags = flags & write_half[None, :, None]
    return propagate_read_chunks(flags, period), flags
