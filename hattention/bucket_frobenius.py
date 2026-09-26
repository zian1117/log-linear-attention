"""Exact bucket squared Frobenius norms from chunk-sized Gram matrices.

The ordinary path avoids materializing matrices at every token. The Gram
expansion contains subtractions, so its arithmetic uses FP64, with a direct
recurrent fallback restricted to segments that suffer severe cancellation.
The caller applies the configured squared norm floor *before* taking a square
root. These functions preserve gradients through both writes and erasure.
"""

import torch


def _terms(k, beta, cumulative_decay):
    length = k.shape[-2]
    gram = k @ k.transpose(-1, -2)
    eye = torch.eye(length, dtype=k.dtype, device=k.device)
    system = eye + (beta.unsqueeze(-1) * gram).tril(-1)
    inverse = torch.linalg.solve_triangular(
        system, eye.expand_as(system), upper=False, unitriangular=True
    )
    causal = torch.ones((length, length), dtype=torch.bool, device=k.device).tril()
    difference = cumulative_decay.unsqueeze(-1) - cumulative_decay.unsqueeze(-2)
    decay = difference.masked_fill(~causal, -torch.inf).exp()
    return gram, inverse, decay


def _write_norm2(gram, residual, decay):
    # Prefix norm recurrence, expanded into a triangular matrix multiplication.
    # Every exponent in decay is nonpositive for nonpositive GDN log decays.
    product = gram * (residual @ residual.transpose(-1, -2))
    increments = product.diagonal(dim1=-2, dim2=-1)
    increments = increments + 2 * (decay * product).tril(-1).sum(-1)
    norm2 = (decay.square() @ increments.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        absolute_increments = product.diagonal(dim1=-2, dim2=-1).abs()
        absolute_increments = absolute_increments + 2 * (decay * product).tril(-1).abs().sum(-1)
        bound = (decay.square() @ absolute_increments.unsqueeze(-1)).squeeze(-1)
    return norm2, bound


def _repair_cancellation(norm2, bound, k, beta, cumulative_decay, *, v=None, state=None, norm_floor=0.):
    """Recompute ill-conditioned segments directly, preserving their gradients.

    A Gram expansion loses relative accuracy when its answer is much smaller
    than the magnitudes being added. At the square-root-of-machine-epsilon
    threshold, recompute the complete affected segment from raw inputs. The
    ordinary path stores only scalar norms; the rare fallback has at most one
    chunk of recurrent matrices per affected segment in its autograd graph.
    """
    with torch.no_grad():
        threshold = torch.finfo(norm2.dtype).eps ** 0.5
        # Below the caller's denominator floor, the norm has no derivative
        # in the score. Avoid reconstructing matrices just to resolve tiny
        # (including subnormal) norms that cannot affect that denominator.
        scale = norm2.clamp_min(norm_floor*norm_floor) if norm_floor else norm2
        selected = ((scale <= threshold * bound) & (bound > 0)).any(-1).reshape(-1)
    if not selected.any().item():
        return norm2
    indices = selected.nonzero(as_tuple=False).flatten()
    length, key_dim = k.shape[-2:]
    key = k.reshape(-1, length, key_dim).index_select(0, indices)
    b = beta.reshape(-1, length).index_select(0, indices)
    gc = cumulative_decay.reshape(-1, length).index_select(0, indices)
    if v is not None:
        value = v.reshape(-1, length, v.shape[-1]).index_select(0, indices)
        current = k.new_zeros((indices.numel(), key_dim, v.shape[-1]))
    else:
        value = None
        current = state.reshape(-1, key_dim, state.shape[-1]).index_select(0, indices)
    squared_norms = []
    previous_decay = torch.zeros_like(gc[:, 0])
    for position in range(length):
        step_decay = (gc[:, position] - previous_decay).exp()
        previous_decay = gc[:, position]
        current = current * step_decay[:, None, None]
        token_key = key[:, position]
        residual = -(token_key.unsqueeze(-2) @ current).squeeze(-2)
        if value is not None and position < length // 2:
            residual = residual + value[:, position]
        current = current + token_key.unsqueeze(-1) * (b[:, position, None] * residual).unsqueeze(-2)
        squared_norms.append(current.square().sum((-2, -1)))
    corrected = torch.stack(squared_norms, dim=-1)
    return norm2.reshape(-1, length).index_copy(0, indices, corrected).reshape_as(norm2)


def _segment_blocks(matrix, period):
    """Extract contiguous diagonal blocks without copying the whole matrix."""
    groups = matrix.shape[-1] // period
    blocks = matrix.reshape(*matrix.shape[:-2], groups, period, groups, period)
    return blocks.diagonal(dim1=-4, dim2=-2).movedim(-1, -3)


@torch.compile
def _local_expansion(gram, inverse, decay, beta, value, writes):
    residual = (inverse * decay) @ (beta.unsqueeze(-1) * value * writes.unsqueeze(-1))
    return _write_norm2(gram, residual, decay)


@torch.compile
def _high_expansion(gram, decay, w, k, state, cumulative_decay):
    gain = cumulative_decay.exp()
    residual = -gain.unsqueeze(-1) * (w @ state)
    cross = ((k @ state) * residual).sum(-1)
    boundary = gain.square() * state.square().sum((-2, -1)).unsqueeze(-1)
    boundary_cross = 2 * gain * (decay @ cross.unsqueeze(-1)).squeeze(-1)
    written_norm2, bound = _write_norm2(gram, residual, decay)
    norm2 = boundary + boundary_cross + written_norm2
    with torch.no_grad():
        bound = bound + boundary.abs() + 2 * gain * (decay @ cross.abs().unsqueeze(-1)).squeeze(-1)
    return norm2, bound


def local_bucket_norm2(k, v, beta, cumulative_decay, level, *, terms=None, norm_floor=0.):
    """Squared norms for a local Fenwick level; inputs end in [C,K/V].

    At levels above zero, each period starts from zero, writes during its first
    half, then applies only erasure during its second half. Only second-half
    positions are active buckets; the router masks the other positions.
    """
    if level < 0:
        raise ValueError("Bucket level must be nonnegative")
    k, v, beta, cumulative_decay = (
        x.double() for x in (k, v, beta, cumulative_decay)
    )
    if level == 0:
        return beta.square() * k.square().sum(-1) * v.square().sum(-1)
    length = k.shape[-2]
    period = 1 << level
    if period > length or length % period:
        raise ValueError("Local bucket period must divide the chunk length")
    scalar_shape = (*beta.shape[:-1], length // period, period)
    key = k.reshape(*scalar_shape, k.shape[-1])
    value = v.reshape(*scalar_shape, v.shape[-1])
    b = beta.reshape(scalar_shape)
    gc = cumulative_decay.reshape(scalar_shape)
    if terms is None:
        gram, inverse, decay = _terms(key, b, gc)
    else:
        # Contiguous diagonal blocks of a triangular inverse equal the inverse
        # of the corresponding diagonal blocks of its triangular system.
        gram, inverse, decay = (
            _segment_blocks(x.double(), period) for x in terms[:3]
        )
    writes = torch.arange(period, device=k.device) < period // 2
    norm2, bound = _local_expansion(gram, inverse, decay, b, value, writes)
    norm2 = _repair_cancellation(norm2, bound, key, b, gc, v=value, norm_floor=norm_floor)
    return norm2.reshape_as(beta)


def high_bucket_norm2(k, beta, cumulative_decay, state, *, terms=None, norm_floor=0.):
    """Squared norms after erasing a boundary [K,V] state within each chunk.

    No new value writes belong to these older buckets. cumulative_decay is the
    cumulative log decay measured from the start of the chunk.
    """
    k, beta, cumulative_decay, state = (
        x.double() for x in (k, beta, cumulative_decay, state)
    )
    if terms is None:
        gram, inverse, decay = _terms(k, beta, cumulative_decay)
        w = inverse @ (beta.unsqueeze(-1) * k)
    else:
        gram, _, decay, w = (x.double() for x in terms)
    norm2, bound = _high_expansion(gram, decay, w, k, state, cumulative_decay)
    return _repair_cancellation(norm2, bound, k, beta, cumulative_decay, state=state, norm_floor=norm_floor)
