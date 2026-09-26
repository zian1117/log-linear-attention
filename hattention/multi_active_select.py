"""Select several Fenwick levels with one deterministic gradient accumulation.

Forward uses the existing compact gathers. Backward reads those compact
gradients directly and writes one full input gradient, avoiding a separate
dense scatter and gradient-add for every level.
"""
import math

import torch
import triton
import triton.language as tl

from .fenwick_gather import _count, _gather, _validate


@triton.jit
def _sum_selected(GRADIENTS, OUTPUT, N: tl.constexpr, WIDTH: tl.constexpr,
                  TOTAL: tl.constexpr, PERIODS,
                  COUNTS, DOUBLE: tl.constexpr,
                  BLOCK: tl.constexpr):
    offset = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    inner = offset % WIDTH
    chunk = (offset//WIDTH) % N
    batch = offset//(WIDTH*N)
    dtype = tl.float64 if DOUBLE else tl.float32
    value = tl.full((BLOCK,), 0, dtype)
    for index in tl.static_range(len(PERIODS)):
        period = PERIODS[index]
        half = period//2
        selected = chunk % period >= half
        compact = (chunk//period)*(period-half)+chunk%period-half
        gradient = tl.load(
            GRADIENTS[index]+(batch*COUNTS[index]+compact)*WIDTH+inner,
            (offset < TOTAL) & selected, other=0,
        ).to(dtype)
        value = value+gradient
    tl.store(OUTPUT+offset, value, offset < TOTAL)


def _sum_compact_impl(sources: list[torch.Tensor], periods: list[int],
                      shape: list[int]) -> torch.Tensor:
    """Opaque compiled-backward boundary for Triton's pointer-tuple launch."""
    sources = tuple(source.contiguous() for source in sources)
    result = sources[0].new_empty(shape)
    chunks = shape[1]
    counts = tuple(_count(chunks, period) for period in periods)
    _sum_selected[(triton.cdiv(result.numel(), 1024),)](
        sources, result, chunks, math.prod(shape[2:]), result.numel(),
        tuple(periods), counts, result.dtype == torch.float64, 1024, num_warps=4,
    )
    return result


# Tests can import this standalone helper without importing the whole model
# package. Reuse the registration if both import routes occur in one process.
try:
    _sum_compact = torch.ops.hattention.multi_active_select_backward.default
except AttributeError:
    _sum_compact = torch.library.custom_op(
        'hattention::multi_active_select_backward', _sum_compact_impl, mutates_args=(),
    )

    @_sum_compact.register_fake
    def _sum_compact_fake(sources, periods, shape):
        return sources[0].new_empty(shape)


class _MultiActiveSelect(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, periods):
        ctx.input_shape = x.shape
        ctx.input_dtype, ctx.input_device = x.dtype, x.device
        ctx.periods = periods
        # Missing downstream gradients must remain None, not dense zeros.
        ctx.set_materialize_grads(False)
        source = x.contiguous()
        return tuple(_gather(source, period) for period in periods)

    @staticmethod
    def backward(ctx, *gradients):
        supplied = tuple((period, gradient) for period, gradient in zip(ctx.periods, gradients)
                         if gradient is not None and gradient.numel())
        if not supplied:
            return torch.zeros(ctx.input_shape, dtype=ctx.input_dtype, device=ctx.input_device), None
        chunks = ctx.input_shape[1]
        periods, sources = zip(*supplied)
        if ctx.input_device.type != 'cuda':
            accumulation_dtype = torch.float64 if ctx.input_dtype == torch.float64 else torch.float32
            summed = torch.zeros(ctx.input_shape, dtype=accumulation_dtype, device=ctx.input_device)
            for period, source in supplied:
                indices = torch.arange(chunks, device=source.device)
                indices = indices[indices % period >= period//2]
                summed.index_add_(1, indices, source.to(accumulation_dtype))
            return summed.to(ctx.input_dtype), None
        return _sum_compact(list(sources), list(periods), list(ctx.input_shape)), None


def multi_active_select(x, periods):
    """Return one compact selection per period, in the requested order.

Each result selects chunk indices i satisfying i % period >= period//2.
Duplicate periods are allowed; their gradient contributions are added.
"""
    periods = tuple(periods)
    if x.ndim < 2:
        raise ValueError('Fenwick selection expects [batch, chunks, ...].')
    for period in periods:
        _validate(x, period)
    return _MultiActiveSelect.apply(x, periods)
