"""Deterministic gather/scatter for each Fenwick period's second half.

Every selected chunk occurs once. The inverse gradient therefore uses a
plain copy with zeros for inactive chunks, never an atomic index reduction.
CUDA outputs are contiguous; a contiguous selected interval may alias input.
"""
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _select(SOURCE, OUTPUT, N: tl.constexpr, ACTIVE: tl.constexpr,
            WIDTH: tl.constexpr, PERIOD: tl.constexpr, TOTAL: tl.constexpr,
            BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    inner = offset % WIDTH
    chunk = (offset // WIDTH) % ACTIVE
    batch = offset // (WIDTH * ACTIVE)
    half = PERIOD // 2
    span = PERIOD - half
    original = (chunk // span) * PERIOD + half + chunk % span
    value = tl.load(SOURCE + (batch * N + original) * WIDTH + inner, offset < TOTAL, other=0)
    tl.store(OUTPUT + offset, value, offset < TOTAL)


@triton.jit
def _scatter(SOURCE, OUTPUT, N: tl.constexpr, ACTIVE: tl.constexpr,
             WIDTH: tl.constexpr, PERIOD: tl.constexpr, TOTAL: tl.constexpr,
             BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    inner = offset % WIDTH
    chunk = (offset // WIDTH) % N
    batch = offset // (WIDTH * N)
    half = PERIOD // 2
    selected = chunk % PERIOD >= half
    compact = (chunk // PERIOD) * (PERIOD - half) + chunk % PERIOD - half
    value = tl.load(SOURCE + (batch * ACTIVE + compact) * WIDTH + inner,
                    (offset < TOTAL) & selected, other=0)
    tl.store(OUTPUT + offset, value, offset < TOTAL)


def _count(chunks, period):
    half = period // 2
    return (chunks // period) * (period - half) + max(0, chunks % period - half)


def _validate(x, period):
    if x.ndim < 2:
        raise ValueError('Fenwick selection expects [batch, chunks, ...].')
    if not isinstance(period, int) or isinstance(period, bool) or period < 2:
        raise ValueError('Fenwick period must be an integer >= 2.')


def _gather(x, period):
    batch, chunks = x.shape[:2]
    active = _count(chunks, period)
    shape = (batch, active, *x.shape[2:])
    if not active or not x.numel():
        return x.new_empty(shape)
    # This case is a contiguous interval even across all leading dimensions.
    if batch == 1 and chunks <= period and x.is_contiguous():
        return x[:, period//2:]
    if not x.is_cuda:
        index = torch.arange(chunks, device=x.device)
        return x.index_select(1, index[index % period >= period//2])
    source = x.contiguous()
    result = x.new_empty(shape)
    _select[(triton.cdiv(result.numel(), 1024),)](
        source, result, chunks, active, math.prod(x.shape[2:]), period,
        result.numel(), 1024)
    return result


def _expand(x, chunks, period):
    shape = (x.shape[0], chunks, *x.shape[2:])
    if not math.prod(shape):
        return x.new_empty(shape)
    if not x.is_cuda:
        result = x.new_zeros(shape)
        index = torch.arange(chunks, device=x.device)
        return result.index_copy(1, index[index % period >= period//2], x)
    source = x.contiguous()
    result = x.new_empty(shape)
    _scatter[(triton.cdiv(result.numel(), 1024),)](
        source, result, chunks, x.shape[1], math.prod(x.shape[2:]), period,
        result.numel(), 1024)
    return result


class _ActiveSelect(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, period):
        ctx.chunks, ctx.period = x.shape[1], period
        return _gather(x, period)

    @staticmethod
    def backward(ctx, gradient):
        return _expand(gradient, ctx.chunks, ctx.period), None


class _ActiveScatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, chunks, period):
        ctx.period = period
        return _expand(x, chunks, period)

    @staticmethod
    def backward(ctx, gradient):
        return _gather(gradient, ctx.period), None, None


def active_select(x, period):
    """Select chunks i with i % period >= period//2 along dimension one."""
    _validate(x, period)
    return _ActiveSelect.apply(x, period)


def active_scatter(x, chunks, period):
    """Put compact active chunks into a full sequence; inactive chunks are zero."""
    _validate(x, period)
    if not isinstance(chunks, int) or isinstance(chunks, bool) or chunks < 0:
        raise ValueError('Chunk count must be a nonnegative integer.')
    if x.shape[1] != _count(chunks, period):
        raise ValueError('Compact chunk count does not match the requested sequence and period.')
    return _ActiveScatter.apply(x, chunks, period)
