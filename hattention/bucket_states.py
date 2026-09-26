"""FP64 chunk-boundary Fenwick states with recomputation in backward.

Only chunk-boundary matrices are retained while evaluating a hierarchy level.
The recurrence is batched across independent periods. Unlike a token recurrence,
each step processes a complete chunk with matrix multiplications.
"""

import torch
import torch.nn.functional as F


def _group_inputs(k, w, writes, decay, period):
    chunks = k.shape[1]
    width = min(period, chunks)
    groups = (chunks + width - 1) // width
    padding = groups * width - chunks
    if padding:
        k, w, writes = (F.pad(x, (0, 0, 0, 0, 0, padding)) for x in (k, w, writes))
        decay = F.pad(decay, (0, padding), value=1)
    k, w, writes = (
        x.reshape(x.shape[0], groups, width, *x.shape[2:]) for x in (k, w, writes)
    )
    return k, w, writes, decay.reshape(decay.shape[0], groups, width)


def _scan_states(k, w, writes, decay, period):
    current = k.new_zeros((*k.shape[:2], k.shape[-1], writes.shape[-1]))
    states = []
    for position in range(k.shape[2]):
        states.append(current)
        if position + 1 < k.shape[2]:
            residual = -(w[:, :, position] @ current)
            if position < period // 2:
                residual = residual + writes[:, :, position]
            current = (
                decay[:, :, position, None, None] * current
                + k[:, :, position].transpose(-1, -2) @ residual
            )
    return torch.stack(states, dim=2)


class BucketStates(torch.autograd.Function):
    """Return pre-chunk states; writes occur only in each period's first half."""

    @staticmethod
    def forward(ctx, k, w, writes, decay, period):
        if not isinstance(period, int) or period < 2:
            raise ValueError("Boundary-state period must be an integer >= 2")
        if k.ndim != 4 or k.shape[1] == 0:
            raise ValueError("Expected nonempty [batch_heads,chunks,chunk,key_dim] keys")
        ctx.input_dtypes = tuple(x.dtype for x in (k, w, writes, decay))
        k, w, writes, decay = (x.double() for x in (k, w, writes, decay))
        ctx.save_for_backward(k, w, writes, decay)
        ctx.period = period
        grouped = _group_inputs(k, w, writes, decay, period)
        states = _scan_states(*grouped, period)
        return states.flatten(1, 2)[:, :k.shape[1]].contiguous()

    @staticmethod
    def backward(ctx, direct):
        raw_k, raw_w, raw_writes, raw_decay = ctx.saved_tensors
        period = ctx.period
        k, w, writes, decay = _group_inputs(raw_k, raw_w, raw_writes, raw_decay, period)
        states = _scan_states(k, w, writes, decay, period)
        chunks = raw_k.shape[1]
        padding = k.shape[1] * k.shape[2] - chunks
        direct = F.pad(direct.double(), (0, 0, 0, 0, 0, padding)).reshape_as(states)
        dk, dw, du, da = (torch.zeros_like(x) for x in (k, w, writes, decay))
        adjoint = torch.zeros_like(states[:, :, 0])
        for position in range(k.shape[2] - 1, -1, -1):
            state = states[:, :, position]
            key, weight = k[:, :, position], w[:, :, position]
            residual = -(weight @ state)
            if position < period // 2:
                residual = residual + writes[:, :, position]
            dresidual = key @ adjoint
            dk[:, :, position] = residual @ adjoint.transpose(-1, -2)
            dw[:, :, position] = -(dresidual @ state.transpose(-1, -2))
            if position < period // 2:
                du[:, :, position] = dresidual
            da[:, :, position] = (state * adjoint).sum((-2, -1))
            adjoint = (
                decay[:, :, position, None, None] * adjoint
                - weight.transpose(-1, -2) @ dresidual
                + direct[:, :, position]
            )
        gradients = (x.flatten(1, 2)[:, :chunks] for x in (dk, dw, du, da))
        return tuple(x.to(dtype) for x, dtype in zip(gradients, ctx.input_dtypes)) + (None,)
