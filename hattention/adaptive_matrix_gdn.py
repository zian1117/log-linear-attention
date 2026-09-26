"""Selective precision for the bilinear matrix router.

Conditioning flags detect cancellation and possible historical rounding error;
they are not certified bounds for the complete model's loss gradient.
"""

import torch

from .bilinear_matrix_gdn import precise_bilinear_matrix_gdn as _precise
from .fast_matrix_gdn import fast_matrix_gdn as _fast


class _DiscardFlaggedGradients(torch.autograd.Function):
    """Isolate replaced heads before their gradients reach shared projections.

    Selecting replacement outputs alone is insufficient: a discarded branch
    can compute NaN * 0 during backward. All router operations are independent
    between heads. Masking at its inputs therefore discards every gradient
    from a replaced head, including NaNs, without changing retained heads.

    `selection` is a private holder set exactly once after the fast forward.
    Unlike a saved tensor, it is intentionally not a snapshot of the initial
    selection. No caller-visible tensor is mutated.
    """

    @staticmethod
    def forward(ctx, selection, *inputs):
        ctx.selection = selection
        return tuple(x.view_as(x) for x in inputs)

    @staticmethod
    def backward(ctx, *gradients):
        replaced = ctx.selection[0]
        result = []
        for grad in gradients:
            if grad is None:
                result.append(None)
            else:
                # Token inputs have [1,T,batch_heads,...]; temperatures have
                # [batch_heads]. Flattening batch into heads prevents their
                # shared original temperature from mixing good/bad gradients.
                shape = (1, 1, -1) + (1,) * (grad.ndim - 3) if grad.ndim > 1 else (-1,)
                result.append(torch.where(replaced.reshape(shape), 0, grad))
        return (None, *result)


def adaptive_matrix_gdn(r, k, v, g, beta, u, q, log_temperature,
                        norm_floor=1e-6, vector_eps=1e-6, repair_periods=False,
                        shared_local=False, fused_local=False, repair_chunks=False,
                        shared_gathers=False, compact_cache_reads=False, shared_states=False,
                        radial_guards=False):
    """Evaluate FP32, replacing all tokens of flagged batch/head pairs.

    Recomputing the whole head preserves its history, including matrix error
    created by an earlier erase. Tiny buckets are never removed from softmax.
    """
    batch, length, heads = k.shape[:3]

    def flatten_heads(x):
        return x.transpose(1, 2).reshape(batch * heads, length, *x.shape[3:]).transpose(0, 1).unsqueeze(0)

    inputs = tuple(flatten_heads(x) for x in (r, k, v, g, beta, u, q))
    inputs += (log_temperature.unsqueeze(0).expand(batch, -1).reshape(-1),)
    selection = [None]
    isolated = _DiscardFlaggedGradients.apply(selection, *inputs)
    out, flagged = _fast(*isolated, norm_floor=norm_floor, vector_eps=vector_eps,
                         repair_periods=repair_periods,shared_local=shared_local,fused_local=fused_local,
                         repair_chunks=repair_chunks,shared_gathers=shared_gathers,
                         compact_cache_reads=compact_cache_reads,shared_states=shared_states,
                         radial_guards=radial_guards)
    selection[0] = flagged
    if flagged.any().item():
        selected = flagged.nonzero(as_tuple=False).flatten()
        precise_inputs = tuple(x.index_select(2, selected) for x in inputs[:-1])
        precise_inputs += (inputs[-1].index_select(0, selected),)
        replacement = _precise(*precise_inputs, norm_floor=norm_floor, vector_eps=vector_eps)
        out = out.index_copy(2, selected, replacement)
    return out.squeeze(0).transpose(0, 1).reshape(batch, heads, length, v.shape[-1]).transpose(1, 2)
