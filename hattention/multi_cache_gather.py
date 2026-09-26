"""Gather several row selections while accumulating one cache gradient."""
import torch


class _GatherRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, *indices):
        ctx.save_for_backward(*indices)
        ctx.shape = value.shape
        ctx.dtype, ctx.device = value.dtype, value.device
        ctx.set_materialize_grads(False)
        return tuple(value.index_select(0, index.flatten()).reshape((*index.shape, *value.shape[1:]))
                     for index in indices)

    @staticmethod
    def backward(ctx, *gradients):
        # All levels meet here before the gradient reaches any lower-precision
        # source. Repeated indices (including padding) are accumulated safely.
        result = torch.zeros(ctx.shape, dtype=ctx.dtype, device=ctx.device)
        for index, gradient in zip(ctx.saved_tensors, gradients):
            if gradient is not None:
                result.index_add_(0, index.flatten(), gradient.reshape(-1, *ctx.shape[1:]))
        return (result,) + (None,) * len(gradients)


def gather_rows(value, indices):
    """Each output has indices.shape + value.shape[1:]; duplicates are valid."""
    indices = tuple(indices)
    return _GatherRows.apply(value, *indices) if indices else ()

