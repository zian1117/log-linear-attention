"""Differentiable FP32 products for Hopper GDN chunk preparation.

Only the measured Hopper path uses the split BF16 expansion. CPU, FP64
precise repair, and other GPU architectures retain ordinary matrix products.
No global matmul precision setting is changed.
"""
import torch

from .projected_state_tree_kernels import mm, _use_split_bf16


def _product_impl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return mm(a, b)


try:
    _product = torch.ops.hattention.preparation_product.default
except AttributeError:
    _product = torch.library.custom_op(
        'hattention::preparation_product', _product_impl, mutates_args=())

    @_product.register_fake
    def _product_fake(a, b):
        return a.new_empty((*a.shape[:-1], b.shape[-1]))

    def _setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs)

    def _backward(ctx, gradient):
        a, b = ctx.saved_tensors
        if any(stride <= 0 for stride in gradient.stride()):
            gradient = gradient.contiguous()
        return (
            _product(gradient, b.transpose(-1, -2)) if ctx.needs_input_grad[0] else None,
            _product(a.transpose(-1, -2), gradient) if ctx.needs_input_grad[1] else None,
        )

    _product.register_autograd(_backward, setup_context=_setup_context)


def preparation_product(a, b):
    if (a.is_cuda and a.dtype == b.dtype == torch.float32
            and _use_split_bf16(a.device)):
        return _product(a, b)
    return a @ b
