"""Fuse vector normalization and the router's chunk layout.

Outputs remain FP32. A single source promotion feeds every normalization
branch, and AOT autograd sums every consumer before the one input-dtype cast.
No hand-derived Jacobian, changed epsilon convention, or adaptive-head edits.
"""
from functools import lru_cache
from types import FunctionType
import torch
import torch.nn.functional as F


def _normalize_chunks(x, chunk, kind, eps):
    torch._dynamo.mark_static(x, -1)
    batch, length, heads, dimension = x.shape
    value = x.float()
    if kind == 'gdn':
        value = value * torch.rsqrt(value.square().sum(-1, keepdim=True) + eps)
    elif kind == 'l2':
        value = F.normalize(value, dim=-1, eps=eps)
    padded = ((length + chunk - 1) // chunk) * chunk
    if padded != length:
        value = F.pad(value, (0, 0, 0, 0, 0, padded - length))
    return value.reshape(batch, padded // chunk, chunk, heads, dimension).movedim(3, 1).reshape(
        batch * heads, padded // chunk, chunk, dimension).contiguous()


@lru_cache(maxsize=None)
def _compiled(chunk, dimension, dtype, kind, eps, backend):
    # Independent static-specialization cache entries prevent the global
    # per-code-object cache limit from conflating different norm conventions.
    name = '_normalize_chunks_' + repr((chunk, dimension, dtype, kind, eps, backend))
    function = FunctionType(_normalize_chunks.__code__.replace(co_name=name), globals(), name)
    return torch.compile(function, fullgraph=True, dynamic=True, backend=backend)


def normalized_chunks(x, chunk, kind, eps=1e-6, *, backend='inductor'):
    if x.ndim != 4 or min(x.shape) <= 0:
        raise ValueError('Expected nonempty [batch,tokens,heads,dimension] vectors')
    if kind not in ('gdn','l2','none') or not isinstance(chunk,int) or chunk <= 0:
        raise ValueError('Choose gdn/l2/none and a positive computational chunk size')
    return _compiled(chunk, x.shape[-1], x.dtype, kind, eps, backend)(x, chunk, kind, eps)
