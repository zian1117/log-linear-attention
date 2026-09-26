"""Experimental joint local routing with shared Fenwick residuals."""
from functools import lru_cache
from types import FunctionType

import torch
from torch.utils.checkpoint import checkpoint

from .bucket_frobenius import _segment_blocks
from .shared_local_norm import _shared_core as _shared_norm_core
from .current_bucket_score import current_bucket_score

_shared_norm_core = getattr(_shared_norm_core, '_torchdynamo_orig_callable', _shared_norm_core)


def _all_local(ar, aq, k, v, beta, u, q, temperature, inverse, decay,
               floor, output_dtype, levels, use_fused):
    from .fast_matrix_gdn import _score, _diagnostics
    torch._dynamo.mark_static(k,-2)
    chunk = k.shape[-2]
    projections, residuals, decay_blocks = [], [], []
    with torch.no_grad():
        positive_mass = beta*k.norm(dim=-1)*v.norm(dim=-1)
    for level in range(1, levels):
        period = 1 << level
        half = period//2
        shape = (*beta.shape[:-1], chunk//period, period)
        value = v.reshape(*shape, v.shape[-1])[..., :half, :]
        dr = _segment_blocks(ar,period)[..., half:, :half]
        dq = _segment_blocks(aq,period)[..., half:, :half]
        inv = _segment_blocks(inverse,period)
        dec = _segment_blocks(decay,period).contiguous()
        residual_coefficients = inv[..., half:, :half]*dec[..., half:, :half]
        if use_fused:
            from .local_bucket_projections import local_bucket_projections
            y, read, residual = local_bucket_projections(dr,dq,residual_coefficients,value)
        else:
            y, read, residual = dr@value, dq@value, residual_coefficients@value
        projections.append((y,read))
        residuals.append(residual)
        decay_blocks.append(dec)
    norms = _shared_norm_core(k,v,beta,inverse,decay,levels,tuple(residuals))
    result = []
    y = (ar.diagonal(dim1=-2,dim2=-1).unsqueeze(-1)*v).to(output_dtype)
    norm2,bound = norms[0]
    score = current_bucket_score(k,v,beta,u,q,temperature,floor,norm2)
    bad,nonfinite = _diagnostics(norm2,bound,norm2.clamp_min(0).sqrt(),score,y,floor)
    result.extend((y,score,bad,nonfinite))
    for level in range(1,levels):
        period = 1 << level
        half = period//2
        shape = (*beta.shape[:-1],chunk//period,period)
        norm2,bound = (x.reshape(shape)[...,half:] for x in norms[level])
        y,read = projections[level-1]
        y = y.to(output_dtype)
        query = u.reshape(*shape,u.shape[-1])[...,half:,:]
        score = _score(read,query,norm2,temperature.unsqueeze(-1),floor)
        with torch.no_grad():
            mass = (decay_blocks[level-1][...,half:,:half]
                    @ positive_mass.reshape(shape)[...,:half].unsqueeze(-1)).squeeze(-1)
        bad,nonfinite = _diagnostics(norm2,bound,mass,score,y,floor)
        y = torch.cat((torch.zeros_like(y),y),-2).flatten(-3,-2)
        score,bad,nonfinite = (torch.cat((torch.zeros_like(x),x),-1).flatten(-2,-1)
                              for x in (score,bad,nonfinite))
        result.extend((y,score,bad,nonfinite))
    return tuple(result)


def _checkpoint_impl(*args):
    return checkpoint(_all_local,*args,use_reentrant=False)


@lru_cache(maxsize=None)
def _compiled_shape(chunk,key_dim,value_dim,input_dtype,output_dtype,floor,levels,use_fused):
    shape = (chunk,key_dim,value_dim,input_dtype,output_dtype,floor,levels,use_fused)
    name = '_shared_local_checkpoint_'+repr(shape)
    function = FunctionType(_checkpoint_impl.__code__.replace(co_name=name),globals(),name)
    return torch.compile(function,fullgraph=True,dynamic=True)


def shared_local_router(ar,aq,k,v,beta,gc,u,temperature,terms,floor,output_dtype,levels,
                        *, q, use_fused=False):
    args = (ar,aq,k,v,beta,u,q,temperature,terms[1],terms[2],floor,output_dtype,levels,use_fused)
    if torch.is_grad_enabled():
        values = _compiled_shape(k.shape[-2],k.shape[-1],v.shape[-1],k.dtype,output_dtype,
                                 floor,levels,use_fused)(*args)
    else:
        values = _all_local(*args)
    return tuple(list(values[offset::4]) for offset in range(4))
