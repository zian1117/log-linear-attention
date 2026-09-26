"""Experimental joint local routing with shared Fenwick residuals."""
from functools import lru_cache
from types import FunctionType

import torch
from torch.utils.checkpoint import checkpoint

from .joint_local_blocks import joint_blocks
from .shared_local_norm import _shared_core as _shared_norm_core
from .current_bucket_score import current_bucket_score, previous_bucket_score

_shared_norm_core = getattr(_shared_norm_core, '_torchdynamo_orig_callable', _shared_norm_core)


def _all_local(ar, aq, k, v, beta, u, q, temperature, inverse, decay,
               floor, output_dtype, levels, use_fused, return_metadata=False):
    from .fast_matrix_gdn import _score, _diagnostics
    torch._dynamo.mark_static(k,-2)
    chunk = k.shape[-2]
    periods = tuple(1 << level for level in range(1, levels))
    ar_blocks = joint_blocks(ar, periods)
    aq_blocks = joint_blocks(aq, periods)
    inverse_blocks = joint_blocks(inverse, periods)
    decay_blocks_joint = joint_blocks(decay, periods)
    projections, residuals, decay_blocks = [], [], []
    with torch.no_grad():
        positive_mass = beta*k.norm(dim=-1)*v.norm(dim=-1)
    for level in range(1, levels):
        period = 1 << level
        half = period//2
        shape = (*beta.shape[:-1], chunk//period, period)
        value = v.reshape(*shape, v.shape[-1])[..., :half, :]
        dr = ar_blocks[level-1][..., half:, :half]
        dq = aq_blocks[level-1][..., half:, :half]
        inv = inverse_blocks[level-1]
        dec = decay_blocks_joint[level-1].contiguous()
        residual_coefficients = inv[..., half:, :half]*dec[..., half:, :half]
        if use_fused:
            from .local_bucket_projections import local_bucket_projections
            y, read, residual = local_bucket_projections(dr,dq,residual_coefficients,value)
        else:
            y, read, residual = dr@value, dq@value, residual_coefficients@value
        projections.append((y,read))
        residuals.append(residual)
        decay_blocks.append(dec)
    norm_result = _shared_norm_core(k,v,beta,inverse_blocks,decay_blocks_joint,levels,tuple(residuals), return_metadata)
    if return_metadata:
        norms, prefix_delta, write_flags = norm_result
    else:
        norms = norm_result
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
        if period == 2:
            score = previous_bucket_score(
                k.reshape(*shape,k.shape[-1]), v.reshape(*shape,v.shape[-1]),
                beta.reshape(shape), decay_blocks[level-1],
                u.reshape(*shape,u.shape[-1]), q.reshape(*shape,q.shape[-1]),
                temperature.unsqueeze(-1),floor,norm2)
        else:
            score = _score(read,query,norm2,temperature.unsqueeze(-1),floor)
        with torch.no_grad():
            mass = (decay_blocks[level-1][...,half:,:half]
                    @ positive_mass.reshape(shape)[...,:half].unsqueeze(-1)).squeeze(-1)
        bad,nonfinite = _diagnostics(norm2,bound,mass,score,y,floor)
        if return_metadata:
            from .radial_metadata import positive_pair
            from .radial_diagnostics import mass_decision, erase_decision, dominance_decision
            with torch.no_grad():
                key2 = k.square().sum(-1).reshape(shape)[...,half:]
                b = beta.reshape(shape)[...,half:]
                projection2 = residuals[level-1].square().sum(-1)
                before = norm2 + b*(2-b*key2)*projection2
                erase = erase_decision(norm2,before,projection2,key2,level=level,floor=floor)
                source = mass_decision(norm2,mass,level=level,floor=floor)
                if level >= 2:
                    contributions = (decay_blocks[level-1][...,half:,:half]
                        * positive_mass.reshape(shape)[...,:half].unsqueeze(-2))
                    _, remainder = positive_pair(contributions)
                    source = source | dominance_decision(norm2,remainder,floor)
                history = write_flags[level-1].any(-1,keepdim=True).expand_as(bad)
                bad = bad | source | erase | history
        y = torch.cat((torch.zeros_like(y),y),-2).flatten(-3,-2)
        score,bad,nonfinite = (torch.cat((torch.zeros_like(x),x),-1).flatten(-2,-1)
                              for x in (score,bad,nonfinite))
        result.extend((y,score,bad,nonfinite))
    if return_metadata:
        result.append(prefix_delta)
    return tuple(result)


def _checkpoint_impl(*args):
    return checkpoint(_all_local,*args,use_reentrant=False)


@lru_cache(maxsize=None)
def _compiled_shape(chunk,key_dim,value_dim,input_dtype,output_dtype,floor,levels,use_fused,return_metadata):
    shape = (chunk,key_dim,value_dim,input_dtype,output_dtype,floor,levels,use_fused,return_metadata)
    name = '_shared_local_checkpoint_'+repr(shape)
    function = FunctionType(_checkpoint_impl.__code__.replace(co_name=name),globals(),name)
    return torch.compile(function,fullgraph=True,dynamic=True)


def shared_local_router(ar,aq,k,v,beta,gc,u,temperature,terms,floor,output_dtype,levels,
                        *, q, use_fused=False, return_metadata=False):
    args = (ar,aq,k,v,beta,u,q,temperature,terms[1],terms[2],floor,output_dtype,levels,use_fused,return_metadata)
    if torch.is_grad_enabled():
        values = _compiled_shape(k.shape[-2],k.shape[-1],v.shape[-1],k.dtype,output_dtype,
                                 floor,levels,use_fused,return_metadata)(*args)
    else:
        values = _all_local(*args)
    if return_metadata:
        return (*tuple(list(values[:-1][offset::4]) for offset in range(4)), values[-1])
    return tuple(list(values[offset::4]) for offset in range(4))
