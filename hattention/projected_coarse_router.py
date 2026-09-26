"""Experimental coarse routing that reuses the state scan's projections."""
from functools import lru_cache
from types import FunctionType
import math

import torch
import torch.nn.functional as F
import triton

from .projected_bucket_states import ProjectedStates
from .fenwick_gather import active_select, active_scatter
from .decay_mass import boundary_mass, high_mass


def _torch_core(projected, beta, k, gc, state, rn, qn, u, temperature, mass0,
          floor, output_dtype, return_masks=False):
    from .fast_matrix_gdn import _score, _diagnostics
    coefficient = beta * (2-beta*k.square().sum(-1))
    energy = coefficient * projected.square().sum(-1)
    initial = state.square().sum((-2,-1)).unsqueeze(-1)
    gain2 = (2*gc).exp().to(state.dtype)
    norm2 = gain2*(initial-energy.cumsum(-1))
    with torch.no_grad():
        bound = gain2*(initial+energy.abs().cumsum(-1))
    y = (rn @ state).to(output_dtype)
    score = _score(qn @ state,u,norm2,temperature,floor)
    bad,nonfinite = _diagnostics(norm2,bound,high_mass(mass0,gc),score,y,floor)
    if return_masks:
        return y,score,bad,nonfinite
    return y,score,bad.flatten(1).any(-1)


# The fused autograd core already retains only the tensors needed by its
# backward. Checkpointing it again repeats those projections without reducing
# the dominant state storage.
from .fused_projected_coarse_router import fused_projected_coarse_core as _core
_checkpoint_core = torch.compile(_core, fullgraph=True, dynamic=True)


def _grouped_core_impl(*args):
    return _core(*args, raw_half=_GROUPED_RAW_HALF, head_groups=_GROUPED_HEAD_GROUPS)


# Compiled clones below replace these structural constants in their globals.
# Their independent code objects avoid exhausting one Dynamo cache across
# hierarchy periods and then again when the training batch size changes.
_GROUPED_RAW_HALF, _GROUPED_HEAD_GROUPS = 0, 1


@lru_cache(maxsize=None)
def _compiled_grouped_core(half, groups):
    name = f'_grouped_coarse_{half}_{groups}'
    namespace = dict(globals(), _GROUPED_RAW_HALF=half, _GROUPED_HEAD_GROUPS=groups)
    function = FunctionType(_grouped_core_impl.__code__.replace(co_name=name), namespace, name)
    return torch.compile(function, fullgraph=True, dynamic=True)


def _check_grouped_view(x, half):
    tail = x.shape[2:]
    width = math.prod(tail)
    expected = [2*half*width, width]
    for dimension in tail:
        width //= dimension
        expected.append(width)
    if x.shape[1] != half or any(size > 1 and actual != wanted
                               for size,actual,wanted in zip(x.shape,x.stride(),expected)):
        raise ValueError('Expected a grouped active-half view of contiguous raw chunks.')


def projected_coarse(kn, wn, writes, end, rn, qn, k, beta, gc, u, temperature,
                     period, terms, mass_decay, mass_addition, floor, output_dtype,
                     return_masks=False, selected_inputs=None, state_pair=None,
                     state_pair_active=False, selected_inputs_grouped=False,
                     radial_summaries=None, history_flags=None):
    radial = radial_summaries is not None
    if radial and not return_masks:
        raise ValueError('Radial diagnostics require per-token repair masks.')
    key_dim,value_dim = k.shape[-1],writes.shape[-1]
    kp = max(16,triton.next_power_of_2(key_dim))-key_dim
    vp = max(16,triton.next_power_of_2(value_dim))-value_dim
    if state_pair is None:
        z = terms[3]
        factor = beta*gc.exp().to(beta.dtype)
        state,projected = ProjectedStates.apply(
            F.pad(kn,(0,kp)) if kp else kn,
            F.pad(z,(0,kp)) if kp else z,
            factor,F.pad(writes,(0,vp)) if vp else writes,end,period)
        state,projected = state[...,:key_dim,:value_dim],projected[...,:value_dim]
    else:
        state,projected = state_pair
    chunks = k.shape[1]
    if state_pair is None or not state_pair_active:
        state,projected = (active_select(x,period) for x in (state,projected))
    if selected_inputs_grouped and (selected_inputs is None or chunks % period):
        raise ValueError('Grouped raw inputs require supplied complete-period views.')
    rn,qn,k,beta,gc,u = (tuple(active_select(x,period) for x in (rn,qn,k,beta,gc,u))
                        if selected_inputs is None else selected_inputs)
    mass0 = active_select(boundary_mass(mass_decay,mass_addition,period),period)
    if radial:
        from .radial_metadata import boundary_pair
        _, remainder = boundary_pair(mass_decay,*radial_summaries,period)
        remainder = active_select(remainder,period)
    if selected_inputs_grouped:
        batch, active = projected.shape[:2]
        half, groups = period//2, chunks//period
        for x in (rn,qn,k,beta,gc,u):
            if x.shape[:2] != (batch*groups,half):
                raise ValueError('Grouped raw input batch dimensions do not match the state.')
            _check_grouped_view(x,half)
        projected = projected.reshape(batch*groups,half,*projected.shape[2:])
        state = state.reshape(batch*groups,half,*state.shape[2:])
        mass0 = mass0.reshape(batch*groups,half)
        if radial:
            remainder = remainder.reshape(batch*groups,half)
    args = (projected,beta,k,gc,state,rn,qn,u,temperature,mass0,floor,output_dtype,return_masks)
    if radial:
        args = (*args, True)
    if selected_inputs_grouped:
        if torch.is_grad_enabled():
            y,score,*diagnostic = _compiled_grouped_core(half,groups)(*args)
        else:
            y,score,*diagnostic = _core(*args,raw_half=half,head_groups=groups)
    else:
        y,score,*diagnostic = _checkpoint_core(*args) if torch.is_grad_enabled() else _core(*args)
    if radial:
        from .radial_diagnostics import (
            mass_decision,erase_decision,dominance_decision,propagate_read_chunks,
        )
        metadata = diagnostic.pop()
        with torch.no_grad():
            norm2,raw,projection2,key2 = (metadata[...,i,:] for i in (0,1,3,4))
            before = raw + beta*(2-beta*key2)*projection2
            erase = erase_decision(norm2,before,projection2,key2,level=1,floor=floor)
            source = mass_decision(norm2,high_mass(mass0,gc),level=2,floor=floor)
            source = source | dominance_decision(norm2,high_mass(remainder,gc),floor)
            diagnostic[0] = diagnostic[0] | source
    if selected_inputs_grouped:
        y,score = (x.reshape(batch,active,*x.shape[2:]) for x in (y,score))
        if return_masks:
            diagnostic = [x.reshape(batch,active,*x.shape[2:]) for x in diagnostic]
        if radial:
            erase = erase.reshape(batch,active,*erase.shape[2:])
    if return_masks:
        diagnostic = [active_scatter(x,chunks,period) for x in diagnostic]
    if radial:
        with torch.no_grad():
            historical = propagate_read_chunks(active_scatter(erase,chunks,period),period)
            if history_flags is not None:
                historical = historical | history_flags
            diagnostic[0] = diagnostic[0] | historical.unsqueeze(-1)
    return active_scatter(y,chunks,period),active_scatter(score,chunks,period),*diagnostic
