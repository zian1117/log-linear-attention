"""Experimental coarse routing that reuses the state scan's projections."""
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


def projected_coarse(kn, wn, writes, end, rn, qn, k, beta, gc, u, temperature,
                     period, terms, mass_decay, mass_addition, floor, output_dtype,
                     return_masks=False, selected_inputs=None, state_pair=None):
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
    state,projected = (active_select(x,period) for x in (state,projected))
    rn,qn,k,beta,gc,u = (tuple(active_select(x,period) for x in (rn,qn,k,beta,gc,u))
                        if selected_inputs is None else selected_inputs)
    mass0 = active_select(boundary_mass(mass_decay,mass_addition,period),period)
    args = (projected,beta,k,gc,state,rn,qn,u,temperature,mass0,floor,output_dtype,return_masks)
    y,score,*diagnostic = _checkpoint_core(*args) if torch.is_grad_enabled() else _core(*args)
    if return_masks:
        diagnostic = [active_scatter(x,chunks,period) for x in diagnostic]
    return active_scatter(y,chunks,period),active_scatter(score,chunks,period),*diagnostic
