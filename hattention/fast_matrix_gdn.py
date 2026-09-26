"""Experimental FP32 route with explicit per-head conditioning diagnostics.

The caller must recompute flagged heads from original inputs in the precise
implementation. This module alone is not the public numerical contract.
"""
import torch
import torch.nn.functional as F
import triton
from functools import lru_cache
from types import FunctionType
from torch.utils.checkpoint import checkpoint

from .bilinear_matrix_gdn import _RoutingReduce, _CHUNK_SIZE, _gdn_normalize
from .bucket_frobenius import _segment_blocks
from .energy_bucket_norm import _local_energy, _high_energy
from .fenwick_gather import active_select, active_scatter

# Compile the checkpoint boundary as a whole. Nested compiled cores can save
# a different autograd tensor set during checkpoint recomputation.
_local_energy = getattr(_local_energy, "_torchdynamo_orig_callable", _local_energy)
_high_energy = getattr(_high_energy, "_torchdynamo_orig_callable", _high_energy)
from .softmax_matrix_gdn import _states, _state_bwd
from .decay_mass import local_mass, chunk_mass_summaries, boundary_mass, high_mass

# Bound the number of compiled repair-prefix lengths. This is a scheduling
# granularity in computational chunks, unrelated to key/value dimensions.
_REPAIR_PREFIX_QUANTUM = 32


class _FloatStates(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k, w, value, decay, period):
        k, w, value, decay = (x.contiguous() for x in (k, w, value, decay))
        ctx.save_for_backward(k, w, value, decay)
        ctx.period = period
        return _states(k, w, value, decay, period)

    @staticmethod
    def backward(ctx, direct):
        k, w, value, decay = ctx.saved_tensors
        states = _states(k, w, value, decay, ctx.period)
        dk, dw, dv, da = (torch.empty_like(x) for x in (k, w, value, decay))
        bh, n, c, key_dim = k.shape
        _state_bwd[(bh, triton.cdiv(n, ctx.period))](
            k, w, value, decay, states, direct.contiguous(), dk, dw, dv, da,
            n, c, key_dim, value.shape[-1], ctx.period, num_warps=4)
        return dk, dw, dv, da, None


@torch.compile
def _prepare(r, k, v, beta, gc, q, omit_weighted_keys=False):
    c = k.shape[-2]
    gram = k @ k.transpose(-1, -2)
    eye = torch.eye(c, device=k.device, dtype=k.dtype)
    system = eye + (gram * beta.unsqueeze(-2)).tril(-1)
    inverse = torch.linalg.solve_triangular(system, eye.expand_as(system), upper=False, unitriangular=True)
    causal = torch.ones((c, c), device=k.device, dtype=torch.bool).tril()
    decay = (gc.unsqueeze(-1)-gc.unsqueeze(-2)).masked_fill(~causal, -torch.inf).exp().to(k.dtype)
    z = inverse @ k
    w = beta.unsqueeze(-1) * z
    writes = beta.unsqueeze(-1) * ((inverse * decay) @ v)
    rk = (r @ k.transpose(-1, -2)).tril()
    qk = (q @ k.transpose(-1, -2)).tril()
    ar = ((rk * beta.unsqueeze(-2)) @ inverse) * decay
    aq = ((qk * beta.unsqueeze(-2)) @ inverse) * decay
    gain = gc.exp().to(k.dtype).unsqueeze(-1)
    rn = gain * (r-rk @ w)
    qn = gain * (q-qk @ w)
    kn = k * (gc[..., -1:]-gc).exp().to(k.dtype).unsqueeze(-1)
    wn = None if omit_weighted_keys else w * gain
    end = gc[..., -1].exp().to(k.dtype).contiguous()
    return kn, wn, writes, end, rn, qn, ar, aq, (gram, inverse, decay, z)


def _score(read, u, norm2, temperature, floor):
    # The scalar temperature stays FP64; matrix arithmetic is FP32 here.
    return temperature * (read*u).sum(-1) * norm2.clamp_min(floor*floor).rsqrt()


def _condition_masks(norm2, bound, mass, scores, floor):
    with torch.no_grad():
        threshold = torch.finfo(norm2.dtype).eps ** .5
        scale2 = norm2.clamp_min(floor*floor)
        bad = (scale2 < threshold * torch.maximum(bound, mass.square()))
        # A forward norm can be accurate while rounding across the hard
        # denominator floor selects the wrong derivative. This also matters
        # for the rank-one current-token bucket, which has no subtractive
        # norm expansion. Resolve numerically ambiguous branches precisely.
        uncertainty = threshold * torch.maximum(bound, norm2.abs()).clamp_min(floor*floor)
        bad = bad | ((norm2 - floor*floor).abs() <= uncertainty)
        nonfinite = ~torch.isfinite(norm2) | ~torch.isfinite(bound) | ~torch.isfinite(mass) | ~torch.isfinite(scores)
        return bad | nonfinite, nonfinite


def _flag(norm2, bound, mass, scores, floor):
    bad, _ = _condition_masks(norm2, bound, mass, scores, floor)
    return bad.reshape(bad.shape[0], -1).any(-1)


def _diagnostics(norm2, bound, mass, score, y, floor):
    bad, nonfinite = _condition_masks(norm2, bound, mass, score, floor)
    nonfinite = nonfinite | ~torch.isfinite(y).all(-1)
    return bad | nonfinite, nonfinite


def _current_core(ar, aq, k, v, beta, u, temperature, floor, output_dtype, return_masks=False):
    norm2 = beta.square()*k.square().sum(-1)*v.square().sum(-1)
    y = ar.diagonal(dim1=-2, dim2=-1).unsqueeze(-1)*v
    read = aq.diagonal(dim1=-2, dim2=-1).unsqueeze(-1)*v
    score = _score(read, u, norm2, temperature, floor)
    y = y.to(output_dtype)
    bad, nonfinite = _diagnostics(norm2, norm2, norm2.clamp_min(0).sqrt(), score, y, floor)
    if return_masks:
        return y, score, bad, nonfinite
    return y, score, bad.reshape(bad.shape[0], -1).any(-1)


def _local_core(inv, decay, beta, key, value, ar, aq, u, temperature, floor, output_dtype, return_masks=False):
    # Hierarchy width is structural; batch and chunk counts remain dynamic.
    torch._dynamo.mark_static(key, -2)
    period = key.shape[-2]
    half = period//2
    writes = torch.arange(period, device=key.device) < half
    norm2, bound = _local_energy(inv, decay, beta, key.square().sum(-1), value, writes)
    norm2, bound = norm2[..., half:], bound[..., half:]
    y = ar[..., half:, :half] @ value[..., :half, :]
    read = aq[..., half:, :half] @ value[..., :half, :]
    score = _score(read, u[..., half:, :], norm2, temperature.unsqueeze(-1), floor)
    with torch.no_grad():
        positive_mass = beta * key.norm(dim=-1) * value.norm(dim=-1)
        mass = (decay @ (positive_mass * writes).unsqueeze(-1)).squeeze(-1)[..., half:]
    bad, nonfinite = _diagnostics(norm2, bound, mass, score, y.to(output_dtype), floor)
    flag = bad.reshape(bad.shape[0], -1).any(-1)
    y = torch.cat((torch.zeros_like(y), y), -2).flatten(-3, -2)
    score = torch.cat((torch.zeros_like(score), score), -1).flatten(-2, -1)
    if return_masks:
        bad, nonfinite = (torch.cat((torch.zeros_like(x), x), -1).flatten(-2, -1)
                          for x in (bad, nonfinite))
        return y.to(output_dtype), score, bad, nonfinite
    return y.to(output_dtype), score, flag


def _coarse_core(z, beta, k, gc, state, rn, qn, u, temperature, mass0, floor, output_dtype, return_masks=False):
    norm2, bound = _high_energy(z, beta, k.square().sum(-1), gc, state)
    y = rn @ state
    read = qn @ state
    score = _score(read, u, norm2, temperature, floor)
    y = y.to(output_dtype)
    bad, nonfinite = _diagnostics(norm2, bound, high_mass(mass0, gc), score, y, floor)
    if return_masks:
        return y, score, bad, nonfinite
    return y, score, bad.reshape(bad.shape[0], -1).any(-1)


@torch.compile(fullgraph=True, dynamic=True)
def _checkpoint_current_core(*args):
    return checkpoint(_current_core, *args, use_reentrant=False)


def _local_checkpoint_impl(*args):
    return checkpoint(_local_core, *args, use_reentrant=False)


@lru_cache(maxsize=None)
def _compiled_local_shape(period, key_dim, value_dim, input_dtype, output_dtype, floor, return_masks):
    # Each hierarchy width is intentionally static inside the tensor program.
    # Sharing one Dynamo code cache between all widths consumes its recompile
    # limit before even two batch sizes have run. Give each structural shape
    # an independent code object while keeping batch/chunk counts dynamic.
    # A closure alone shares the original code object and does not isolate it.
    name = f'_local_checkpoint_{period}_{key_dim}_{value_dim}_{input_dtype}_{output_dtype}_{floor}_{return_masks}'
    code = _local_checkpoint_impl.__code__.replace(co_name=name)
    implementation = FunctionType(code, globals(), name)
    return torch.compile(implementation, fullgraph=True, dynamic=True)


def _checkpoint_local_core(inv, decay, beta, key, value, ar, aq, u, temperature,
                           floor, output_dtype, return_masks=False):
    return _compiled_local_shape(key.shape[-2], key.shape[-1], value.shape[-1],
                                 key.dtype, output_dtype, floor, return_masks)(
        inv, decay, beta, key, value, ar, aq, u, temperature, floor, output_dtype, return_masks)


@torch.compile(fullgraph=True, dynamic=True)
def _checkpoint_coarse_core(*args):
    return checkpoint(_coarse_core, *args, use_reentrant=False)


def _local(ar, aq, k, v, beta, gc, u, temperature, level, terms, floor, output_dtype, return_masks=False):
    if level == 0:
        args = (ar, aq, k, v, beta, u, temperature, floor, output_dtype, return_masks)
        return _checkpoint_current_core(*args) if torch.is_grad_enabled() else _current_core(*args)
    period = 1 << level
    shape = (*beta.shape[:-1], k.shape[-2]//period, period)
    key = k.reshape(*shape, k.shape[-1])
    value = v.reshape(*shape, v.shape[-1])
    inv, decay = (_segment_blocks(x, period).contiguous() for x in terms[1:3])
    args = (inv, decay, beta.reshape(shape), key, value,
            _segment_blocks(ar, period).contiguous(), _segment_blocks(aq, period).contiguous(),
            u.reshape(*shape, u.shape[-1]), temperature, floor, output_dtype, return_masks)
    return _checkpoint_local_core(*args) if torch.is_grad_enabled() else _local_core(*args)


def _coarse(kn, wn, writes, end, rn, qn, k, beta, gc, u, temperature,
            period, terms, mass_decay, mass_addition, floor, output_dtype, return_masks=False,
            selected_inputs=None):
    key_dim, value_dim = k.shape[-1], writes.shape[-1]
    kp = max(16, triton.next_power_of_2(key_dim))-key_dim
    vp = max(16, triton.next_power_of_2(value_dim))-value_dim
    state = _FloatStates.apply(F.pad(kn,(0,kp)) if kp else kn,
                               F.pad(wn,(0,kp)) if kp else wn,
                               F.pad(writes,(0,vp)) if vp else writes, end, period)
    state = state[..., :key_dim, :value_dim]
    chunks = k.shape[1]
    state = active_select(state, period)
    rn,qn,k,beta,gc,u = (tuple(active_select(x,period) for x in (rn,qn,k,beta,gc,u))
                        if selected_inputs is None else selected_inputs)
    z = active_select(terms[3], period)
    mass0 = active_select(boundary_mass(mass_decay, mass_addition, period), period)
    args = (z, beta, k, gc, state, rn, qn, u, temperature, mass0, floor, output_dtype, return_masks)
    y, score, *diagnostics = (_checkpoint_coarse_core(*args) if torch.is_grad_enabled()
                       else _coarse_core(*args))
    output = active_scatter(y, chunks, period)
    logits = active_scatter(score, chunks, period)
    if return_masks:
        diagnostics = [active_scatter(x, chunks, period) for x in diagnostics]
    return output, logits, *diagnostics


def _replace_selected_periods(y, score, ids, replacement_y, replacement_score, level):
    """Replace independent periods, retaining exact gradients for kept inputs."""
    period = 1 << level
    batch, chunks, chunk = score.shape
    length = chunks * chunk
    padded = triton.cdiv(length, period) * period
    values = y.flatten(1, 2)
    logits = score.flatten(1, 2)
    if padded != length:
        values = F.pad(values, (0, 0, 0, padded-length))
        logits = F.pad(logits, (0, padded-length))
    values = values.reshape(-1, period, y.shape[-1]).index_copy(0, ids, replacement_y)
    logits = logits.reshape(-1, period).index_copy(0, ids, replacement_score)
    return (values.reshape(batch, padded, y.shape[-1])[:, :length].reshape_as(y),
            logits.reshape(batch, padded)[:, :length].reshape_as(score))


def _repair_periods(ys, scores, masks, nonfinite, inputs, floor, output_dtype,
                    repair_chunks=False, compact_cache_reads=False):
    """Repair finite ill-conditioned periods; defer nonfinite heads to caller.

    A whole period's history starts from an exact zero state. For coarse
    periods, optionally evaluate readouts only in affected chunks, while
    retaining the complete precise differentiable history before them.
    Nonfinite heads must instead use the caller's input-gradient barrier.
    """
    from .precise_period_reads import precise_period_reads
    bad_heads = torch.stack([x.flatten(1).any(-1) for x in nonfinite]).any(0)
    period_masks = []
    for level, mask in enumerate(masks):
        period = 1 << level
        flat = mask.flatten(1) & ~bad_heads[:, None]
        padding = (-flat.shape[1]) % period
        if padding:
            flat = F.pad(flat, (0, padding))
        period_masks.append(flat.reshape(flat.shape[0], -1, period).any(-1))
    # Transfer both repair decisions and last requested read positions in one
    # synchronization. A shorter history preserves the original Fenwick period
    # and its write half; it only omits states after every requested read.
    read_masks, history_chunks = {}, {}
    chunk = inputs[1].shape[-2]
    coarse_start = chunk.bit_length()
    if repair_chunks and compact_cache_reads and len(masks) > coarse_start:
        offsets = torch.arange(inputs[1].shape[1], device=inputs[1].device)
        decisions = []
        for level, selected in enumerate(period_masks):
            required = selected.any().long()
            last = torch.zeros_like(required)
            if level >= coarse_start:
                read_masks[level] = masks[level].any(-1) & ~bad_heads[:, None]
                period_chunks = (1 << level) // chunk
                last = torch.where(read_masks[level], offsets % period_chunks + 1, 0).amax()
            decisions.append(torch.stack((required, last)))
        decisions = torch.stack(decisions).tolist()
        needed = [bool(required) for required, _ in decisions]
        for level in range(coarse_start, len(masks)):
            last = decisions[level][1]
            rounded = triton.cdiv(last, _REPAIR_PREFIX_QUANTUM) * _REPAIR_PREFIX_QUANTUM
            history_chunks[level] = min((1 << level) // chunk, rounded)
    else:
        needed = torch.stack([x.any() for x in period_masks]).tolist()
    chunk_cache = None
    read_input_cache = None
    if repair_chunks:
        if any(needed[coarse_start:]):
            from .precise_period_reads import prepare_precise_chunk_cache
            with torch.no_grad():
                indices = torch.arange(inputs[1].shape[1], device=inputs[1].device)
                required = torch.zeros(inputs[1].shape[:2], device=indices.device, dtype=torch.bool)
                for level in range(coarse_start, len(period_masks)):
                    if needed[level]:
                        period_chunks = (1 << level) // chunk
                        selected = period_masks[level].index_select(1, indices // period_chunks)
                        if compact_cache_reads:
                            selected = selected & (indices % period_chunks < history_chunks[level])
                        required |= selected
            chunk_cache = prepare_precise_chunk_cache(
                inputs[1], inputs[2], inputs[3], inputs[4], required)
            if compact_cache_reads:
                from .cached_chunk_reads import prepare_precise_read_input_cache
                r, k, _, beta, gc, u, q, _ = inputs
                required_reads = torch.stack(tuple(read_masks.values())).any(0)
                read_input_cache = prepare_precise_read_input_cache(
                    r, k, beta, gc, u, q, required_reads, chunk_cache=chunk_cache)
    for level, selected in enumerate(period_masks):
        if needed[level]:
            if repair_chunks and (1 << level) > inputs[1].shape[-2]:
                selected_chunks = (read_masks[level] if compact_cache_reads else
                                   masks[level].any(-1) & ~bad_heads[:, None])
                if compact_cache_reads:
                    from .cached_chunk_reads import cached_chunk_reads
                    r,k,_,beta,gc,u,q,temperature = inputs
                    ids,y,score = cached_chunk_reads(
                        r,k,beta,gc,u,q,temperature,level,selected,selected_chunks,
                        chunk_cache,norm_floor=floor,output_dtype=output_dtype,
                        read_input_cache=read_input_cache,
                        history_chunks=history_chunks[level])
                else:
                    ids, y, score = precise_period_reads(
                        *inputs, level, selected, norm_floor=floor,
                        output_dtype=output_dtype, selected_chunks=selected_chunks,
                        chunk_cache=chunk_cache)
                ys[level] = ys[level].flatten(0, 1).index_copy(0, ids, y).reshape_as(ys[level])
                scores[level] = scores[level].flatten(0, 1).index_copy(0, ids, score).reshape_as(scores[level])
                continue
            ids, y, score = precise_period_reads(*inputs, level, selected,
                                                  norm_floor=floor, output_dtype=output_dtype)
            ys[level], scores[level] = _replace_selected_periods(
                ys[level], scores[level], ids, y, score, level)
    return bad_heads


def fast_matrix_gdn(r, k, v, g, beta, u, q, log_temperature, norm_floor=1e-6, vector_eps=1e-6,
                    repair_periods=False, shared_local=False, fused_local=False,
                    repair_chunks=False, shared_gathers=False, compact_cache_reads=False,
                    shared_states=False):
    """Return output and [batch*heads] flags requiring precise recomputation."""
    bsz, length, heads, key_dim = k.shape
    value_dim = v.shape[-1]
    output_dtype = v.dtype
    levels = (length-1).bit_length()+1
    chunk = _CHUNK_SIZE
    padded = triton.cdiv(length, chunk)*chunk
    nchunks = padded//chunk
    def chunks(x):
        x=x.float()
        if length != padded:
            x=F.pad(x,(0,0)*(x.ndim-2)+(0,padded-length))
        return x.reshape(bsz,nchunks,chunk,heads,*x.shape[3:]).movedim(3,1).reshape(
            bsz*heads,nchunks,chunk,*x.shape[3:]).contiguous()
    with torch.autocast('cuda',enabled=False):
        r,k=_gdn_normalize(r),_gdn_normalize(k)
        u=F.normalize(u.float(),dim=-1,eps=vector_eps)
        q=F.normalize(q.float(),dim=-1,eps=vector_eps)
        r,k,v,u,q=map(chunks,(r,k,v,u,q))
        beta=chunks(beta)
        # Keep scalar prefix sums/differences accurate even after a large
        # negative decay. Matrix products remain FP32.
        gc=chunks(g).double().cumsum(-1)
        kn,wn,writes,end,rn,qn,ar,aq,terms=_prepare(r,k,v,beta,gc,q,omit_weighted_keys=shared_states)
        mass_decay,mass_addition=chunk_mass_summaries(k,v,beta,gc)
        temperature=log_temperature.double().unsqueeze(0).expand(bsz,-1).reshape(bsz*heads,1,1).exp()
        ys,scores,flags,nonfinite=[],[],[],[]
        local_levels=min(chunk.bit_length(),levels)
        if shared_local:
            from .shared_local_router import shared_local_router
            ys,scores,flags,nonfinite = shared_local_router(
                ar,aq,k,v,beta,gc,u,temperature,terms,norm_floor,output_dtype,
                local_levels,use_fused=fused_local)
            if not repair_periods:
                flags = [x.flatten(1).any(-1) for x in flags]
        else:
            for level in range(local_levels):
                args=(ar,aq,k,v,beta,gc,u,temperature,level,terms,norm_floor,output_dtype,repair_periods)
                y,score,*diagnostic=_local(*args)
                ys.append(y);scores.append(score);flags.append(diagnostic[0])
                if repair_periods:
                    nonfinite.append(diagnostic[1])
        gathered, state_pairs = None, None
        if (shared_gathers or shared_states) and levels > local_levels:
            periods = tuple(1 << (level-(chunk.bit_length()-1))
                            for level in range(local_levels,levels))
            if shared_gathers:
                from .multi_active_select import multi_active_select
                gathered = tuple(multi_active_select(x,periods) for x in (rn,qn,k,beta,gc,u))
            if shared_states:
                from .multi_projected_states import multi_projected_states
                factor = beta*gc.exp().to(beta.dtype)
                state_pairs = multi_projected_states(kn,terms[3],factor,writes,end,periods,
                                                    active_outputs=True)
        for level in range(local_levels,levels):
            period=1<<(level-(chunk.bit_length()-1))
            args=(kn,wn,writes,end,rn,qn,k,beta,gc,u,temperature,period,terms,
                  mass_decay,mass_addition,norm_floor,output_dtype,repair_periods)
            if state_pairs is not None:
                from .projected_coarse_router import projected_coarse
                selected = None if gathered is None else tuple(x[level-local_levels] for x in gathered)
                y,score,*diagnostic=projected_coarse(
                    *args,selected_inputs=selected,state_pair=state_pairs[level-local_levels],
                    state_pair_active=True)
            elif gathered is None:
                y,score,*diagnostic=_coarse(*args)
            else:
                selected = tuple(x[level-local_levels] for x in gathered)
                y,score,*diagnostic=_coarse(*args,selected_inputs=selected)
            ys.append(y);scores.append(score);flags.append(diagnostic[0])
            if repair_periods:
                nonfinite.append(diagnostic[1])
        flagged = (_repair_periods(ys, scores, flags, nonfinite,
                                  (r,k,v,beta,gc,u,q,temperature),norm_floor,output_dtype,
                                  repair_chunks=repair_chunks,compact_cache_reads=compact_cache_reads)
                   if repair_periods else torch.stack(flags).any(0))
        out=_RoutingReduce.apply(torch.stack(ys,-2),torch.stack(scores,-1),key_dim**-.5)
    return out.reshape(bsz,heads,nchunks,chunk,value_dim).permute(0,2,3,1,4).reshape(
        bsz,padded,heads,value_dim)[:,:length],flagged
