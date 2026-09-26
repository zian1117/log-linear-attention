"""Shared affine tree for compact FP32 Fenwick boundary states.

The chunk transition is A=aI-K^T(fZ), B=K^T V. Summaries are shared
across periods; the manual adjoint reverses each shared node once after all
periods contribute. This changes evaluation order, not the recurrence.
The caller retains its existing conditioning diagnostics and precise repair.

Only complete power-of-two chunk groups and compact outputs use this backend.
The public multi_projected_states dispatcher retains the scan for other cases.
"""
import torch

from .projected_state_tree_kernels import (
    mm, bmm_add, bmm_accumulate, bmm_difference, diagonal_bmm, interleaved_bmm,
)

def select_active(x, period):
    """View each period's right half as [BH*groups, half, M, N]."""
    bh, chunks, rows, columns = x.shape
    if chunks % period:
        raise ValueError('Strided selection requires complete period groups.')
    groups = chunks // period
    selected = x.view(bh, groups, period, rows, columns)[:, :, period // 2:]
    return selected.view(bh * groups, period // 2, rows, columns)

def select_read_left(x, period, node_size):
    """View left children inside each right half at one tree depth.

x has [BH, original_chunks/node_size, M, N]. Returned batch axes are
[BH*period_groups, period/(4*node_size)] and share the original storage.
"""
    bh, nodes, rows, columns = x.shape
    width = period // node_size
    if period % node_size or width < 4 or width % 4 or nodes % width:
        raise ValueError('Strided child selection requires complete binary groups.')
    groups = nodes // width
    selected = x.view(bh, groups, width, rows, columns)[:, :, width // 2::2]
    return selected.view(bh * groups, width // 4, rows, columns)

class _SharedTree(torch.autograd.Function):

    @staticmethod
    def forward(ctx, k, z, factor, value, decay, periods):
        k, z, factor, value, decay = (x.contiguous() for x in (k, z, factor, value, decay))
        bh, n, c, kdim = k.shape
        vdim = value.shape[-1]
        useful = [p for p in periods if p // 2 < n]
        depth = max((p // 2 for p in useful), default=1).bit_length() - 1
        atree, btree = ([], [])
        if useful and depth:
            a = diagonal_bmm(k.transpose(-1, -2), factor[..., None] * z, decay)
            b = mm(k.transpose(-1, -2), value)
            atree.append(a)
            btree.append(b)
            for _ in range(1, depth):
                la, ra = (a[:, 0::2], a[:, 1::2])
                lb, rb = (b[:, 0::2], b[:, 1::2])
                a = mm(ra, la)
                b = bmm_add(ra, lb, rb)
                atree.append(a)
                btree.append(b)
            top = bmm_add(a[:, 1::4], b[:, 0::4], b[:, 1::4])
        elif useful:
            top = mm(k[:, 0::2].transpose(-1, -2), value[:, 0::2])
        else:
            top = k.new_empty((bh, 0, kdim, vdim))
        outputs = []
        states = []
        for period in periods:
            half = period // 2
            if half >= n:
                state = k.new_empty((bh, 0, kdim, vdim))
                projected = k.new_empty((bh, 0, c, vdim))
            else:
                groups = n // period
                level = half.bit_length() - 1
                incoming = top if level == depth else btree[level][:, 0::2]
                incoming = incoming.view(bh * groups, 1, kdim, vdim)
                for child_level in range(level - 1, -1, -1):
                    transition = select_read_left(atree[child_level], period, 1 << child_level)
                    incoming = interleaved_bmm(transition, incoming)
                state = incoming.view(bh, groups * half, kdim, vdim)
                projected = mm(select_active(z, period), incoming).view(bh, groups * half, c, vdim)
            states.append(state)
            outputs.extend((state, projected))
        ctx.save_for_backward(k, z, factor, value, decay, *atree, *btree, top, *states)
        ctx.periods = periods
        ctx.depth = depth
        ctx.useful = bool(useful)
        ctx.set_materialize_grads(False)
        return tuple(outputs)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, *gradients):
        k, z, factor, value, decay, *saved = ctx.saved_tensors
        depth = ctx.depth
        atree = saved[:depth]
        btree = saved[depth:2 * depth]
        top = saved[2 * depth]
        states = saved[2 * depth + 1:]
        if not ctx.useful or all((x is None or not x.numel() for x in gradients)):
            return (*(torch.zeros_like(x) for x in (k, z, factor, value, decay)), None)
        bh, n, c, kdim = k.shape
        vdim = value.shape[-1]
        ga = [torch.zeros_like(x) for x in atree]
        gb = [torch.zeros_like(x) for x in btree]
        gt = torch.zeros_like(top)
        direct_dz = torch.zeros_like(z)
        # Returned H retains each subtree root at its first leaf, so the
        # reverse descent needs no saved copies of intermediate read states.
        for index, period in enumerate(ctx.periods):
            dh, dp = gradients[2 * index:2 * index + 2]
            if (dh is None or not dh.numel()) and (dp is None or not dp.numel()):
                continue
            if dh is not None and any((s <= 0 for s in dh.stride())):
                dh = dh.contiguous()
            if dp is not None and any((s <= 0 for s in dp.stride())):
                dp = dp.contiguous()
            half = period // 2
            groups = n // period
            level = half.bit_length() - 1
            h = states[index].view(bh * groups, half, kdim, vdim)
            if dp is not None:
                pgrad = dp.reshape(bh * groups, half, c, vdim)
                g = mm(select_active(z, period).transpose(-1, -2), pgrad)
                bmm_accumulate(pgrad, h.transpose(-1, -2), select_active(direct_dz, period))
                if dh is not None:
                    g.add_(dh.reshape_as(g))
            else:
                g = dh.reshape(bh * groups, half, kdim, vdim)
            for child_level in range(level):
                left, right = (g[:, 0::2], g[:, 1::2])
                incoming = h[:, ::(1 << (child_level + 1))]
                bmm_accumulate(right, incoming.transpose(-1, -2), select_read_left(ga[child_level], period, 1 << child_level))
                transition = select_read_left(atree[child_level], period, 1 << child_level)
                g = bmm_add(transition.transpose(-1, -2), right, left)
            root = g.view(bh, groups, kdim, vdim)
            if level == depth:
                gt.add_(root)
            else:
                gb[level][:, 0::2].add_(root)
        if depth:
            bmm_accumulate(gt, btree[-1][:, 0::4].transpose(-1, -2), ga[-1][:, 1::4])
            bmm_accumulate(atree[-1][:, 1::4].transpose(-1, -2), gt, gb[-1][:, 0::4])
            gb[-1][:, 1::4].add_(gt)
            # All consumer periods have contributed before shared summaries
            # are reversed. These buffers never alias saved A/B/H or inputs.
            for level in range(depth - 1, 0, -1):
                al, ar = (atree[level - 1][:, 0::2], atree[level - 1][:, 1::2])
                bl = btree[level - 1][:, 0::2]
                bmm_accumulate(ar.transpose(-1, -2), ga[level], ga[level - 1][:, 0::2])
                bmm_accumulate(ga[level], al.transpose(-1, -2), ga[level - 1][:, 1::2])
                bmm_accumulate(gb[level], bl.transpose(-1, -2), ga[level - 1][:, 1::2])
                bmm_accumulate(ar.transpose(-1, -2), gb[level], gb[level - 1][:, 0::2])
                gb[level - 1][:, 1::2].add_(gb[level])
            leaf_a, leaf_b = (ga[0], gb[0])
            j = mm(k, leaf_a)
            dk = bmm_difference(value, leaf_b.transpose(-1, -2), factor[..., None] * z, leaf_a.transpose(-1, -2))
            dz = direct_dz - factor[..., None] * j
            df = -(j * z).sum(-1)
            dv = mm(k, leaf_b)
            da = leaf_a.diagonal(dim1=-2, dim2=-1).sum(-1)
        else:
            leaf_b = k.new_zeros((bh, n, kdim, vdim))
            leaf_b[:, 0::2] = gt
            dk = mm(value, leaf_b.transpose(-1, -2))
            dv = mm(k, leaf_b)
            dz = direct_dz
            df = torch.zeros_like(factor)
            da = torch.zeros_like(decay)
        return (dk, dz, df, dv, da, None)


def projected_state_tree(k, z, factor, value, decay, periods, *, active_outputs=True):
    """Return compact (H, Z@H) pairs for complete binary Fenwick periods.

Inputs are FP32 with shapes [BH,N,C,K], [BH,N,C,K], [BH,N,C],
[BH,N,C,V], [BH,N]. CPU execution is a functional proof path; CUDA uses
TF32x3 products with FP32 accumulation. First-order gradients only.
Unsupported layouts should use multi_projected_states, which dispatches to
its original scan. Empty/unused outputs and duplicate periods are supported.
"""
    periods = tuple(periods)
    if not active_outputs:
        raise ValueError('The projected tree returns compact active outputs only.')
    if k.ndim != 4 or min(k.shape) <= 0 or value.ndim != 4 or value.shape[-1] <= 0:
        raise ValueError('Expected nonempty [BH,N,C,K/V] inputs.')
    if k.shape[1] & (k.shape[1] - 1):
        raise ValueError('The projected tree requires power-of-two chunk counts.')
    if any(not isinstance(p, int) or isinstance(p, bool) or p < 2 or p & (p - 1) for p in periods):
        raise ValueError('The projected tree requires power-of-two periods >= 2.')
    if (z.shape != k.shape or factor.shape != k.shape[:-1]
            or value.shape[:-1] != k.shape[:-1] or decay.shape != k.shape[:2]):
        raise ValueError('Incompatible projected tree input shapes.')
    if any(x.dtype != torch.float32 or x.device != k.device for x in (k,z,factor,value,decay)):
        raise ValueError('Projected tree inputs must be FP32 on one device.')
    if not periods:
        return ()
    flat = _SharedTree.apply(k,z,factor,value,decay,periods)
    return tuple((flat[2*i],flat[2*i+1]) for i in range(len(periods)))
