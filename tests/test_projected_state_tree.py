"""Production projected-state tree versus an independent matrix recurrence."""
import importlib
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch


def modules():
    name = '_projected_tree_tests'
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(Path(__file__).resolve().parents[1] / 'hattention')]
        sys.modules[name] = package
    return tuple(importlib.import_module(name + '.' + module) for module in
                 ('projected_state_tree', 'multi_projected_states'))


def recurrence(k, z, factor, value, decay, periods):
    result = []
    for period in periods:
        state = k.new_zeros(k.shape[0], k.shape[-1], value.shape[-1])
        states, projections = [], []
        for t in range(k.shape[1]):
            if t % period == 0:
                state = torch.zeros_like(state)
            projected = z[:, t] @ state
            if t % period >= period // 2:
                states.append(state)
                projections.append(projected)
            write = value[:, t] if t % period < period // 2 else torch.zeros_like(value[:, t])
            state = (decay[:, t, None, None] * state
                     + k[:, t].transpose(-1, -2)
                     @ (write - factor[:, t, :, None] * projected))
        shape = (k.shape[0], 0)
        result.append((torch.stack(states, 1) if states else k.new_empty(*shape, k.shape[-1], value.shape[-1]),
                       torch.stack(projections, 1) if projections else k.new_empty(*shape, k.shape[-2], value.shape[-1])))
    return tuple(result)


def inputs(shape, device, strided=False):
    generator = torch.Generator().manual_seed(339 + shape[1])
    b, n, c, k, v = shape
    def randn(*dims):
        return torch.randn(dims, generator=generator).to(device)
    raw = (randn(b, n, c, k) * .05, randn(b, n, c, k) * .05,
           randn(b, n, c).sigmoid() * .4, randn(b, n, c, v) * .1,
           randn(b, n).sigmoid() * .2 + .7)
    if strided:
        raw = tuple(x.transpose(-1, -2).contiguous().transpose(-1, -2) for x in raw)
    return tuple(x.detach().requires_grad_() for x in raw)


class TestProjectedStateTree(unittest.TestCase):
    def check_recurrence(self, device):
        tree, _ = modules()
        cases = (
            ((2, 1, 3, 5, 7), (2, 4), 'all'),
            ((2, 8, 3, 5, 7), (2, 4, 8, 4, 32), 'all'),
            ((1, 16, 5, 7, 3), (2, 8, 16), 'state'),
            ((2, 8, 3, 5, 7), (2, 4, 8), 'projection'),
            ((2, 8, 3, 5, 7), (2, 4, 8, 4), 'one'),
            ((2, 8, 5, 7, 3), (2, 4, 8), 'strided'),
        )
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            for shape, periods, selection in cases:
                with self.subTest(device=device, shape=shape, periods=periods, selection=selection):
                    raw = inputs(shape, device, selection == 'strided')
                    exact = tuple(x.detach().double().requires_grad_() for x in raw)
                    actual = tree.projected_state_tree(*raw, periods)
                    expected = recurrence(*exact, periods)
                    # Zero anchors make empty-active periods' reference gradients
                    # explicit zeros without changing any nonempty objective.
                    loss = sum(x.sum() * 0 for x in raw)
                    ref_loss = sum(x.sum() * 0 for x in exact)
                    for i, (pair, reference) in enumerate(zip(actual, expected)):
                        for component, (got, want) in enumerate(zip(pair, reference)):
                            torch.testing.assert_close(got.double(), want, rtol=3e-5, atol=3e-7)
                            if ((selection == 'state' and component == 1)
                                    or (selection == 'projection' and component == 0)
                                    or (selection == 'one' and (i, component) != (1, 0))):
                                continue
                            grad = torch.randn_like(got)
                            if selection == 'strided':
                                grad = grad.transpose(-1, -2).contiguous().transpose(-1, -2)
                            loss = loss + (got * grad).sum()
                            ref_loss = ref_loss + (want * grad.double()).sum()
                    got_grad = torch.autograd.grad(loss, raw)
                    want_grad = torch.autograd.grad(ref_loss, exact)
                    for name, got, want in zip(('k', 'z', 'factor', 'value', 'decay'), got_grad, want_grad):
                        self.assertTrue(torch.isfinite(got).all(), name)
                        torch.testing.assert_close(got.double(), want, rtol=1e-4, atol=2e-6, msg=name)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32

    def check_all_none(self, device):
        tree, _ = modules()
        class Context:
            def save_for_backward(self, *tensors):
                self.saved_tensors = tensors
            def set_materialize_grads(self, value):
                self.materialize_grads = value
        context = Context()
        raw = inputs((2, 8, 3, 5, 7), device)
        outputs = tree._SharedTree.forward(context, *raw, (2, 4, 8))
        gradients = tree._SharedTree.backward(context, *(None for _ in outputs))
        self.assertEqual(len(gradients), 6)
        for got, original in zip(gradients[:5], raw):
            torch.testing.assert_close(got, torch.zeros_like(original), rtol=0, atol=0)
        self.assertIsNone(gradients[-1])

    def test_cpu_recurrence_all_gradients_and_unused_outputs(self):
        self.check_recurrence('cpu')

    def test_cpu_all_none_backward(self):
        self.check_all_none('cpu')

    def check_kernel_views(self, device):
        tree, _ = modules()
        kernels = importlib.import_module(tree.__package__ + '.projected_state_tree_kernels')
        generator = torch.Generator().manual_seed(298)
        def view(rows, cols):
            # Nonunit strides in both chunk and matrix axes, without expand.
            return torch.randn((2, 6, cols, rows), generator=generator).to(device)[:, ::2].transpose(-1, -2)
        a, b, addend = view(5, 7), view(7, 9), view(5, 9)
        expected = .3 * (a.double() @ b.double()) - .7 * addend.double()
        destination = view(5, 9)
        result = kernels.bmm_add(a, b, addend, out=destination, alpha=.3, beta=-.7)
        self.assertEqual(result.data_ptr(), destination.data_ptr())
        torch.testing.assert_close(result.double(), expected, rtol=3e-5, atol=2e-6)
        expected = (a.double() @ b.double()) + destination.double()
        kernels.bmm_accumulate(a, b, destination)
        torch.testing.assert_close(destination.double(), expected, rtol=3e-5, atol=2e-6)
        # beta=0 must not read the destination's uninitialized/NaN contents.
        destination.fill_(float('nan'))
        kernels.bmm_accumulate(a, b, destination, beta=0.)
        torch.testing.assert_close(destination.double(), a.double() @ b.double(), rtol=3e-5, atol=2e-6)
        right = view(7, 5)
        decay = torch.rand((6, 2), generator=generator).to(device).transpose(0, 1)[:, ::2]
        diagonal = kernels.diagonal_bmm(a, right, decay)
        expected = decay.double()[..., None, None] * torch.eye(5, device=device).double() - a.double() @ right.double()
        torch.testing.assert_close(diagonal.double(), expected, rtol=3e-5, atol=2e-6)
        transition, incoming = view(5, 5), view(5, 9)
        interleaved = kernels.interleaved_bmm(transition, incoming)
        expected = torch.stack((incoming.double(), transition.double() @ incoming.double()), 2).flatten(1, 2)
        torch.testing.assert_close(interleaved.double(), expected, rtol=3e-5, atol=2e-6)

    def test_cpu_positive_strided_kernel_views(self):
        self.check_kernel_views('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_recurrence_all_gradients_and_unused_outputs(self):
        self.check_recurrence('cuda')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_all_none_backward(self):
        self.check_all_none('cuda')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_positive_strided_kernel_views(self):
        self.check_kernel_views('cuda')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_public_dispatch_eligibility_and_scan_fallback(self):
        tree, dispatcher = modules()
        cases = ((8, (2, 4, 8, 4), True, True),
                 (7, (2, 4, 8), True, False),
                 (8, (2, 3, 8), True, False),
                 (8, (2, 4, 8), False, False))
        for chunks, periods, active, use_tree in cases:
            with self.subTest(chunks=chunks, periods=periods, active=active):
                raw = inputs((2, chunks, 3, 5, 7), 'cuda', strided=True)
                with patch.object(tree, 'projected_state_tree', wraps=tree.projected_state_tree) as tree_spy, \
                     patch.object(dispatcher._MultiProjectedStates, 'apply', wraps=dispatcher._MultiProjectedStates.apply) as scan_spy:
                    outputs = dispatcher.multi_projected_states(*raw, periods, active_outputs=active)
                    torch.autograd.grad(sum(x.square().sum() for pair in outputs for x in pair), raw)
                self.assertEqual(tree_spy.call_count, int(use_tree))
                self.assertEqual(scan_spy.call_count, int(not use_tree))
                if active:
                    expected = recurrence(*(x.detach().double() for x in raw), periods)
                    for pair, reference in zip(outputs, expected):
                        for got, want in zip(pair, reference):
                            torch.testing.assert_close(got.double(), want, rtol=3e-5, atol=3e-7)


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main(verbosity=2)
