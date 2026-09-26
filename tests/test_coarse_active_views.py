"""Grouped production coarse views versus compact selection and its adjoint."""
import importlib
from pathlib import Path
import sys
import types
import unittest

import torch


def modules():
    name = '_coarse_active_view_tests'
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(Path(__file__).resolve().parents[1] / 'hattention')]
        sys.modules[name] = package
    return tuple(importlib.import_module(name + '.' + item) for item in
                 ('multi_active_select', 'fused_projected_coarse_router', 'projected_coarse_router'))


def mathematical_core(projected, beta, key, gc, state, rn, qn, u, temperature, batch):
    initial = state.square().sum((-2, -1)).unsqueeze(-1)
    energy = beta * (2 - beta * key.square().sum(-1)) * projected.square().sum(-1)
    norm2 = (2 * gc).exp().float() * (initial - energy.cumsum(-1))
    numerator = ((qn @ state) * u).sum(-1)
    n = projected.numel() // (batch * projected.shape[-2] * projected.shape[-1])
    score = (temperature * numerator.reshape(batch, n, -1).double()
             * norm2.reshape(batch, n, -1).clamp_min(1e-12).rsqrt().double())
    return (rn @ state).reshape(batch, n, *u.shape[-2:]), score


class TestCoarseActiveViews(unittest.TestCase):
    def check_views(self, device):
        selection, _, _ = modules()
        for strided in (False, True):
            source = torch.randn(2, 12, 3, 10 if strided else 5, device=device)
            raw = source[..., ::2] if strided else source
            actual = raw.detach().requires_grad_()
            expected = raw.detach().clone().requires_grad_()
            periods = (2, 4, 6, 4, 12)
            views = selection.multi_active_views(actual, periods)
            loss = actual.sum() * 0
            want = torch.zeros_like(actual)
            for i, (period, view) in enumerate(zip(periods, views)):
                ids = torch.arange(12, device=device)
                ids = ids[ids % period >= period // 2]
                reference = expected.index_select(1, ids)
                torch.testing.assert_close(view.reshape_as(reference), reference, rtol=0, atol=0)
                if not strided:
                    self.assertEqual(view.untyped_storage().data_ptr(), actual.untyped_storage().data_ptr())
                if i in (1, 4):  # Deliberately leave outputs unused.
                    continue
                gradient = torch.randn_like(reference)
                loss = loss + (view.reshape_as(reference) * gradient).sum()
                # Match the documented ordered adjoint sum explicitly; a
                # separate autograd graph may reverse the FP32 add order.
                want.index_add_(1, ids, gradient)
            got, = torch.autograd.grad(loss, actual)
            torch.testing.assert_close(got, want, rtol=0, atol=0)
        for invalid in ((3,), (8,)):
            with self.assertRaises(ValueError):
                selection.multi_active_views(torch.ones(2, 12, 3, device=device), invalid)

    def check_core(self, device):
        selection, fused, _ = modules()
        configurations = ((2, 8, 3, 5, 7, (2, 4, 8, 4)),
                          (3, 12, 5, 17, 9, (2, 4, 6, 12)),
                          (1, 4, 64, 128, 64, (2, 4)))
        for bh, n, c, key_dim, value_dim, periods in configurations:
            for scalar in (False, True):
                for scale in (1., 1e-9):
                    with self.subTest(device=device, bh=bh, n=n, scalar=scalar, scale=scale):
                        generator = torch.Generator().manual_seed(811 + n)
                        def random(*shape, dtype=torch.float32):
                            return torch.randn(shape, dtype=dtype, generator=generator).to(device)
                        raw = (random(bh, n, c).sigmoid() * .2,
                               random(bh, n, c, key_dim) * .1,
                               -random(bh, n, c, dtype=torch.float64).sigmoid() * .2,
                               random(bh, n, c, key_dim) * .2,
                               random(bh, n, c, key_dim) * .2,
                               random(bh, n, c, value_dim) * .2)
                        temperature = (torch.tensor(8., dtype=torch.float64, device=device) if scalar else
                                       torch.linspace(2., 8., bh, dtype=torch.float64, device=device).view(bh, 1, 1))
                        payload = tuple(x for _ in periods for x in (
                            random(bh, n // 2, c, value_dim) * (.01 * scale),
                            random(bh, n // 2, key_dim, value_dim) * scale))
                        inputs = (*raw, temperature, *payload)
                        a = tuple(x.detach().clone().requires_grad_() for x in inputs)
                        r = tuple(x.detach().clone().requires_grad_() for x in inputs)
                        views = tuple(selection.multi_active_views(x, periods) for x in a[:6])
                        compact = tuple(selection.multi_active_select(x, periods) for x in r[:6])
                        losses, reference_losses = [], []
                        for i, period in enumerate(periods):
                            half, groups = period // 2, n // period
                            av = tuple(x[i] for x in views); rv = tuple(x[i] for x in compact)
                            projected, state = a[7 + 2*i:9 + 2*i]
                            rp, rs = r[7 + 2*i:9 + 2*i]
                            args = (projected.view(bh * groups, half, c, value_dim), *av[:3],
                                    state.view(bh * groups, half, key_dim, value_dim), *av[3:], a[6])
                            refargs = (rp, *rv[:3], rs, *rv[3:], r[6])
                            if device == 'cuda':
                                mass = torch.ones(bh, n // 2, device=device)
                                got = fused.fused_projected_coarse_core(
                                    *args, mass.reshape(bh * groups, half), 1e-6, torch.float32, True,
                                    raw_half=half, head_groups=groups)
                                got = tuple(x.reshape(bh, n // 2, *x.shape[2:]) for x in got)
                                want = fused.fused_projected_coarse_core(*refargs, mass, 1e-6, torch.float32, True)
                            else:
                                got = mathematical_core(*args, batch=bh)
                                want = mathematical_core(*refargs, batch=bh)
                            for x, y in zip(got, want):
                                torch.testing.assert_close(x, y, rtol=0, atol=0)
                            dy, ds = torch.randn_like(got[0]), torch.randn_like(got[1])
                            losses.append((got[0] * dy).sum() + (got[1] * ds).sum())
                            reference_losses.append((want[0] * dy).sum() + (want[1] * ds).sum())
                        gradients = torch.autograd.grad(sum(losses), a)
                        reference_gradients = torch.autograd.grad(sum(reference_losses), r)
                        for got, want in zip(gradients, reference_gradients):
                            torch.testing.assert_close(got, want, rtol=0, atol=0)

    def test_cpu_view_storage_duplicates_unused_and_strides(self):
        self.check_views('cpu')

    def test_cpu_grouped_all_gradients(self):
        self.check_core('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_view_storage_duplicates_unused_and_strides(self):
        self.check_views('cuda')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_grouped_all_gradients_temperature_and_flags(self):
        self.check_core('cuda')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_compiled_batch_one_four_one(self):
        selection, fused, router = modules()
        self.assertFalse(torch._dynamo.config.disable, 'This test requires genuine compilation')
        router._compiled_grouped_core.cache_clear()
        before = torch._dynamo.utils.counters['stats']['unique_graphs']
        for batch in (1, 4, 1):
            n, period, c, key_dim, value_dim = 8, 4, 3, 5, 7
            half, groups = period // 2, n // period
            raw = (torch.rand(batch, n, c, device='cuda') * .2,
                   torch.randn(batch, n, c, key_dim, device='cuda') * .1,
                   -torch.rand(batch, n, c, device='cuda', dtype=torch.float64) * .2,
                   torch.randn(batch, n, c, key_dim, device='cuda'),
                   torch.randn(batch, n, c, key_dim, device='cuda'),
                   torch.randn(batch, n, c, value_dim, device='cuda'))
            def make_args():
                leaves = tuple(x.detach().clone().requires_grad_() for x in raw)
                view = tuple(selection.multi_active_views(x, (period,))[0] for x in leaves)
                projected = torch.randn(batch * groups, half, c, value_dim, device='cuda') * .01
                state = torch.randn(batch * groups, half, key_dim, value_dim, device='cuda')
                temperature = torch.full((batch, 1, 1), 8., dtype=torch.float64, device='cuda')
                extra = tuple(x.requires_grad_() for x in (projected, state, temperature))
                args = (extra[0], *view[:3], extra[1], *view[3:], extra[2],
                        torch.ones(batch * groups, half, device='cuda'), 1e-6, torch.float32, True)
                return leaves + extra, args
            leaves, args = make_args()
            reference = tuple(
                x.detach().contiguous().reshape(batch, n // 2, *x.shape[2:]).requires_grad_(x.requires_grad)
                if isinstance(x, torch.Tensor) and index != 8 else
                x.detach().clone().requires_grad_(x.requires_grad) if isinstance(x, torch.Tensor) else x
                for index, x in enumerate(args))
            actual = router._compiled_grouped_core(half, groups)(*args)
            # Compare layouts under the same compilation context. The
            # four-way CUDA control found both compiled layouts bit-identical,
            # while both differed from eager by ordinary FMA/reduction rounding
            # in beta/state/temperature gradients. Eager layout equivalence is
            # checked separately above, also with zero tolerance.
            expected = router._checkpoint_core(*reference)
            for a, e in zip(actual, expected):
                torch.testing.assert_close(a.reshape_as(e), e, rtol=0, atol=0)
            dy, ds = torch.randn_like(actual[0]), torch.randn_like(actual[1])
            ga = torch.autograd.grad((actual[0]*dy).sum()+(actual[1]*ds).sum(), args[:9])
            ge = torch.autograd.grad((expected[0]*dy.reshape_as(expected[0])).sum()
                                    +(expected[1]*ds.reshape_as(expected[1])).sum(), reference[:9])
            for a, e in zip(ga, ge):
                torch.testing.assert_close(a.reshape_as(e), e, rtol=0, atol=0)
        self.assertGreater(torch._dynamo.utils.counters['stats']['unique_graphs'], before)


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main(verbosity=2)
