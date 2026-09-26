"""Reuse guarded adjoint products; refresh every rejected period before grads."""
import importlib
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import torch


def load_modules():
    name = '_refined_projection_reuse_tests'
    root = Path(__file__).resolve().parents[1]
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(root / 'hattention')]
        sys.modules[name] = package
    module = importlib.import_module(name + '.refined_bucket_states')
    guard = importlib.import_module(name + '.refinement_guard')
    spec = importlib.util.spec_from_file_location(
        name + '_oracle', root / 'tests/test_refined_guard_fallback.py')
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    return module, guard, oracle


class TestRefinedProjectionReuse(unittest.TestCase):
    def check_reuse(self, device, actual_guard):
        module, guard, oracle = load_modules()
        generator = torch.Generator(device=device).manual_seed(32561)

        def randn(*shape):
            return torch.randn(*shape, generator=generator, device=device, dtype=torch.float64)

        for chunks, chunk, key_dim, value_dim, period in (
                (1, 3, 5, 7, 2), (7, 5, 7, 9, 3), (33, 7, 13, 9, 32),
                (5, 4, 3, 5, 8), (16, 8, 17, 11, 8)):
            for mode in ('healthy', 'finite_reverse', 'nonfinite_reverse', 'both'):
                with self.subTest(shape=(chunks, chunk, key_dim, value_dim), period=period, mode=mode):
                    groups = (chunks + period - 1) // period
                    key = .01 * randn(2, chunks, chunk, key_dim)
                    raw = (key, .1 * key, randn(2, chunks, chunk, value_dim),
                           torch.full((2, chunks), .9, dtype=torch.float64, device=device))
                    inputs = tuple(x.clone().requires_grad_() for x in raw)
                    references = tuple(x.clone().requires_grad_() for x in raw)
                    forward_mask = torch.zeros((2, groups), dtype=torch.bool, device=device)
                    reverse_mask = torch.zeros_like(forward_mask)
                    if mode == 'both':
                        forward_mask[0, 0] = True
                    if mode != 'healthy':
                        reverse_mask[0, 0] = True
                        if chunks > period:
                            reverse_mask[1, -1] = True
                    captured = {}

                    def forward(k, w, v, a, p, refinements=2):
                        state = oracle.forward_reference(k, w, v, a, p)
                        if mode == 'both':
                            state[0, min(1, chunks - 1)] = torch.nan
                        return state

                    def reverse(k, w, a, direct, p, refinements=2):
                        adjoint = oracle.reverse_reference(k, w, a, direct, p)
                        if mode != 'healthy':
                            adjoint[0, min(1, chunks - 1)] = torch.nan if mode in ('nonfinite_reverse', 'both') else .5
                            if chunks > period:
                                adjoint[1, -1] = .4
                        return adjoint

                    def reverse_guard(k, w, a, direct, adjoint, p, *, return_projection=False):
                        self.assertTrue(return_projection, 'Backward must request the existing product')
                        if actual_guard:
                            failed, product = guard.reverse_failures(
                                k, w, a, direct, adjoint, p, return_projection=True)
                        else:
                            # Evaluate the actual product-producing guard core;
                            # only the GPU scalar scan is replaced on CPU.
                            quantities = guard._reverse_quantities(
                                k, w, a, direct, adjoint, p, return_projection=True)
                            failed, product = reverse_mask, quantities[4]
                        captured.update(failed=failed, product=product)
                        return failed, product

                    original_gradients = module._factor_gradients

                    def gradients(k, w, v, state, adjoint, p, projected_adjoint=None):
                        self.assertIsNotNone(projected_adjoint)
                        torch.testing.assert_close(projected_adjoint, k @ adjoint, atol=1e-13, rtol=1e-12)
                        self.assertTrue(torch.isfinite(projected_adjoint).all())
                        failed = captured['failed']
                        if failed.any().item():
                            self.assertIsNot(projected_adjoint, captured['product'])
                            for head in range(k.shape[0]):
                                for group in range(groups):
                                    if not failed[head, group].item():
                                        interval = slice(group * p, min((group + 1) * p, chunks))
                                        torch.testing.assert_close(projected_adjoint[head, interval],
                                                                   captured['product'][head, interval], atol=0, rtol=0)
                        else:
                            self.assertIs(projected_adjoint, captured['product'])
                        return original_gradients(k, w, v, state, adjoint, p,
                                                  projected_adjoint=projected_adjoint)

                    with mock.patch.object(module, 'refined_states', forward), \
                         mock.patch.object(module, 'refined_adjoints', reverse), \
                         mock.patch.object(module, 'reverse_failures', reverse_guard), \
                         mock.patch.object(module, '_factor_gradients', gradients):
                        if actual_guard:
                            actual = module.RefinedStates.apply(*inputs, period, 1e-6)
                        else:
                            with mock.patch.object(module, 'forward_failures', return_value=forward_mask):
                                actual = module.RefinedStates.apply(*inputs, period, 1e-6)
                        upstream = randn(*actual.shape)
                        actual_gradients = torch.autograd.grad((actual * upstream).sum(), inputs)
                    expected = oracle.forward_reference(*references, period)
                    expected = expected + sum(x.sum() * 0 for x in references)
                    expected_gradients = torch.autograd.grad((expected * upstream).sum(), references)
                    torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-12)
                    for a, b in zip(actual_gradients, expected_gradients):
                        self.assertTrue(torch.isfinite(a).all())
                        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-11)

    def test_cpu_reuse_and_refresh_all_factor_gradients(self):
        with torch._dynamo.config.patch(disable=True):
            self.check_reuse('cpu', actual_guard=False)

    @unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA')
    def test_cuda_actual_guards_reuse_and_refresh(self):
        self.check_reuse('cuda', actual_guard=True)

    def test_default_guard_still_returns_only_mask(self):
        with torch._dynamo.config.patch(disable=True):
            _, guard, _ = load_modules()
            k = torch.zeros((1, 3, 2, 5), dtype=torch.float64)
            adjoint = torch.zeros((1, 3, 5, 7), dtype=torch.float64)
            decay = torch.ones((1, 3), dtype=torch.float64)
            mask = torch.zeros((1, 1), dtype=torch.bool)
            with mock.patch.object(guard, '_dispatch', return_value=mask):
                result = guard.reverse_failures(k, k, decay, adjoint, adjoint, 4)
                optional = guard.reverse_failures(k, k, decay, adjoint, adjoint, 4, return_projection=True)
            self.assertIs(result, mask)
            self.assertIs(optional[0], mask)
            torch.testing.assert_close(optional[1], k @ adjoint, atol=0, rtol=0)


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
