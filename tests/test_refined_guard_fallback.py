"""Guarded forward/adjoint repair isolates rejected numerical results."""
import importlib
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import torch


def load_module():
    name = '_refined_guard_tests'
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(Path(__file__).resolve().parents[1] / 'hattention')]
        sys.modules[name] = package
    return importlib.import_module(name + '.refined_bucket_states')


def forward_reference(k, w, value, decay, period):
    state = k.new_zeros(k.shape[0], k.shape[-1], value.shape[-1])
    eye = torch.eye(k.shape[-1], dtype=k.dtype, device=k.device)
    result = []
    for n in range(k.shape[1]):
        if n % period == 0:
            state = torch.zeros_like(state)
        result.append(state)
        state = (decay[:, n, None, None] * eye - k[:, n].transpose(-1, -2) @ w[:, n]) @ state
        if n % period < period // 2:
            state = state + k[:, n].transpose(-1, -2) @ value[:, n]
    return torch.stack(result, 1)


def reverse_reference(k, w, decay, direct, period):
    current = torch.zeros_like(direct[:, 0])
    eye = torch.eye(k.shape[-1], dtype=k.dtype, device=k.device)
    result = [None] * k.shape[1]
    for n in range(k.shape[1] - 1, -1, -1):
        if n == k.shape[1] - 1 or (n + 1) % period == 0:
            current = torch.zeros_like(current)
        result[n] = current
        matrix = decay[:, n, None, None] * eye - k[:, n].transpose(-1, -2) @ w[:, n]
        current = matrix.transpose(-1, -2) @ current + direct[:, n]
    return torch.stack(result, 1)


class TestRefinedGuardFallback(unittest.TestCase):
    def compare(self, device, actual_guard):
        module = load_module()
        generator = torch.Generator(device=device).manual_seed(163)
        def randn(*shape):
            return torch.randn(*shape, dtype=torch.float64, device=device, generator=generator)
        k = .01 * randn(2, 7, 5, 7)
        raw = (k, .1 * k, randn(2, 7, 5, 9), torch.full((2, 7), .9, dtype=torch.float64, device=device))
        actual_inputs = tuple(x.clone().requires_grad_() for x in raw)
        expected_inputs = tuple(x.clone().requires_grad_() for x in raw)
        forward_mask = torch.tensor([[True, False, False], [False, False, True]], device=device)
        reverse_mask = torch.tensor([[False, True, False], [True, False, False]], device=device)
        def bad_forward(k, w, v, a, period, refinements=2):
            out = forward_reference(k, w, v, a, period)
            out[0, 1] = torch.nan
            out[1, 6] = .5  # Corrupt a fixed origin in the partial final period.
            return out
        def bad_reverse(k, w, a, direct, period, refinements=2):
            out = reverse_reference(k, w, a, direct, period)
            out[0, 3] = torch.nan
            out[1, 0] += .5
            return out
        with mock.patch.object(module, 'refined_states', bad_forward), mock.patch.object(module, 'refined_adjoints', bad_reverse):
            if actual_guard:
                actual = module.RefinedStates.apply(*actual_inputs, 3, 1e-6)
                upstream = randn(*actual.shape)
                actual_gradients = torch.autograd.grad((actual * upstream).sum(), actual_inputs)
            else:
                with mock.patch.object(module, 'forward_failures', return_value=forward_mask), mock.patch.object(module, 'reverse_failures', return_value=reverse_mask):
                    actual = module.RefinedStates.apply(*actual_inputs, 3, 1e-6)
                    upstream = randn(*actual.shape)
                    actual_gradients = torch.autograd.grad((actual * upstream).sum(), actual_inputs)
        expected = forward_reference(*expected_inputs, 3)
        expected_gradients = torch.autograd.grad((expected * upstream).sum(), expected_inputs)
        torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-12)
        for got, want in zip(actual_gradients, expected_gradients):
            self.assertTrue(torch.isfinite(got).all())
            torch.testing.assert_close(got, want, atol=1e-12, rtol=1e-11)

    def test_cpu_mixed_period_replacement_and_nan_isolation(self):
        with torch._dynamo.config.patch(disable=True):
            self.compare('cpu', actual_guard=False)

    @unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA')
    def test_cuda_actual_guard_repairs_both_scans(self):
        self.compare('cuda', actual_guard=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
