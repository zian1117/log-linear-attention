"""CPU checks of router derivatives at unusual but valid model inputs.

Native CUDA kernels have separate GPU coverage. Here only the boundary scan
and final reduction are replaced by differentiable CPU implementations; the
normalization, FP32 factors, masks and precision-repair orchestration are real.
"""
import math
import unittest
from unittest import mock

import torch
import test_period_repair as period_tests


class TestRouterBoundaryAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        period_tests.TestPeriodReplacementCPU.setUpClass()
        cls.fast = period_tests.TestPeriodReplacementCPU.fast
        cls.oracle = period_tests.TestPeriodReplacementCPU.oracle

    @classmethod
    def tearDownClass(cls):
        period_tests.TestPeriodReplacementCPU.tearDownClass()

    def test_odd_dimensions_padding_zero_gates_and_tiny_routing_vectors(self):
        generator = torch.Generator().manual_seed(87513)
        def normal(*shape):
            return torch.randn(shape, generator=generator)
        cases = (
            (1, 1, 1, 'zero_beta'),
            (2, 3, 5, 'full_erase'),
            (31, 7, 3, 'tiny_router'),
            (65, 3, 9, 'large_decay_prefix'),
            (130, 17, 5, 'mixed'),
        )
        for length, key_dim, value_dim, mode in cases:
            with self.subTest(length=length, key_dim=key_dim, mode=mode):
                shape = (1, length, 2)
                r, k, q = (normal(*shape, key_dim) for _ in range(3))
                v, u = (normal(*shape, value_dim) for _ in range(2))
                beta = torch.rand(shape, generator=generator)
                g = -.02 * torch.rand(shape, generator=generator)
                if mode == 'zero_beta':
                    beta.zero_()
                elif mode == 'full_erase':
                    k[:] = k[:, :1]
                    beta.fill_(1)
                elif mode == 'tiny_router':
                    q *= 1e-8
                    u *= 1e-8
                    u[:, :, 1].zero_()
                elif mode == 'large_decay_prefix':
                    g[:, 0] = -1e6
                    g[:, 1:] = -1e-5
                elif mode == 'mixed':
                    beta[:, ::3] = 0
                    beta[:, 1::3] = 1
                    g[:, ::17] = -80
                    g[:, 64] = -1e6
                    g[:, 65:128] = -1e-5
                    q[:, ::5] *= 1e-8
                    u[:, ::7] *= 1e-8
                raw = (r, k, v, g, beta, u, q,
                       torch.full((2,), .5 * math.log(key_dim * value_dim)))
                actual_inputs = tuple(x.clone().requires_grad_() for x in raw)
                expected_inputs = tuple(x.double().requires_grad_() for x in raw)
                with mock.patch.object(self.fast._FloatStates, 'apply', period_tests.TestPeriodReplacementCPU.float_states), \
                     mock.patch.object(self.fast._RoutingReduce, 'apply', period_tests.TestPeriodReplacementCPU.reduce):
                    actual, flags = self.fast.fast_matrix_gdn(*actual_inputs, repair_periods=True)
                    self.assertFalse(flags.any())
                    expected = self.oracle.full_matrix_reference(
                        *expected_inputs, gdn_norm_dtype=torch.float32)
                    upstream = normal(*actual.shape)
                    actual_grad = torch.autograd.grad((actual * upstream).sum(), actual_inputs)
                    expected_grad = torch.autograd.grad((expected * upstream.double()).sum(), expected_inputs, allow_unused=True)
                    expected_grad = tuple(torch.zeros_like(x) if grad is None else grad
                                          for x, grad in zip(expected_inputs, expected_grad))
                torch.testing.assert_close(actual.double(), expected, rtol=5e-4, atol=3e-6)
                for name, actual_g, expected_g in zip(
                        ('r', 'k', 'v', 'g', 'beta', 'u', 'q', 'temperature'), actual_grad, expected_grad):
                    self.assertTrue(torch.isfinite(actual_g).all(), name)
                    # Absolute error for genuinely zero derivatives; relative
                    # error alone is meaningless at exact cancellation.
                    error = (actual_g.double() - expected_g).norm()
                    self.assertLess(error.item(), 3e-6 + .002 * expected_g.norm().item(), name)


if __name__ == '__main__':
    unittest.main(verbosity=2)
