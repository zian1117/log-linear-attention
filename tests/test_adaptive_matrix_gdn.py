"""Check per-head adaptive replacement and isolation of rejected NaN gradients."""
import importlib
import math
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import torch


GPU_AUDIT_RESULTS = []


class TestAdaptiveGradientIsolation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        package_name = '_adaptive_isolation_tests'
        package = types.ModuleType(package_name)
        package.__path__ = [str(Path(__file__).resolve().parents[1] / 'hattention')]
        fast_stub = types.ModuleType(package_name + '.fast_matrix_gdn')
        fast_stub.fast_matrix_gdn = None
        sys.modules.setdefault(package_name, package)
        sys.modules.setdefault(fast_stub.__name__, fast_stub)
        cls.module = importlib.import_module(package_name + '.adaptive_matrix_gdn')

    @staticmethod
    def branch(inputs, precise):
        r, k, v, g, beta, u, q, temperature = inputs
        scalar = r.sum(-1) + .7 * k.sum(-1) + q.sum(-1) + g + beta + temperature[None, None, :]
        if precise:
            return v * scalar.tanh().unsqueeze(-1) + .3 * u.square()
        return v * scalar.sin().unsqueeze(-1) + .2 * u

    def test_mixed_batch_heads_reject_nan_gradients_before_shared_parameters(self):
        generator = torch.Generator().manual_seed(502)
        batch, length, heads, key_dim, value_dim = 2, 17, 3, 3, 5
        def randn(*shape):
            return torch.randn(*shape, generator=generator, dtype=torch.float64)
        raw = (
            randn(batch, length, heads, key_dim), randn(batch, length, heads, key_dim),
            randn(batch, length, heads, value_dim), randn(batch, length, heads),
            randn(batch, length, heads), randn(batch, length, heads, value_dim),
            randn(batch, length, heads, key_dim), randn(heads),
        )
        hidden = randn(batch, length, heads, 7)
        raw_projection = randn(7, value_dim)
        upstream = randn(batch, length, heads, value_dim)
        selections = (
            torch.tensor([False] * (batch * heads)),
            torch.tensor([True] * (batch * heads)),
            torch.tensor([False, True, False, True, False, True]),
        )
        for selected in selections:
            with self.subTest(selected=selected.tolist()):
                actual_inputs = [x.clone().requires_grad_() for x in raw]
                expected_inputs = [x.clone().requires_grad_() for x in raw]
                actual_projection = raw_projection.clone().requires_grad_()
                expected_projection = raw_projection.clone().requires_grad_()
                actual_inputs[5] = actual_inputs[5] + hidden @ actual_projection
                expected_inputs[5] = expected_inputs[5] + hidden @ expected_projection
                calls = []

                def fast(*inputs, **kwargs):
                    output = self.branch(inputs, precise=False)
                    # The rejected branch has genuinely NaN local derivatives,
                    # not merely a NaN forward constant with no gradient path.
                    factor = torch.where(selected, torch.nan, 1.).to(output.dtype)
                    return output * factor[None, None, :, None], selected

                def precise(*inputs, **kwargs):
                    calls.append(inputs[0].shape[2])
                    return self.branch(inputs, precise=True)

                with mock.patch.object(self.module, '_fast', fast), mock.patch.object(self.module, '_precise', precise):
                    actual = self.module.adaptive_matrix_gdn(*actual_inputs)
                mask = selected.reshape(batch, heads)[:, None, :, None]
                expected = torch.where(mask, self.branch(expected_inputs, precise=True),
                                       self.branch(expected_inputs, precise=False))
                torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
                actual_grads = torch.autograd.grad((actual * upstream).sum(), (*actual_inputs, actual_projection))
                expected_grads = torch.autograd.grad((expected * upstream).sum(), (*expected_inputs, expected_projection))
                for name, grad, reference in zip(('r', 'k', 'v', 'g', 'beta', 'u', 'q', 'shared_temperature', 'shared_projection'), actual_grads, expected_grads):
                    self.assertTrue(torch.isfinite(grad).all(), name)
                    torch.testing.assert_close(grad, reference, rtol=1e-11, atol=1e-11, msg=lambda msg: f'{name}: {msg}')
                self.assertEqual(calls, [int(selected.sum())] if selected.any() else [])

    def test_output_replacement_alone_leaks_nan_gradients(self):
        # Preserve the failure mechanism that the input barrier prevents.
        x = torch.tensor([2., 3.], requires_grad=True)
        fast = x * torch.tensor([1., torch.nan])
        output = fast.index_copy(0, torch.tensor([1]), x[1:].square())
        self.assertTrue(torch.isfinite(output).all())
        gradient, = torch.autograd.grad(output.sum(), x)
        self.assertTrue(torch.isnan(gradient[1]))


@unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA')
class TestAdaptiveGPUAgainstPrecise(unittest.TestCase):
    def test_norm_rounding_onto_floor_requires_precise_backward(self):
        adaptive_module = importlib.import_module('hattention.adaptive_matrix_gdn')
        precise = importlib.import_module('hattention.bilinear_matrix_gdn').precise_bilinear_matrix_gdn
        fast = importlib.import_module('hattention.fast_matrix_gdn').fast_matrix_gdn
        for bucket, length in (('current', 16), ('local_old', 16), ('coarse_old', 128)):
            for floor in (1e-4, 1e-6, 1e-8):
                for side in ('below', 'fp32_rounding', 'above'):
                    with self.subTest(bucket=bucket, floor=floor, side=side):
                        batch, heads, key_dim, value_dim = 1, 1, 16, 8
                        key = torch.zeros(batch, length, heads, key_dim, device='cuda')
                        key[..., 0] = 8.
                        key[:, -1, :, 0] = 0.
                        key[:, -1, :, 1] = 8.
                        beta = torch.zeros(batch, length, heads, device='cuda')
                        value = torch.zeros(batch, length, heads, value_dim, device='cuda')
                        near_floor = torch.tensor(floor, device='cuda', dtype=torch.float32)
                        if side != 'fp32_rounding':
                            toward = -torch.inf if side == 'below' else torch.inf
                            near_floor = torch.nextafter(near_floor, torch.full_like(near_floor, toward))
                        if bucket == 'current':
                            beta[:, 0], beta[:, -1] = .5, 1.
                            value[:, 0, :, 1], value[:, -1, :, 0] = 1., near_floor
                            source = length - 1
                        else:
                            beta[:, 0], beta[:, -1] = 1., .5
                            value[:, 0, :, 0], value[:, -1, :, 1] = near_floor, 1.
                            source = 0
                        r, u = torch.zeros_like(key), torch.zeros_like(value)
                        r[..., :2], u[..., :2] = 1., 1.
                        raw = (r, key, value, torch.zeros_like(beta), beta, u, r.clone(),
                               torch.full((heads,), math.log(math.sqrt(key_dim * value_dim)), device='cuda'))
                        actual_inputs = tuple(x.detach().requires_grad_() for x in raw)
                        reference_inputs = tuple(x.detach().clone().requires_grad_() for x in raw)
                        with torch.no_grad():
                            _, flagged = fast(*actual_inputs, norm_floor=floor)
                        self.assertTrue(flagged[0], 'Near-floor rounding must not silently choose a different derivative branch')
                        actual = adaptive_module.adaptive_matrix_gdn(*actual_inputs, norm_floor=floor)
                        expected = precise(*reference_inputs, norm_floor=floor)
                        actual_grads = torch.autograd.grad(actual[:, -1, :, 1].sum(), actual_inputs)
                        expected_grads = torch.autograd.grad(expected[:, -1, :, 1].sum(), reference_inputs)
                        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
                        if side == 'below':
                            self.assertGreater(abs(expected_grads[4][0, source, 0].item()), .01)
                        elif side == 'above':
                            self.assertLess(abs(expected_grads[4][0, source, 0].item()), 1e-4)
                        for name, grad, ref_grad in zip(('r', 'k', 'v', 'g', 'beta', 'u', 'q', 'log_temperature'), actual_grads, expected_grads):
                            self.assertTrue(torch.isfinite(grad).all(), name)
                            torch.testing.assert_close(grad, ref_grad, rtol=1e-5, atol=1e-6, msg=lambda msg: f'{name}: {msg}')

    def test_large_initial_decay_preserves_later_relative_decays(self):
        adaptive = importlib.import_module('hattention.adaptive_matrix_gdn').adaptive_matrix_gdn
        precise = importlib.import_module('hattention.bilinear_matrix_gdn').precise_bilinear_matrix_gdn
        generator = torch.Generator(device='cuda').manual_seed(411)
        batch, length, heads, key_dim, value_dim = 1, 64, 1, 16, 8
        def randn(*shape):
            return torch.randn(*shape, generator=generator, device='cuda')
        k = randn(batch, length, heads, key_dim)
        v = randn(batch, length, heads, value_dim) * .1
        r, q = randn(*k.shape), randn(*k.shape)
        u = randn(*v.shape)
        beta = torch.full((batch, length, heads), .1, device='cuda')
        upstream = randn(*v.shape)
        # A very negative first gate erases the previous state, but must not
        # quantize the later relative decays within this same chunk.
        for prefix, later in ((100., .1), (1000., .001), (1e6, .1), (1e6, .001)):
            with self.subTest(prefix=prefix, later=later):
                g = torch.full_like(beta, -later)
                g[:, 0] = -prefix
                raw = (r, k, v, g, beta, u, q,
                       torch.full((heads,), math.log(math.sqrt(key_dim * value_dim)), device='cuda'))
                actual_inputs = tuple(x.detach().clone().requires_grad_() for x in raw)
                reference_inputs = tuple(x.detach().clone().requires_grad_() for x in raw)
                actual, expected = adaptive(*actual_inputs), precise(*reference_inputs)
                actual_grads = torch.autograd.grad((actual * upstream).sum(), actual_inputs)
                expected_grads = torch.autograd.grad((expected * upstream).sum(), reference_inputs)
                torch.testing.assert_close(actual, expected, rtol=5e-4, atol=3e-6)
                for name, grad, reference in zip(('r', 'k', 'v', 'g', 'beta', 'u', 'q', 'log_temperature'), actual_grads, expected_grads):
                    self.assertTrue(torch.isfinite(grad).all(), name)
                    relative = (grad.double()-reference.double()).norm()/reference.double().norm().clamp_min(1e-8)
                    self.assertLess(relative.item(), .002, name)

    @staticmethod
    def inputs(case, dtype):
        generator = torch.Generator(device='cuda').manual_seed(816)
        batch, length, heads, key_dim, value_dim = 2, 128, 3, 16, 8
        if case.startswith('propagated_residue'):
            batch, length, heads, key_dim, value_dim = 1, 256, 1, 128, 64
        def randn(*shape):
            return torch.randn(*shape, generator=generator, device='cuda', dtype=torch.float32)
        key = randn(batch, length, heads, key_dim)
        values = randn(batch, length, heads, value_dim) * .1
        beta = torch.full((batch, length, heads), .1, device='cuda')
        g = torch.full_like(beta, -.02)
        if case == 'collinear':
            key = key[:, :1].expand_as(key).clone()
            beta.fill_(.999)
        elif case == 'mixed':
            for b, h in ((0, 1), (1, 2)):
                key[b, :, h] = key[b, :1, h].expand(length, -1)
                beta[b, :, h] = .999
        elif case.startswith('propagated_residue'):
            basis = torch.linalg.qr(randn(key_dim, 2)).Q
            key = basis[:, 0].expand(length, key_dim).clone()
            key[130:] = basis[:, 1]
            key = key.reshape(batch, length, heads, key_dim)
            beta.fill_(1.)
            beta[:, 130:] = 0. if case.endswith('no_updates') else .5
            values[:, 128:] = 0.
            g.fill_(-.001)
        args = (
            randn(batch, length, heads, key_dim), key, values, g, beta,
            randn(batch, length, heads, value_dim), randn(batch, length, heads, key_dim),
            torch.full((heads,), math.log(math.sqrt(key_dim * value_dim)), device='cuda'),
        )
        # Preserve FP32 decay/gates/temperature, as in mixed-precision training.
        return tuple(x.to(dtype if i in (0, 1, 2, 5, 6) else torch.float32).detach().requires_grad_()
                     for i, x in enumerate(args))

    def test_outputs_all_gradients_and_precision_selection(self):
        adaptive_module = importlib.import_module('hattention.adaptive_matrix_gdn')
        precise = importlib.import_module('hattention.bilinear_matrix_gdn').precise_bilinear_matrix_gdn
        fast = importlib.import_module('hattention.fast_matrix_gdn').fast_matrix_gdn
        for case in ('normal', 'collinear', 'mixed', 'propagated_residue_no_updates',
                     'propagated_residue_orthogonal_updates'):
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(case=case, dtype=dtype):
                    actual_inputs = self.inputs(case, dtype)
                    reference_inputs = tuple(x.detach().clone().requires_grad_() for x in actual_inputs)
                    with torch.no_grad():
                        _, flagged = fast(*actual_inputs)
                    if case == 'mixed':
                        self.assertTrue(flagged.any())
                        self.assertTrue((~flagged).any())
                        self.assertTrue(flagged.reshape(2, 3)[0, 1])
                        self.assertTrue(flagged.reshape(2, 3)[1, 2])
                    elif case != 'normal':
                        self.assertTrue(flagged.all())
                    actual = adaptive_module.adaptive_matrix_gdn(*actual_inputs)
                    expected = precise(*reference_inputs)
                    generator = torch.Generator(device='cuda').manual_seed(922)
                    upstream = torch.randn(actual.shape, generator=generator, device='cuda', dtype=torch.float32)
                    actual_grads = torch.autograd.grad((actual.float() * upstream).sum(), actual_inputs)
                    expected_grads = torch.autograd.grad((expected.float() * upstream).sum(), reference_inputs)
                    row = {
                        'case': case, 'dtype': str(dtype), 'flags': flagged.tolist(),
                        'forward_max_abs': (actual.float()-expected.float()).abs().max().item(),
                        'forward_relative_l2': ((actual.float()-expected.float()).norm()/expected.float().norm().clamp_min(1e-20)).item(),
                        'gradients': {},
                    }
                    for name, grad, ref_grad in zip(('r', 'k', 'v', 'g', 'beta', 'u', 'q', 'log_temperature'), actual_grads, expected_grads):
                        relative = ((grad.double()-ref_grad.double()).norm()/ref_grad.double().norm().clamp_min(1e-8)).item()
                        row['gradients'][name] = {'relative_l2': relative, 'finite': torch.isfinite(grad).all().item()}
                    GPU_AUDIT_RESULTS.append(row)
                    self.assertTrue(torch.isfinite(actual).all())
                    torch.testing.assert_close(actual.float(), expected.float(), rtol=.02 if dtype == torch.bfloat16 else 5e-4,
                                               atol=.002 if dtype == torch.bfloat16 else 3e-6)
                    for name, metrics in row['gradients'].items():
                        self.assertTrue(metrics['finite'], name)
                        self.assertLess(metrics['relative_l2'], .015 if dtype == torch.bfloat16 else .002, name)


if __name__ == '__main__':
    unittest.main(verbosity=2)
