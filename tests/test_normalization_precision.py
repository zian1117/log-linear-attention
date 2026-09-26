"""GDN normalization must merge FP32 gradient branches before casting to BF16.

These tests exercise the shared normalizer used by both bilinear entrypoints,
not a copied implementation. The oracle differentiates in FP64. An almost
radial upstream gradient exposes loss of the surviving tangential gradient
when numerator and denominator branches are rounded to BF16 separately.
"""
import importlib
from pathlib import Path
import sys
import types
import unittest

import torch


def load_normalizer():
    # Import numerical code without optional model/CUDA extension imports.
    package_name = '_normalization_precision_tests'
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(Path(__file__).resolve().parents[1] / 'hattention')]
        sys.modules[package_name] = package
    module = importlib.import_module(package_name + '.bilinear_matrix_gdn')
    return module._gdn_normalize


def fixture(scale, kind, device='cpu'):
    generator = torch.Generator().manual_seed(921)
    value = (scale * torch.randn(128, generator=generator)).to(torch.bfloat16)
    upstream = torch.randn(128, generator=generator)
    if kind == 'almost_radial':
        upstream = value.float() / scale + .003 * upstream
    return value.to(device), upstream.to(device)


def fp64_gradient(value, upstream):
    reference = value.double().requires_grad_()
    result = reference * torch.rsqrt(reference.square().sum(-1, keepdim=True) + 1e-6)
    return torch.autograd.grad((result * upstream.double()).sum(), reference)[0]


class TestNormalizationPrecision(unittest.TestCase):
    def check_case(self, normalize, scale, kind, device, compiled=False):
        value, upstream = fixture(scale, kind, device)
        source = value.clone().requires_grad_()
        output = normalize(source)
        previous_output = value.float() * torch.rsqrt(
            value.float().square().sum(-1, keepdim=True) + 1e-6)
        if compiled:
            # Compiler fusion can change normal FP32 rounding. Eager forward
            # must remain bitwise identical to the original implementation.
            torch.testing.assert_close(output, previous_output, rtol=1e-6, atol=1e-7)
        else:
            self.assertTrue(torch.equal(output, previous_output))
        gradient = torch.autograd.grad((output * upstream).sum(), source)[0]
        reference = fp64_gradient(value, upstream)
        self.assertEqual(gradient.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(gradient).all())
        error = (gradient.double() - reference).norm()
        self.assertLess(error.item(), .008 * reference.norm().item())

    def test_cpu_forward_and_gradients(self):
        normalize = load_normalizer()
        for scale in (1., .05):
            for kind in ('ordinary', 'almost_radial'):
                with self.subTest(scale=scale, kind=kind):
                    self.check_case(normalize, scale, kind, 'cpu')

    def test_fixture_detects_separate_bf16_gradient_rounding(self):
        value, upstream = fixture(.05, 'almost_radial')
        source = value.clone().requires_grad_()
        # Deliberately reproduce the former bug, independent of the helper.
        output = source.float() * torch.rsqrt(
            source.float().square().sum(-1, keepdim=True) + 1e-6)
        gradient = torch.autograd.grad((output * upstream).sum(), source)[0]
        reference = fp64_gradient(value, upstream)
        self.assertGreater(reference.norm().item(), .07)
        self.assertGreater((gradient.double() - reference).norm().item(), .02)

    @unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA')
    def test_cuda_eager_and_compiled(self):
        eager = load_normalizer()
        compiled = torch.compile(eager, fullgraph=True)
        for use_compile, normalize in ((False, eager), (True, compiled)):
            for scale in (1., .05):
                for kind in ('ordinary', 'almost_radial'):
                    with self.subTest(compiled=use_compile, scale=scale, kind=kind):
                        self.check_case(normalize, scale, kind, 'cuda', compiled=use_compile)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
