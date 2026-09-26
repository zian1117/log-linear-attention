"""Independent products/adjoints for the private chunk-preparation primitive."""
import unittest

import torch

from hattention.preparation_product import preparation_product


class PreparationProductTests(unittest.TestCase):
    def test_cpu_preserves_ordinary_product_and_gradient(self):
        generator = torch.Generator().manual_seed(933)
        for dtype in (torch.float32, torch.float64):
            a = torch.randn(2, 3, 7, 11, generator=generator, dtype=dtype, requires_grad=True)
            b = torch.randn(2, 3, 11, 5, generator=generator, dtype=dtype, requires_grad=True)
            upstream = torch.randn(2, 3, 7, 5, generator=generator, dtype=dtype)
            got = preparation_product(a, b)
            expected = a @ b
            torch.testing.assert_close(got, expected, rtol=0, atol=0)
            actual = torch.autograd.grad(got, (a, b), upstream, retain_graph=True)
            reference = torch.autograd.grad(expected, (a, b), upstream)
            for x, y in zip(actual, reference):
                torch.testing.assert_close(x, y, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_strided_products_and_all_adjoints(self):
        generator = torch.Generator(device='cuda').manual_seed(934)
        for m, k, n in ((7, 17, 5), (64, 128, 64), (33, 65, 19)):
            a = torch.randn(2, 3, k, m, device='cuda', generator=generator).transpose(-1, -2).requires_grad_()
            b = torch.randn(2, 3, n, k, device='cuda', generator=generator).transpose(-1, -2).requires_grad_()
            upstream = torch.randn(2, 3, n, m, device='cuda', generator=generator).transpose(-1, -2)
            got = preparation_product(a, b)
            gradients = torch.autograd.grad(got, (a, b), upstream, retain_graph=True)
            a64, b64 = (x.detach().double().requires_grad_() for x in (a, b))
            expected = a64 @ b64
            expected_gradients = torch.autograd.grad(expected, (a64, b64), upstream.double())
            for actual, reference in zip((got, *gradients), (expected, *expected_gradients)):
                error = (actual.double() - reference).norm() / reference.norm()
                self.assertLess(float(error), 2e-6)
            # Zero strides occur for scalar losses; backward must materialize
            # their expanded upstream gradient before calling the CUDA kernel.
            summed = torch.autograd.grad(got.sum(), (a, b))
            for actual, reference in zip(summed, (torch.ones_like(expected) @ b64.transpose(-1, -2),
                                               a64.transpose(-1, -2) @ torch.ones_like(expected))):
                error = (actual.double() - reference).norm() / reference.norm()
                self.assertLess(float(error), 2e-6)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_compiled_preparation_and_all_six_input_gradients(self):
        from hattention.fast_matrix_gdn import _prepare
        function = getattr(_prepare, '_torchdynamo_orig_callable', _prepare)
        compiled = torch.compile(function, fullgraph=True)
        torch.manual_seed(935)
        r, k, q = (torch.nn.functional.normalize(
            torch.randn(1, 2, 16, 32, device='cuda'), dim=-1) for _ in range(3))
        raw = (r, k, torch.randn(1, 2, 16, 17, device='cuda') * .1,
               torch.full((1, 2, 16), .6, device='cuda'),
               torch.full((1, 2, 16), -.03, device='cuda', dtype=torch.float64).cumsum(-1), q)
        inputs = tuple(x.detach().requires_grad_() for x in raw)
        reference_inputs = tuple(x.detach().double().requires_grad_() for x in raw)
        def flatten(result):
            return tuple(x for x in (*result[:-1], *result[-1]) if x is not None)
        actual = flatten(compiled(*inputs, omit_weighted_keys=True))
        reference = flatten(function(*reference_inputs, omit_weighted_keys=True))
        upstream = tuple(torch.randn_like(x) for x in actual)
        actual_gradients = torch.autograd.grad(actual, inputs, upstream)
        reference_gradients = torch.autograd.grad(reference, reference_inputs,
                                                  tuple(x.double() for x in upstream))
        for got, expected in zip((*actual, *actual_gradients), (*reference, *reference_gradients)):
            relative = (got.double() - expected).norm() / expected.norm().clamp_min(1e-20)
            self.assertLess(float(relative), 1e-5)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_precise_product_remains_fp64(self):
        a = torch.randn(1, 2, 9, 13, device='cuda', dtype=torch.float64, requires_grad=True)
        b = torch.randn(1, 2, 13, 7, device='cuda', dtype=torch.float64, requires_grad=True)
        actual = preparation_product(a, b)
        expected = a @ b
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual_gradients = torch.autograd.grad(actual.sum(), (a, b), retain_graph=True)
        expected_gradients = torch.autograd.grad(expected.sum(), (a, b))
        for x, y in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(x, y, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
