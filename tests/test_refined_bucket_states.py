"""Independent affine-recurrence checks for refined CUDA bucket states."""
import importlib
import unittest
from unittest import mock
import math

import torch
from torch.utils.checkpoint import checkpoint


def affine_reference(k, w, value, decay, period):
    state = k.new_zeros(k.shape[0], k.shape[-1], value.shape[-1])
    identity = torch.eye(k.shape[-1], device=k.device, dtype=k.dtype)
    result = []
    for position in range(k.shape[1]):
        if position % period == 0:
            state = torch.zeros_like(state)
        result.append(state)
        transition = decay[:, position, None, None] * identity - k[:, position].transpose(-1, -2) @ w[:, position]
        state = transition @ state
        if position % period < period // 2:
            state = state + k[:, position].transpose(-1, -2) @ value[:, position]
    # Keep all input derivatives defined even for one chunk (all outputs zero).
    return torch.stack(result, 1) + sum(x.sum() * 0 for x in (k, w, value, decay))


@unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA')
class TestRefinedStates(unittest.TestCase):
    def compare(self, chunks, period, chunk_size, key_dim, value_dim, checkpointed=False, zeros=False):
        refined = importlib.import_module('hattention.refined_bucket_states').RefinedStates
        generator = torch.Generator(device='cuda').manual_seed(723)
        def randn(*shape):
            return torch.randn(*shape, generator=generator, device='cuda', dtype=torch.float64)
        keys = .03 * randn(3, chunks, chunk_size, key_dim)
        if zeros:
            keys.zero_()
        raw = (keys, .1 * keys.clone(), randn(3, chunks, chunk_size, value_dim),
               torch.full((3, chunks), .999, device='cuda', dtype=torch.float64))
        actual_inputs = tuple(x.clone().requires_grad_() for x in raw)
        reference_inputs = tuple(x.clone().requires_grad_() for x in raw)
        if checkpointed:
            actual = checkpoint(refined.apply, *actual_inputs, period, use_reentrant=False)
        else:
            actual = refined.apply(*actual_inputs, period)
        expected = affine_reference(*reference_inputs, period)
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
        upstream = randn(*actual.shape)
        actual_grads = torch.autograd.grad((actual * upstream).sum(), actual_inputs)
        expected_grads = torch.autograd.grad((expected * upstream).sum(), reference_inputs)
        for name, grad, expected_grad in zip(('key', 'erase', 'writes', 'decay'), actual_grads, expected_grads):
            self.assertTrue(torch.isfinite(grad).all(), name)
            torch.testing.assert_close(grad, expected_grad, rtol=1e-10, atol=1e-11, msg=lambda msg: f'{name}: {msg}')
        suppressed = torch.arange(chunks, device='cuda') % period >= period // 2
        self.assertTrue((actual_grads[2][:, suppressed] == 0).all())
        final = torch.arange(chunks, device='cuda') % period == period - 1
        for gradient in actual_grads:
            self.assertTrue((gradient[:, final] == 0).all())

    def test_a_checkpoint_with_compiled_helpers_cold_and_warm(self):
        for _ in range(2):
            self.compare(7, 3, 7, 13, 9, checkpointed=True)

    def test_arbitrary_dimensions_periods_and_factor_gradients(self):
        for shape in ((1, 2, 3, 5, 7), (5, 2, 7, 13, 9), (7, 3, 7, 13, 9),
                      (9, 4, 16, 16, 33), (3, 8, 7, 13, 9), (257, 256, 8, 16, 4)):
            with self.subTest(shape=shape):
                self.compare(*shape)
        self.compare(7, 3, 7, 13, 9, zeros=True)


    def test_router_coarse_floor_and_propagated_residue_against_fp64_scan(self):
        router = importlib.import_module('hattention.bilinear_matrix_gdn')
        refined = importlib.import_module('hattention.refined_bucket_states').RefinedStates
        original = importlib.import_module('hattention.bucket_states').BucketStates
        generator = torch.Generator(device='cuda').manual_seed(861)
        def randn(*shape):
            return torch.randn(*shape, generator=generator, device='cuda')
        cases = [('floor', f, side) for f in (1e-4, 1e-6, 1e-8) for side in (-1, 0, 1)]
        cases.extend((('propagated', 1e-6, 0), ('propagated', 1e-6, 1)))
        for case, floor, side in cases:
            with self.subTest(case=case, floor=floor, side=side):
                length, key_dim, value_dim = (128, 16, 8) if case == 'floor' else (256, 128, 64)
                key = torch.zeros(1, length, 1, key_dim, device='cuda')
                value = torch.zeros(1, length, 1, value_dim, device='cuda')
                beta = torch.zeros(1, length, 1, device='cuda')
                g = torch.zeros_like(beta)
                if case == 'floor':
                    key[..., 0] = 8.
                    key[:, -1, :, 0], key[:, -1, :, 1] = 0., 8.
                    tiny = torch.tensor(floor, device='cuda')
                    if side:
                        tiny = torch.nextafter(tiny, torch.full_like(tiny, math.inf * side))
                    value[:, 0, :, 0], value[:, -1, :, 1] = tiny, 1.
                    beta[:, 0], beta[:, -1] = 1., .5
                    r, u = torch.zeros_like(key), torch.zeros_like(value)
                    r[..., :2], u[..., :2] = 1., 1.
                    q = r.clone()
                    upstream = torch.zeros_like(value)
                    upstream[:, -1, :, 1] = 1.
                else:
                    basis = torch.linalg.qr(randn(key_dim, 2)).Q
                    key[..., :] = basis[:, 0]
                    key[:, 130:] = basis[:, 1]
                    value[:, :128] = .1 * randn(1, 128, 1, value_dim)
                    beta[:, :130], beta[:, 130:] = 1., .5 * side
                    g.fill_(-.001)
                    r, q, u = randn(*key.shape), randn(*key.shape), randn(*value.shape)
                    upstream = randn(*value.shape)
                raw = (r, key, value, g, beta, u, q,
                       torch.full((1,), .5*math.log(key_dim*value_dim), device='cuda'))
                actual_inputs = tuple(x.clone().requires_grad_() for x in raw)
                expected_inputs = tuple(x.clone().requires_grad_() for x in raw)
                # Keep each backend patched through backward: checkpoint
                # recomputation must use the same scan as its forward.
                with mock.patch.object(router, 'BucketStates', refined):
                    actual = router.precise_bilinear_matrix_gdn(*actual_inputs, norm_floor=floor)
                    actual_grads = torch.autograd.grad((actual*upstream).sum(), actual_inputs)
                with mock.patch.object(router, 'BucketStates', original):
                    expected = router.precise_bilinear_matrix_gdn(*expected_inputs, norm_floor=floor)
                    expected_grads = torch.autograd.grad((expected*upstream).sum(), expected_inputs)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
                for name, grad, expected_grad in zip(('r', 'k', 'v', 'g', 'beta', 'u', 'q', 'temperature'), actual_grads, expected_grads):
                    self.assertTrue(torch.isfinite(grad).all(), name)
                    torch.testing.assert_close(grad, expected_grad, rtol=1e-4, atol=2e-7, msg=lambda msg: f'{name}: {msg}')


if __name__ == '__main__':
    unittest.main(verbosity=2)
