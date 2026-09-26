"""Cross-level norm reuse versus independent energy and token recurrences."""
import importlib
from pathlib import Path
import sys
import types
import unittest

import torch
import torch.nn.functional as F


def modules():
    name = '_shared_norm_tests'
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(Path(__file__).resolve().parents[1] / 'hattention')]
        sys.modules[name] = package
    return tuple(importlib.import_module(name + '.' + module)
                 for module in ('shared_local_norm', 'energy_bucket_norm', 'bucket_frobenius'))


def token_matrices(k, v, beta, g, level):
    period = 1 << level
    state = k.new_zeros(*k.shape[:-2], k.shape[-1], v.shape[-1])
    matrices = []
    for t in range(k.shape[-2]):
        if t % period == 0:
            state = torch.zeros_like(state)
        state = state * g[..., t, None, None].exp()
        residual = -(k[..., t, None, :] @ state).squeeze(-2)
        if level == 0 or t % period < period // 2:
            residual = residual + v[..., t, :]
        state = state + k[..., t, :, None] * (beta[..., t, None] * residual).unsqueeze(-2)
        matrices.append(state)
    return torch.stack(matrices, dim=-3)


class TestSharedLocalNorm(unittest.TestCase):
    def compare(self, chunk, mode, device='cpu'):
        shared, energy, gram = modules()
        generator = torch.Generator(device=device).manual_seed(829 + chunk)
        def randn(*shape):
            return torch.randn(*shape, dtype=torch.float64, device=device, generator=generator)
        key = F.normalize(randn(2, 2, chunk, 7), dim=-1)
        value = .1 * randn(2, 2, chunk, 5)
        beta = torch.sigmoid(randn(2, 2, chunk))
        g = -.05 * torch.sigmoid(randn(2, 2, chunk))
        if mode == 'collinear':
            key = key[..., :1, :].expand_as(key).clone()
            beta.fill_(.999)
        elif mode == 'zero_beta':
            beta[..., ::3] = 0.
        elif mode == 'padded':
            key[:, -1, -3:] = 0.
            value[:, -1, -3:] = 0.
            beta[:, -1, -3:] = 0.
            g[:, -1, -3:] = 0.
        inputs = tuple(x.requires_grad_() for x in (key, value, beta, g))
        key, value, beta, g = inputs
        gc = g.cumsum(-1)
        terms = energy._right_terms(key, beta, gc)
        actual = shared.shared_local_norms(key, value, beta, gc, terms=terms)
        external = []
        for level in range(1, chunk.bit_length()):
            period = 1 << level
            half = period // 2
            inv, decay = (gram._segment_blocks(x, period) for x in terms[1:3])
            val = value.reshape(*value.shape[:-2], chunk // period, period, value.shape[-1])
            external.append((inv[..., half:, :half] * decay[..., half:, :half]) @ val[..., :half, :])
        supplied = shared.shared_local_norms(key, value, beta, gc, terms=terms, active_residuals=external)
        plain = shared.shared_local_norms(key, value, beta, gc)
        subset = shared.shared_local_norms(key, value, beta, gc, terms=terms, max_levels=2)
        objectives = [key.sum() * 0 for _ in range(4)]
        floor = 1e-6
        for level, ((norm2, bound), (from_external, external_bound), (from_plain, _)) in enumerate(zip(actual, supplied, plain)):
            self.assertFalse(bound.requires_grad)
            # Different supplied projection layouts can choose a different
            # FP64 GEMM reduction order. Scale tolerance by the actual terms,
            # including cancellation, rather than demand bitwise equality.
            reduction_roundoff = torch.finfo(norm2.dtype).eps * chunk
            absolute_roundoff = reduction_roundoff * bound.detach().abs().max().item()
            torch.testing.assert_close(from_external, norm2, atol=absolute_roundoff, rtol=reduction_roundoff)
            torch.testing.assert_close(external_bound, bound, atol=absolute_roundoff, rtol=reduction_roundoff)
            torch.testing.assert_close(from_plain, norm2, atol=0, rtol=0)
            if level < 2:
                torch.testing.assert_close(subset[level][0], norm2, atol=0, rtol=0)
            matrix = token_matrices(key, value, beta, g, level)
            direct = matrix.square().sum((-2, -1))
            torch.testing.assert_close(norm2, direct, atol=3e-13, rtol=2e-10)
            old = energy.local_bucket_norm2(key, value, beta, gc, level, terms=terms, norm_floor=floor)
            if mode != 'collinear':
                torch.testing.assert_close(norm2, old, atol=3e-13, rtol=2e-10)
            if level:
                period = 1 << level
                shape = (*beta.shape[:-1], chunk // period, period)
                args = (bound.reshape(shape), key.reshape(*shape, key.shape[-1]),
                        beta.reshape(shape), gc.reshape(shape))
                kw = dict(v=value.reshape(*shape, value.shape[-1]), norm_floor=floor)
                norm2 = gram._repair_cancellation(norm2.reshape(shape), *args, **kw).reshape_as(beta)
                from_external = gram._repair_cancellation(from_external.reshape(shape), *args, **kw).reshape_as(beta)
            query = F.normalize(randn(*key.shape), dim=-1)
            route = F.normalize(randn(*value.shape), dim=-1)
            numerator = torch.einsum('...tk,...tkv,...tv->...t', query, matrix, route)
            upstream = randn(*beta.shape)
            for index, denominator in enumerate((norm2, from_external, old, direct)):
                objectives[index] = objectives[index] + (numerator * denominator.clamp_min(floor ** 2).rsqrt() * upstream).sum()
        gradients = [torch.autograd.grad(loss, inputs, retain_graph=True) for loss in objectives]
        for index in (1, 2, 3):
            for label, got, expected in zip(('key', 'value', 'beta', 'log_decay'), gradients[0], gradients[index]):
                self.assertTrue(torch.isfinite(got).all(), (chunk, mode, label))
                torch.testing.assert_close(got, expected, atol=5e-6 if mode == 'collinear' else 2e-9,
                                           rtol=2e-6 if mode == 'collinear' else 2e-8,
                                           msg=lambda m: f'{chunk}, {mode}, {label}, reference={index}: {m}')

    def test_values_and_joint_cross_level_gradients(self):
        with torch._dynamo.config.patch(disable=True):
            for chunk in (8, 16, 64):
                for mode in ('random', 'zero_beta', 'collinear', 'padded'):
                    with self.subTest(chunk=chunk, mode=mode):
                        self.compare(chunk, mode)

    @unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA')
    def test_cuda_values_and_joint_cross_level_gradients(self):
        with torch._dynamo.config.patch(disable=True):
            for chunk in (8, 16, 64):
                for mode in ('random', 'zero_beta', 'collinear', 'padded'):
                    with self.subTest(chunk=chunk, mode=mode):
                        self.compare(chunk, mode, device='cuda')


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
