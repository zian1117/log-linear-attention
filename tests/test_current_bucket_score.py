"""Rank-one score derivatives, including small sigmoid gates after RMSNorm."""
import importlib
import importlib.util
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

_spec = importlib.util.spec_from_file_location(
    '_current_score', Path(__file__).resolve().parents[1] / 'hattention/current_bucket_score.py')
_helper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helper)
current_bucket_score = _helper.current_bucket_score


def explicit_score(k, v, beta, u, q, temperature, floor):
    state = beta[..., None, None] * k[..., :, None] * v[..., None, :]
    norm2 = state.square().sum((-2, -1))
    return (temperature * (state * q[..., :, None] * u[..., None, :]).sum((-2, -1))
            * norm2.clamp_min(floor * floor).rsqrt())


def cpu_reduce(values, scores, scale):
    values, scores = torch.stack(values, -2).float(), torch.stack(scores, -1)
    _, chunks, chunk, levels = scores.shape
    position = torch.arange(chunks * chunk).reshape(chunks, chunk, 1)
    level = torch.arange(levels)
    active = (level == 0) | (((position >> (level - 1).clamp_min(0)) & 1) != 0)
    masked = scores.masked_fill(~active, -torch.inf)
    weights = (masked - masked.amax(-1, keepdim=True)).float().softmax(-1)
    return (values * weights[..., None]).sum(-2) * scale


class TestCurrentBucketScore(unittest.TestCase):
    def check_formula(self, device):
        generator = torch.Generator().manual_seed(645)
        for dtype in (torch.float32, torch.float64):
            for floor in (1e-6, 2.**-20, .001):
                with self.subTest(device=device, dtype=dtype, floor=floor):
                    count, key_dim, value_dim = 12, 5, 7
                    key = torch.randn(count, key_dim, dtype=dtype, generator=generator).to(device)
                    key = key / key.norm(dim=-1, keepdim=True)
                    value = torch.randn(count, value_dim, dtype=dtype, generator=generator).to(device)
                    scales = torch.tensor([0., .2, .8, 1.2, 2., -2., -.2, 5., 1e-8, 1., 1., 1.],
                                          dtype=dtype, device=device)
                    beta = scales * floor / (key.norm(dim=-1) * value.norm(dim=-1))
                    key[-3] = 0; value[-2] = 0; key[-1] = 0; value[-1] = 0
                    raw = (key, value, beta,
                           torch.randn(count, value_dim, dtype=dtype, generator=generator).to(device),
                           torch.randn(count, key_dim, dtype=dtype, generator=generator).to(device),
                           torch.full((count,), 90., dtype=dtype, device=device))
                    actual_inputs = tuple(x.clone().requires_grad_() for x in raw)
                    reference_inputs = tuple(x.double().clone().requires_grad_() for x in raw)
                    norm2 = actual_inputs[2].square() * actual_inputs[0].square().sum(-1) * actual_inputs[1].square().sum(-1)
                    actual = current_bucket_score(*actual_inputs, floor, norm2=norm2)
                    expected = explicit_score(*reference_inputs, floor)
                    torch.testing.assert_close(actual.double(), expected, rtol=3e-6, atol=2e-5)
                    got = torch.autograd.grad(actual.sum(), actual_inputs)
                    want = torch.autograd.grad(expected.sum(), reference_inputs)
                    for name, a, e in zip(('k', 'v', 'beta', 'u', 'q', 'temperature'), got, want):
                        self.assertTrue(torch.isfinite(a).all(), name)
                        self.assertLess((a.double() - e).norm().item(), 1e-6 * e.norm().item() + 1e-6, name)
                    # Above-floor beta gradients vanish analytically, including
                    # negative beta. Zero k/v must keep all derivatives finite.
                    above = norm2.detach() > floor * floor
                    self.assertEqual(torch.count_nonzero(got[2][above]).item(), 0)
                    exact_floor = 2.**-20
                    b = torch.tensor([exact_floor], dtype=dtype, device=device, requires_grad=True)
                    unit = torch.tensor([[1., 0.]], dtype=dtype, device=device)
                    query = torch.tensor([[.5, .5]], dtype=dtype, device=device)
                    score = current_bucket_score(unit, unit, b, query, query,
                                                 torch.ones_like(b), exact_floor)
                    self.assertEqual(torch.autograd.grad(score.sum(), b)[0].item(), 0.)

    def test_cpu_formula_all_gradients_zeros_and_floor(self):
        self.check_formula('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_formula_all_gradients_zeros_and_floor(self):
        self.check_formula('cuda')

    def check_shared_path(self, device):
        # Reuse only the established isolated import/CPU environment. The
        # actual current score, shared-local math and repair masks remain live.
        spec = importlib.util.spec_from_file_location(
            '_current_period_fixture', Path(__file__).with_name('test_period_repair.py'))
        fixture = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture)
        environment = fixture.TestPeriodReplacementCPU
        environment.setUpClass()
        fast = environment.fast
        try:
            for seed, beta_value in ((492, 3e-6), (913, 3e-6), (492, 1e-6)):
                with self.subTest(device=device, seed=seed, beta=beta_value):
                    self.compare_shared_case(fast, device, seed, beta_value)
        finally:
            environment.tearDownClass()

    def compare_shared_case(self, fast, device, seed, beta_value):
        generator = torch.Generator().manual_seed(seed)
        batch, length, heads, key_dim, value_dim = 64, 2, 1, 128, 64
        def randn(*shape):
            return torch.randn(shape, generator=generator).to(device)
        r = randn(batch, length, heads, key_dim)
        k = torch.zeros_like(r); k[:, 0, :, 1] = 1; k[:, 1, :, 0] = 1
        v = randn(batch, length, heads, value_dim) * .1
        v[:, 1] = v[:, 0]
        u = v.clone()
        q = torch.zeros_like(r); q[..., 0] = 1; q[..., 1] = 1
        g = torch.full((batch, length, heads), -.02, device=device)
        temperature = torch.full((heads,), math.log(math.sqrt(key_dim * value_dim)), device=device)
        raw_logit = torch.zeros(batch, length, heads, device=device)
        raw_logit[:, 1] = math.log(beta_value / (1 - beta_value))
        gate = F.silu(randn(batch, heads, value_dim))
        upstream = randn(batch, heads, value_dim)
        actual_logit = raw_logit.clone().requires_grad_()
        expected_logit = raw_logit.clone().requires_grad_()
        masks = []
        original_repair = fast._repair_periods
        def record_masks(ys, scores, flags, *args, **kwargs):
            masks.extend(int(flag.sum()) for flag in flags)
            return original_repair(ys, scores, flags, *args, **kwargs)
        reducer = cpu_reduce if device == 'cpu' else fast.tuple_routing_reduce
        with patch.object(fast, 'tuple_routing_reduce', reducer), patch.object(fast, '_repair_periods', record_masks):
            actual, flagged = fast.fast_matrix_gdn(
                r, k, v, g, actual_logit.sigmoid(), u, q, temperature,
                repair_periods=True, shared_local=True, fused_local=device == 'cuda')
            actual = actual[:, 1]
        self.assertFalse(flagged.any())
        self.assertEqual(sum(masks), 0)
        # Independent literal two-token recurrence with exactly the same
        # normalized input vectors. The orthogonal keys make the history's
        # derivative with respect to the current erase gate exactly zero.
        rr, kk = (fast._gdn_normalize(x).double() for x in (r, k))
        uu, qq = (F.normalize(x.float(), dim=-1, eps=1e-6).double() for x in (u, q))
        beta = expected_logit.sigmoid().double()
        first = beta[:, 0, :, None, None] * kk[:, 0, :, :, None] * v[:, 0].double()[..., None, :]
        old = g[:, 1].double().exp()[..., None, None] * (
            first - beta[:, 1, :, None, None] * kk[:, 1, :, :, None]
            * (kk[:, 1, :, :, None] * first).sum(-2)[..., None, :])
        current = beta[:, 1, :, None, None] * kk[:, 1, :, :, None] * v[:, 1].double()[..., None, :]
        states = torch.stack((current, old), -3)
        norm = states.square().sum((-2, -1)).clamp_min(1e-12).sqrt()
        scores = ((qq[:, 1, :, None, :, None] * states).sum(-2) * uu[:, 1, :, None, :]).sum(-1)
        scores = scores * temperature.double().exp()[None, :, None] / norm
        weights = (scores - scores.amax(-1, keepdim=True)).float().softmax(-1)
        values = (rr[:, 1, :, None, :, None] * states).sum(-2).float()
        expected = (weights[..., None] * values).sum(-2) * key_dim**-.5
        def loss(output):
            normalized = output * torch.rsqrt(output.square().mean(-1, keepdim=True) + 1e-6)
            return (normalized * gate * upstream).sum()
        got = torch.autograd.grad(loss(actual), actual_logit)[0][:, 1]
        want = torch.autograd.grad(loss(expected), expected_logit)[0][:, 1]
        self.assertTrue(torch.isfinite(got).all())
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-8)
        self.assertLess((got - want).norm().item(), 3e-5 * want.norm().item() + 1e-9)

    def test_cpu_correlated_shared_path_sigmoid_gradients_after_rmsnorm(self):
        self.check_shared_path('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_correlated_shared_path_sigmoid_gradients_after_rmsnorm(self):
        self.check_shared_path('cuda')


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main(verbosity=2)
