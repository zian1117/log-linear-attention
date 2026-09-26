"""Single-write history routing: literal erase recurrence and repaired gradients."""
import importlib.util
from pathlib import Path
import unittest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('_previous_score', ROOT/'hattention/current_bucket_score.py')
score_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(score_module)
spec = importlib.util.spec_from_file_location('_previous_current_fixture', Path(__file__).with_name('test_current_bucket_score.py'))
fixture = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture)


class TestPreviousBucketScore(unittest.TestCase):
    def check_formula(self, device):
        generator = torch.Generator().manual_seed(94)
        for dtype in (torch.float32, torch.float64):
            for mode in ('random', 'collinear', 'zero') + (('near_erase',) if dtype == torch.float64 else ()):
                with self.subTest(device=device, dtype=dtype, mode=mode):
                    def randn(*shape):
                        return torch.randn(shape, generator=generator, dtype=dtype).to(device)
                    batch, key_dim, value_dim = 12, 9, 5
                    k0, k1 = (F.normalize(randn(batch, key_dim), dim=-1) for _ in range(2))
                    value = randn(batch, value_dim)*.1
                    beta0 = torch.tensor([0., .5, -.5, .01, .003, 3e-6, -3e-6, 1e-6, .9, .2, .8, .1], dtype=dtype, device=device)
                    beta1 = torch.full_like(beta0, .4)
                    if mode in ('collinear', 'near_erase'):
                        k1 = k0.clone()
                    if mode == 'near_erase':
                        beta1.fill_(.999)
                    if mode == 'zero':
                        k0[:3] = 0; value[3:5] = 0
                    raw = (k0, k1, value, beta0, beta1, torch.full_like(beta0, -.02),
                           randn(batch, value_dim), randn(batch, key_dim), torch.full_like(beta0, 10.))
                    actual = tuple(x.clone().requires_grad_() for x in raw)
                    oracle = tuple(x.double().clone().requires_grad_() for x in raw)
                    ka, kb, v, ba, bb, g, u, q, temperature = actual
                    key = torch.stack((ka, kb), -2)
                    values = torch.stack((v, torch.zeros_like(v)), -2)
                    beta = torch.stack((ba, bb), -1)
                    decay = torch.stack((torch.ones_like(g), torch.zeros_like(g), g.exp(), torch.ones_like(g)), -1).reshape(batch, 2, 2)
                    effective = ka - bb[:, None]*(ka*kb).sum(-1, keepdim=True)*kb
                    norm2 = ((ba*g.exp()).square()*effective.square().sum(-1)*v.square().sum(-1))[:, None]
                    got = score_module.previous_bucket_score(key, values, beta, decay,
                        torch.stack((torch.zeros_like(u), u), -2),
                        torch.stack((torch.zeros_like(q), q), -2), temperature[:, None], 1e-6, norm2).squeeze(-1)
                    ka, kb, v, ba, bb, g, u, q, temperature = oracle
                    initial = ba[:, None, None]*ka[:, :, None]*v[:, None, :]
                    state = g.exp()[:, None, None]*(initial-bb[:, None, None]*kb[:, :, None]
                        *(kb[:, :, None]*initial).sum(-2)[:, None, :])
                    expected = temperature*(state*q[:, :, None]*u[:, None, :]).sum((-2,-1))/state.square().sum((-2,-1)).clamp_min(1e-12).sqrt()
                    torch.testing.assert_close(got.double(), expected, rtol=5e-6, atol=1e-6)
                    gradients = torch.autograd.grad(got.sum(), actual)
                    references = torch.autograd.grad(expected.sum(), oracle)
                    tolerance = 5e-6 if dtype == torch.float32 else 2e-9
                    for index, (a, b) in enumerate(zip(gradients, references)):
                        with self.subTest(gradient=index):
                            self.assertTrue(torch.isfinite(a).all())
                            self.assertLess((a.double()-b).norm().item(), tolerance*b.norm().item()+1e-7)

    def check_public_path(self, device):
        spec = importlib.util.spec_from_file_location('_previous_period_fixture', Path(__file__).with_name('test_period_repair.py'))
        period = importlib.util.module_from_spec(spec); spec.loader.exec_module(period)
        environment = period.TestPeriodReplacementCPU
        environment.setUpClass()
        try:
            for shared in (False, True):
                for seed, beta, near_erase in ((492,.003,False),(913,.003,False),(492,.01,False),(492,.5,True)):
                    with self.subTest(device=device, shared=shared, seed=seed, beta=beta, near_erase=near_erase):
                        fixture.TestCurrentBucketScore.compare_shared_case(self, environment.fast, device, seed, beta,
                            small_index=0, shared_local=shared, near_erase=near_erase)
        finally:
            environment.tearDownClass()

    def test_cpu_formula_all_nine_gradients(self):
        self.check_formula('cpu')

    def test_cpu_public_paths_and_near_erase_repair(self):
        self.check_public_path('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_formula_all_nine_gradients(self):
        self.check_formula('cuda')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_public_paths_and_near_erase_repair(self):
        self.check_public_path('cuda')


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main(verbosity=2)
