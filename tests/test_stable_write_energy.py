"""Stable write energy against matrix recurrence and the old cancellation."""
import importlib.util
from pathlib import Path
import unittest

import torch
import torch.nn.functional as F

_spec = importlib.util.spec_from_file_location(
    '_stable_write_oracle', Path(__file__).with_name('test_shared_local_norm.py'))
_oracle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_oracle)


class TestStableWriteEnergy(unittest.TestCase):
    def check_small_first_write(self, device):
        shared, energy, gram = _oracle.modules()
        generator = torch.Generator().manual_seed(887)
        key = F.normalize(torch.randn(4, 8, 7, generator=generator), dim=-1).to(device)
        value = torch.randn(4, 8, 5, generator=generator).to(device) * .1
        beta = torch.tensor([1e-5, 2e-5, 3e-6, 1e-4], device=device)[:, None].expand(4, 8).clone()
        beta.requires_grad_()
        gc = torch.zeros_like(beta)
        terms = energy._right_terms(key, beta, gc)
        actual = shared.shared_local_norms(key, value, beta, gc, terms=terms)[1][0]
        shape = (4, 4, 2)
        inverse, decay = (gram._segment_blocks(x, 2) for x in terms[1:3])
        b = beta.reshape(shape)
        k2 = key.square().sum(-1).reshape(shape)
        val = value.reshape(*shape, 5)
        writes = torch.tensor([True, False], device=device)
        unshared, _ = energy._local_energy(inverse, decay, b, k2, val, writes)
        expected = (beta.double().square() * key.double().square().sum(-1)
                    * value.double().square().sum(-1))[..., ::2]
        for result in (actual[..., ::2], unshared[..., 0]):
            torch.testing.assert_close(result.double(), expected, rtol=5e-7, atol=0)
            got = torch.autograd.grad(result.sum(), beta, retain_graph=True)[0]
            want = torch.autograd.grad(expected.sum(), beta, retain_graph=True)[0]
            torch.testing.assert_close(got, want, rtol=5e-7, atol=0)
        # Explicit negative control: the former energy identity subtracts
        # O(beta) quantities to recover the O(beta²) first-write energy.
        vv = value.square().sum(-1)
        old = (2 * beta * vv - beta * (2 - beta * key.square().sum(-1)) * vv)[..., ::2]
        old_relative = ((old.double() - expected).norm() / expected.norm()).item()
        stable_relative = ((actual[..., ::2].double() - expected).norm() / expected.norm()).item()
        self.assertGreater(old_relative, 1e-4)
        self.assertGreater(old_relative, 100 * stable_relative)

    def check_joint_gradients(self, device):
        shared, energy, gram = _oracle.modules()
        for dtype in (torch.float32, torch.float64):
            for mode in ('small_beta', 'random', 'zero_beta'):
                with self.subTest(device=device, dtype=dtype, mode=mode):
                    generator = torch.Generator().manual_seed(718)
                    def randn(*shape):
                        return torch.randn(shape, dtype=dtype, generator=generator).to(device)
                    key = F.normalize(randn(2, 8, 7), dim=-1)
                    value = randn(2, 8, 5) * .1
                    beta = randn(2, 8).sigmoid()
                    if mode == 'small_beta':
                        beta = beta * 1e-4
                    elif mode == 'zero_beta':
                        beta[..., ::3] = 0
                    g = -.1 * randn(2, 8).sigmoid()
                    actual_inputs = tuple(x.requires_grad_() for x in (key, value, beta, g))
                    reference_inputs = tuple(x.detach().double().requires_grad_() for x in actual_inputs)
                    gc = g.cumsum(-1)
                    terms = energy._right_terms(key, beta, gc)
                    shared_norms = shared.shared_local_norms(key, value, beta, gc, terms=terms)
                    losses = [key.sum() * 0, key.sum() * 0, reference_inputs[0].sum() * 0]
                    multiplier = 1e8 if mode == 'small_beta' else 1.
                    for level in range(1, 4):
                        period = 1 << level
                        shape = (2, 8 // period, period)
                        inverse, decay = (gram._segment_blocks(x, period) for x in terms[1:3])
                        unshared, _ = energy._local_energy(
                            inverse, decay, beta.reshape(shape), key.square().sum(-1).reshape(shape),
                            value.reshape(*shape, 5), torch.arange(period, device=device) < period // 2)
                        direct = _oracle.token_matrices(*reference_inputs, level).square().sum((-2, -1))
                        upstream = randn(2, 8) * multiplier
                        for i, norm2 in enumerate((shared_norms[level][0], unshared.reshape_as(beta), direct)):
                            losses[i] = losses[i] + (norm2 * upstream).sum()
                        for norm2 in (shared_norms[level][0], unshared.reshape_as(beta)):
                            tolerance = 5e-5 if dtype == torch.float32 else 1e-10
                            error = (norm2.double() - direct).norm().item()
                            self.assertLess(error, tolerance * direct.norm().item() + 1e-16)
                    gradients = [torch.autograd.grad(loss, actual_inputs if i < 2 else reference_inputs,
                                                     retain_graph=i == 0)
                                 for i, loss in enumerate(losses)]
                    for got in gradients[:2]:
                        for name, a, e in zip(('k', 'v', 'beta', 'g'), got, gradients[2]):
                            tolerance = 5e-5 if dtype == torch.float32 else 1e-10
                            error = (a.double() - e).norm().item()
                            self.assertTrue(torch.isfinite(a).all(), name)
                            self.assertLess(error, tolerance * e.norm().item() + 1e-8, name)

    def test_cpu_small_first_write_and_old_formula_negative_control(self):
        with torch._dynamo.config.patch(disable=True):
            self.check_small_first_write('cpu')

    def test_cpu_joint_cross_level_gradients(self):
        with torch._dynamo.config.patch(disable=True):
            self.check_joint_gradients('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_small_first_write_and_old_formula_negative_control(self):
        with torch._dynamo.config.patch(disable=True):
            self.check_small_first_write('cuda')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_joint_cross_level_gradients(self):
        with torch._dynamo.config.patch(disable=True):
            self.check_joint_gradients('cuda')


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
