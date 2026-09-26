"""Independent positive-contribution checks for detached Fenwick mass bounds."""
import importlib.util
import math
import os
from pathlib import Path
import unittest

import torch
import torch.nn.functional as F


spec = importlib.util.spec_from_file_location(
    'tested_decay_mass', Path(__file__).resolve().parents[1] / 'hattention' / 'decay_mass.py')
mass_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mass_module)


class TestDecayMass(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(631)
        self.k = F.normalize(torch.randn(2, 3, 8, 3, generator=generator), dim=-1).requires_grad_()
        self.v = torch.randn(2, 3, 8, 5, generator=generator, requires_grad=True)
        self.beta = torch.rand(2, 3, 8, generator=generator, requires_grad=True)
        self.g = (-.1 - torch.rand(2, 3, 8, generator=generator)).requires_grad_()
        self.gc = self.g.cumsum(-1)

    def test_local_masses_against_explicit_positive_contributions(self):
        writes = self.beta.double() * self.k.double().norm(dim=-1) * self.v.double().norm(dim=-1)
        positions = torch.arange(self.k.shape[-2])
        difference = self.gc[..., :, None] - self.gc[..., None, :]
        E = difference.masked_fill(positions[:, None] < positions[None, :], -torch.inf).exp()
        for level in range(4):
            with self.subTest(level=level):
                actual = mass_module.local_mass(self.k, self.v, self.beta, self.gc, level)
                reused = mass_module.local_mass(self.k, self.v, self.beta, self.gc, level, E=E)
                expected = torch.zeros_like(actual, dtype=torch.float64)
                period = 1 << level
                for t in range(self.k.shape[-2]):
                    start = t // period * period
                    stop = t + 1 if level == 0 else min(t + 1, start + period // 2)
                    for j in range(start, stop):
                        expected[..., t] += writes[..., j] * self.g[..., j+1:t+1].double().sum(-1).exp()
                torch.testing.assert_close(actual.double(), expected, rtol=2e-6, atol=2e-7)
                torch.testing.assert_close(reused, actual, rtol=2e-6, atol=2e-7)
                self.assertFalse(actual.requires_grad)

    def test_large_initial_decay_does_not_round_later_contributions(self):
        k = torch.ones(1, 1, 8, 1)
        v, beta = torch.ones_like(k), torch.ones(1, 1, 8)
        g = torch.full_like(beta, -.1)
        g[..., 0] = -1e6
        cumulative = g.double().cumsum(-1)
        actual = mass_module.local_mass(k, v, beta, cumulative, 3)
        expected = torch.zeros_like(beta, dtype=torch.float64)
        for target in range(8):
            for source in range(min(target + 1, 4)):
                expected[..., target] += g[..., source+1:target+1].double().sum(-1).exp()
        torch.testing.assert_close(actual.double(), expected, rtol=2e-7, atol=1e-7)
        decay, addition = mass_module.chunk_mass_summaries(k, v, beta, cumulative)
        expected_addition = sum(g[..., source+1:].double().sum(-1).exp() for source in range(8))
        torch.testing.assert_close(addition.double(), expected_addition, rtol=2e-7, atol=1e-7)
        self.assertEqual(decay.item(), 0.)

    def test_boundary_and_high_masses_against_full_token_contributions(self):
        decay, addition = mass_module.chunk_mass_summaries(self.k, self.v, self.beta, self.gc)
        writes = (self.beta.double() * self.k.double().norm(dim=-1) * self.v.double().norm(dim=-1)).flatten(1, 2)
        log_decay = self.g.double().flatten(1, 2)
        chunk_size, chunks = self.k.shape[-2], self.k.shape[1]
        for period in (2, 3, 4, 8):
            with self.subTest(period=period):
                actual = mass_module.boundary_mass(decay, addition, period)
                expected = torch.zeros_like(actual, dtype=torch.float64)
                for chunk in range(chunks):
                    start = chunk // period * period
                    stop = min(chunk, start + period // 2)
                    for source in range(start * chunk_size, stop * chunk_size):
                        weight = log_decay[:, source+1:chunk*chunk_size].sum(-1).exp()
                        expected[:, chunk] += writes[:, source] * weight
                torch.testing.assert_close(actual.double(), expected, rtol=2e-6, atol=2e-7)
                actual_high = mass_module.high_mass(actual, self.gc)
                expected_high = expected[..., None] * self.g.double().cumsum(-1).exp()
                torch.testing.assert_close(actual_high.double(), expected_high, rtol=3e-6, atol=2e-7)
                self.assertFalse(actual.requires_grad)
                self.assertFalse(actual_high.requires_grad)

    def test_upper_bound_on_explicit_gdn_bucket_matrices(self):
        k, v = self.k.double().flatten(1, 2), self.v.double().flatten(1, 2)
        beta, g = self.beta.double().flatten(1, 2), self.g.double().flatten(1, 2)
        levels = (k.shape[1] - 1).bit_length() + 1
        masses = [mass_module.local_mass(self.k, self.v, self.beta, self.gc, level).flatten(1, 2)
                  for level in range(4)]
        decay, addition = mass_module.chunk_mass_summaries(self.k, self.v, self.beta, self.gc)
        for level in range(4, levels):
            boundary = mass_module.boundary_mass(decay, addition, 1 << (level - 3))
            masses.append(mass_module.high_mass(boundary, self.gc).flatten(1, 2))
        contributions = []
        for t in range(k.shape[1]):
            transition = g[:, t, None, None].exp() * (
                torch.eye(k.shape[-1]) - beta[:, t, None, None] * k[:, t, :, None] * k[:, t, None, :])
            contributions = [transition @ old for old in contributions]
            contributions.append(beta[:, t, None, None] * k[:, t, :, None] * v[:, t, None, :])
            buckets = {}
            for j, value in enumerate(contributions):
                level = (t ^ j).bit_length()
                buckets[level] = value if level not in buckets else buckets[level] + value
            for level, matrix in buckets.items():
                bound = masses[level][:, t].double()
                self.assertTrue((matrix.norm(dim=(-2, -1)) <= bound * (1 + 3e-6) + 1e-10).all())

    def test_zero_write_mass(self):
        beta = torch.zeros_like(self.beta, requires_grad=True)
        decay, addition = mass_module.chunk_mass_summaries(self.k, self.v, beta, self.gc)
        self.assertTrue((addition == 0).all())
        self.assertFalse(addition.requires_grad)
        for level in range(4):
            self.assertTrue((mass_module.local_mass(self.k, self.v, beta, self.gc, level) == 0).all())
        self.assertTrue((mass_module.boundary_mass(decay, addition, 4) == 0).all())

    @unittest.skipUnless(torch.cuda.is_available() or os.environ.get('TRITON_INTERPRET') == '1',
                         'requires CUDA or the Triton interpreter')
    def test_fused_scalar_scan_against_explicit_sum(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        generator = torch.Generator(device=device).manual_seed(827)
        for chunks, period in ((1, 2), (3, 8), (7, 3), (9, 4), (17, 8)):
            with self.subTest(chunks=chunks, period=period):
                decay = torch.rand(6, chunks, generator=generator, device=device)
                addition = torch.rand(6, chunks, generator=generator, device=device)
                actual = torch.empty_like(decay)
                mass_module._boundary_mass_fwd[(6, math.ceil(chunks / period))](
                    decay, addition, actual, chunks, period, num_warps=1)
                expected = torch.zeros_like(decay, dtype=torch.float64)
                for chunk in range(chunks):
                    start = chunk // period * period
                    for source in range(start, min(chunk, start + period // 2)):
                        expected[:, chunk] += addition[:, source].double() * decay[:, source+1:chunk].double().prod(-1)
                torch.testing.assert_close(actual.double(), expected, rtol=2e-6, atol=2e-7)


if __name__ == '__main__':
    unittest.main(verbosity=2)
