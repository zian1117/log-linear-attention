"""Sparse precise inputs must merge cancelling gradients before FP32 casts."""
import importlib
import importlib.util
from pathlib import Path
import unittest

import torch
import torch.nn.functional as F

_spec = importlib.util.spec_from_file_location(
    '_shared_precise_input_oracle', Path(__file__).with_name('test_precise_period_reads.py'))
_oracle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_oracle)


def normalize(raw):
    r, k, v, g, beta, u, q, log_temperature = raw
    r, k = (x * torch.rsqrt(x.square().sum(-1, keepdim=True) + 1e-6) for x in (r, k))
    u, q = (F.normalize(x, dim=-1, eps=1e-6) for x in (u, q))
    return r, k, v, beta, g.double().cumsum(-1), u, q, log_temperature.exp()


def fixture(dtype, kind):
    generator = torch.Generator().manual_seed(65283)
    batch, chunks, chunk, key_dim, value_dim = 3, 14, 8, 5, 7
    shape = batch, chunks, chunk
    def normal(*size):
        return torch.randn(size, dtype=dtype, generator=generator)
    k = normal(*shape, key_dim)
    beta = torch.rand(shape, dtype=dtype, generator=generator) * .8
    g = -torch.rand(shape, dtype=dtype, generator=generator) * .03
    if kind == 'collinear':
        k[:] = k[:, :1, :1]
        beta.fill_(.999)
    elif kind == 'large_decay_prefix':
        g.fill_(-1e-5)
        g[:, 0, 0] = -1e6
        g[:, 5, 2] = -1000
    elif kind == 'zero_beta':
        beta[:, ::2] = 0
        beta[:, 1::3] = 1
    raw = (normal(*shape, key_dim), k, .1 * normal(*shape, value_dim), g, beta,
           normal(*shape, value_dim), normal(*shape, key_dim),
           torch.full((batch, 1, 1), 1., dtype=dtype))
    position = torch.arange(chunks)
    history = torch.zeros(batch, chunks, dtype=torch.bool)
    reads = torch.zeros_like(history)
    cases = []
    for level in (4, 5, 6):
        period_chunks = (1 << level) // chunk
        groups = (chunks + period_chunks - 1) // period_chunks
        selected = torch.arange(batch * groups).reshape(batch, groups) % 3 != 1
        selected[0, 7 // period_chunks] = True
        wanted = (position[None, :].expand(batch, -1) % 3 != 0).clone()
        wanted[0, 7] = True
        selected_members = selected[:, position // period_chunks]
        history |= selected_members
        reads |= wanted & selected_members & ((position % period_chunks)[None, :] >= period_chunks // 2)
        cases.append((level, selected, wanted))
    return raw, history, reads, cases, generator


class TestSharedPreciseInputCache(unittest.TestCase):
    def check_case(self, dtype, kind, device='cpu'):
        period_reads = _oracle.load_helper()
        package = period_reads.__module__.rsplit('.', 1)[0]
        preparation = importlib.import_module(package + '.precise_period_reads')
        compact = importlib.import_module(package + '.cached_chunk_reads')
        raw, history, reads, cases, generator = fixture(dtype, kind)
        raw = tuple(x.to(device) for x in raw)
        history, reads = history.to(device), reads.to(device)
        cases = [(level, selected.to(device), wanted.to(device))
                 for level, selected, wanted in cases]
        # Compare the compact shared cache, compact standalone helper, old
        # selected-period API with factors, and independent token recurrence.
        inputs = [tuple(x.clone().requires_grad_() for x in raw) for _ in range(4)]
        data = [normalize(x) for x in inputs]
        factors = [preparation.prepare_precise_chunk_cache(*p[1:5], history) for p in data[:3]]
        p = data[0]
        read_cache = compact.prepare_precise_read_input_cache(
            p[0], p[1], p[3], p[4], p[5], p[6], reads, chunk_cache=factors[0])
        self.assertEqual(read_cache.r.shape[0], reads.sum().item())
        for cache in factors:
            self.assertTrue((cache.raw_k[0] == 0).all())
            self.assertTrue((cache.raw_beta[0] == 0).all())
            self.assertTrue((cache.raw_gc[0] == 0).all())
            self.assertEqual(cache.end[0].item(), 1.)
        losses = [0.] * 4
        upstreams = []
        for level, selected, wanted in cases:
            actual = []
            for i in (0, 1):
                r, k, _, beta, gc, u, q, temperature = data[i]
                kwargs = {'read_input_cache': read_cache} if i == 0 else {}
                actual.append(compact.cached_chunk_reads(
                    r, k, beta, gc, u, q, temperature, level, selected, wanted,
                    factors[i], **kwargs))
            actual.append(period_reads(*data[2], level, selected, selected_chunks=wanted,
                                       chunk_cache=factors[2]))
            expected_inputs = tuple(x.double() for x in data[3])
            full = _oracle.direct_reference(*expected_inputs, level, torch.ones_like(selected), 1e-6)
            batch, chunks, chunk = data[3][3].shape
            period_chunks = (1 << level) // chunk
            expected = tuple(x.reshape(batch, selected.shape[1] * period_chunks, chunk, *x.shape[2:])
                             [:, :chunks].flatten(0, 1).index_select(0, actual[0][0]) for x in full[1:])
            for index, alternative in enumerate(actual[1:], start=1):
                self.assertTrue(torch.equal(actual[0][0], alternative[0]))
                # Sharing raw gathers changes no arithmetic in the compact
                # helper. The period helper uses two guarded refinement
                # passes instead of one, so its near-zero scores can differ
                # by ~1e-9. Assess both against the independent oracle below,
                # while retaining the stricter matrix-read comparison.
                if index == 1 or device == 'cpu':
                    for a, b in zip(actual[0][1:], alternative[1:]):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
                else:
                    torch.testing.assert_close(actual[0][1], alternative[1],
                                               rtol=2e-10, atol=2e-12)
            for result in actual:
                for a, e in zip(result[1:], expected):
                    torch.testing.assert_close(a.double(), e, rtol=2e-6, atol=2e-8)
            upstream = tuple(torch.randn(x.shape, dtype=x.dtype, generator=generator).to(device)
                             for x in actual[0][1:])
            upstreams.append(upstream)
            for i, values in enumerate([x[1:] for x in actual] + [expected]):
                losses[i] += sum((x * grad).sum() for x, grad in zip(values, upstream))
        gradients = [torch.autograd.grad(loss, x) for loss, x in zip(losses, inputs)]
        for actual in gradients[:3]:
            for name, a, e in zip(('r', 'k', 'v', 'g', 'beta', 'u', 'q', 'log_temperature'), actual, gradients[3]):
                self.assertTrue(torch.isfinite(a).all(), name)
                # Raw FP32 key normalization itself subtracts large nearly
                # equal terms; assess its full gradient, while the beta
                # regression below checks individual components strictly.
                self.assertLess((a-e).norm().item(), 3e-7 + 1e-6 * e.norm().item(), name)
                if name == 'beta':
                    torch.testing.assert_close(a, e, rtol=3e-6, atol=3e-7)
        if kind == 'normal':
            self.assertFalse(reads[0, 0])
            self.assertGreater(gradients[0][2][0, 0].norm().item(), 0.)
        if dtype == torch.float32 and kind == 'collinear':
            # Deliberately reconstruct the former bug: factor/read beta
            # promote independently, so their cancelling gradients are cast
            # separately to FP32 before addition at the original tensor.
            bad_inputs = tuple(x.clone().requires_grad_() for x in raw)
            p = normalize(bad_inputs)
            factor = preparation.prepare_precise_chunk_cache(*p[1:5], history)
            broken = compact.prepare_precise_read_input_cache(
                p[0], p[1], p[3], p[4], p[5], p[6], reads, chunk_cache=factor)
            ids = reads.flatten().nonzero(as_tuple=False).flatten()
            broken = broken._replace(beta=p[3].flatten(0, 1).index_select(0, ids).double())
            bad_loss = 0.
            for (level, selected, wanted), upstream in zip(cases, upstreams):
                r, k, _, beta, gc, u, q, temperature = p
                out = compact.cached_chunk_reads(r, k, beta, gc, u, q, temperature,
                    level, selected, wanted, factor, read_input_cache=broken)
                bad_loss += sum((x * grad).sum() for x, grad in zip(out[1:], upstream))
            bad_beta = torch.autograd.grad(bad_loss, bad_inputs[4])[0]
            self.assertGreater((bad_beta-gradients[3][4]).abs().max().item(), .001)
            self.assertLess((gradients[0][4]-gradients[3][4]).abs().max().item(), 1e-6)
        p = data[0]
        empty = compact.prepare_precise_read_input_cache(
            p[0], p[1], p[3], p[4], p[5], p[6], torch.zeros_like(reads), chunk_cache=factors[0])
        r, k, _, beta, gc, u, q, temperature = p
        level, selected, wanted = cases[-1]
        with self.assertRaisesRegex(ValueError, 'missing a requested chunk'):
            compact.cached_chunk_reads(r, k, beta, gc, u, q, temperature, level,
                                        selected, wanted, factors[0], read_input_cache=empty)
        result = compact.cached_chunk_reads(r, k, beta, gc, u, q, temperature, level,
            selected, torch.zeros_like(wanted), factors[0], read_input_cache=empty)
        self.assertEqual(result[0].numel(), 0)

    def test_shared_inputs_preserve_precise_gradients(self):
        with torch._dynamo.config.patch(disable=True):
            for dtype in (torch.float32, torch.float64):
                for kind in ('normal', 'collinear', 'large_decay_prefix', 'zero_beta'):
                    with self.subTest(dtype=dtype, kind=kind):
                        self.check_case(dtype, kind)

    @unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA')
    def test_cuda_shared_inputs_preserve_precise_gradients(self):
        for dtype in (torch.float32, torch.float64):
            for kind in ('normal', 'collinear', 'large_decay_prefix', 'zero_beta'):
                with self.subTest(dtype=dtype, kind=kind):
                    self.check_case(dtype, kind, 'cuda')


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main(verbosity=2)
