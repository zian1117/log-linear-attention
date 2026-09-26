"""Selected-period reads versus an independent token-matrix recurrence."""
import importlib
from pathlib import Path
import sys
import types
import unittest

import torch
import torch.nn.functional as F


def load_helper():
    # Import the numerical code without loading unrelated model dependencies.
    name = '_precise_period_tests'
    if name not in sys.modules:
        root = Path(__file__).resolve().parents[1] / 'hattention'
        package = types.ModuleType(name)
        package.__path__ = [str(root)]
        sys.modules[name] = package
        preparation = types.ModuleType(name + '.fast_matrix_gdn')
        preparation.torch = torch
        source = (root / 'fast_matrix_gdn.py').read_text()
        start = source.index('@torch.compile\ndef _prepare')
        end = source.index('\ndef _score', start)
        exec(compile(source[start:end], str(root / 'fast_matrix_gdn.py'), 'exec'), preparation.__dict__)
        sys.modules[preparation.__name__] = preparation
    return importlib.import_module(name + '.precise_period_reads').precise_period_reads


def direct_reference(r, k, v, beta, gc, u, q, temperature, level, selected, floor):
    batch, chunks, chunk, key_dim = k.shape
    total, period = chunks * chunk, 1 << level
    groups = (total + period - 1) // period
    r, k, v, beta, gc, u, q = (x.flatten(1, 2) for x in (r, k, v, beta, gc, u, q))
    state = k.new_zeros((batch, key_dim, v.shape[-1]))
    outputs, scores = [], []
    for token in range(total):
        if token % period == 0:
            state = torch.zeros_like(state)
        previous = gc[:, token - 1] if token % chunk else torch.zeros_like(gc[:, token])
        state = state * (gc[:, token] - previous).exp()[:, None, None]
        residual = -(k[:, token, None, :] @ state).squeeze(-2)
        if level == 0 or token % period < period // 2:
            residual = residual + v[:, token]
        state = state + k[:, token, :, None] * (beta[:, token, None] * residual)[:, None, :]
        y = (r[:, token, None, :] @ state).squeeze(-2)
        read = (q[:, token, None, :] @ state).squeeze(-2)
        score = temperature[:, 0, 0] * (read * u[:, token]).sum(-1) / state.square().sum((-2, -1)).clamp_min(floor ** 2).sqrt()
        active = level == 0 or token % period >= period // 2
        outputs.append(y if active else y * 0)
        scores.append(score if active else score * 0)
    y = F.pad(torch.stack(outputs, 1), (0, 0, 0, groups * period - total))
    score = F.pad(torch.stack(scores, 1), (0, groups * period - total))
    ids = selected.flatten().nonzero(as_tuple=False).flatten()
    return (ids, y.reshape(batch * groups, period, -1).index_select(0, ids),
            score.reshape(batch * groups, period).index_select(0, ids))


def full_level_reference(r, k, v, beta, gc, u, q, temperature, level, selected, floor):
    module = importlib.import_module('_precise_period_tests.bilinear_matrix_gdn')
    prepared = module._prepare(r, k, v, beta, gc, q)
    chunk = k.shape[-2]
    if (1 << level) <= chunk:
        y, score = module._local(prepared[6], prepared[7], k, v, beta, gc, u,
                                temperature, level, prepared[8], floor, v.dtype)
    else:
        y, score = module._coarse(*prepared[:6], k, beta, gc, u, temperature,
                                 (1 << level) // chunk, prepared[8], floor, v.dtype)
    total, period = k.shape[1] * chunk, 1 << level
    groups = (total + period - 1) // period
    y = F.pad(y.flatten(1, 2), (0, 0, 0, groups * period - total))
    score = F.pad(score.flatten(1, 2), (0, groups * period - total))
    ids = selected.flatten().nonzero(as_tuple=False).flatten()
    return (ids, y.reshape(-1, period, v.shape[-1]).index_select(0, ids),
            score.reshape(-1, period).index_select(0, ids))


class TestPrecisePeriodReads(unittest.TestCase):
    def check_chunk_cache(self, device):
        helper = load_helper()
        prepare_cache = sys.modules[helper.__module__].prepare_precise_chunk_cache
        generator = torch.Generator(device=device).manual_seed(948)

        def randn(*shape):
            return torch.randn(*shape, generator=generator, device=device, dtype=torch.float64)

        batch, chunks, chunk, key_dim, value_dim = 2, 7, 4, 5, 7
        shape = (batch, chunks, chunk)
        raw = (F.normalize(randn(*shape, key_dim), dim=-1),
               F.normalize(randn(*shape, key_dim), dim=-1), .1 * randn(*shape, value_dim),
               randn(*shape).sigmoid(), (-.01 * randn(*shape).abs()).cumsum(-1),
               F.normalize(randn(*shape, value_dim), dim=-1),
               F.normalize(randn(*shape, key_dim), dim=-1), randn(batch, 1, 1).exp())
        positions = torch.arange(chunks, device=device)
        needed = torch.zeros((batch, chunks), device=device, dtype=torch.bool)
        cases = []
        for level in (3, 4):
            period_chunks = (1 << level) // chunk
            groups = (chunks + period_chunks - 1) // period_chunks
            periods = torch.ones((batch, groups), device=device, dtype=torch.bool)
            periods[1, 0] = False
            requested = positions.expand(batch, -1) % period_chunks >= period_chunks // 2
            needed |= periods[:, positions // period_chunks]
            cases.append((level, periods, requested))
        actual_inputs = tuple(x.clone().requires_grad_() for x in raw)
        reference_inputs = tuple(x.clone().requires_grad_() for x in raw)
        uncached_inputs = tuple(x.clone().requires_grad_() for x in raw)
        cache = prepare_cache(*actual_inputs[1:5], needed)
        self.assertEqual(cache.kn.shape[0], needed.sum().item() + 1)
        self.assertLess(needed.sum().item(), batch * chunks)
        for index, factor in enumerate(cache[1:]):
            self.assertTrue(torch.equal(factor[0], torch.full_like(factor[0], 1. if index == 3 else 0.)))
        totals = [0., 0., 0.]
        for level, periods, requested in cases:
            actual = helper(*actual_inputs, level, periods, selected_chunks=requested, chunk_cache=cache)
            uncached = helper(*uncached_inputs, level, periods, selected_chunks=requested)
            full = direct_reference(*reference_inputs, level, torch.ones_like(periods), 1e-6)
            period_chunks = (1 << level) // chunk
            expected = tuple(x.reshape(batch, periods.shape[1] * period_chunks, chunk, *x.shape[2:])[:, :chunks]
                             .flatten(0, 1).index_select(0, actual[0]) for x in full[1:])
            upstream = (randn(*actual[1].shape), randn(*actual[2].shape))
            for target in (uncached[1:], expected):
                torch.testing.assert_close(actual[1], target[0], atol=2e-12, rtol=2e-10)
                torch.testing.assert_close(actual[2], target[1], atol=2e-8, rtol=2e-7)
            for index, outputs in enumerate((actual[1:], expected, uncached[1:])):
                totals[index] = totals[index] + sum((a * b).sum() for a, b in zip(outputs, upstream))
        actual_grad = torch.autograd.grad(totals[0], actual_inputs)
        for loss, inputs in ((totals[1], reference_inputs), (totals[2], uncached_inputs)):
            expected_grad = torch.autograd.grad(loss, inputs)
            for label, a, b in zip(('r', 'k', 'v', 'beta', 'gc', 'u', 'q', 'temperature'), actual_grad, expected_grad):
                torch.testing.assert_close(a, b, atol=3e-7, rtol=3e-6, msg=lambda m: f'cache reuse, {label}: {m}')
        missing = needed.clone()
        missing[0, 0] = False
        bad_cache = prepare_cache(*raw[1:5], missing)
        level, periods, requested = cases[0]
        with self.assertRaisesRegex(ValueError, 'history'):
            helper(*raw, level, periods, selected_chunks=requested, chunk_cache=bad_cache)
        empty = prepare_cache(*raw[1:5], torch.zeros_like(needed))
        self.assertEqual(empty.kn.shape[0], 1)
        self.assertTrue((empty.lookup == -1).all().item())

    def test_cpu_sparse_cache_reuse_and_gradients(self):
        with torch._dynamo.config.patch(disable=True):
            self.check_chunk_cache('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA')
    def test_cuda_sparse_cache_reuse_and_gradients(self):
        self.check_chunk_cache('cuda')

    def check_chunk_selection(self, device):
        helper = load_helper()
        generator = torch.Generator(device=device).manual_seed(946)

        def randn(*shape):
            return torch.randn(*shape, generator=generator, device=device, dtype=torch.float64)

        batch, chunks, chunk, key_dim, value_dim = 3, 14, 8, 5, 7
        for collinear in (False, True):
            key = F.normalize(randn(batch, chunks, chunk, key_dim), dim=-1)
            if collinear:
                key = key[:, :1, :1].expand_as(key).clone()
            raw = (F.normalize(randn(*key.shape), dim=-1), key,
                   .1 * randn(batch, chunks, chunk, value_dim),
                   torch.full((batch, chunks, chunk), .999 if collinear else .7,
                              device=device, dtype=torch.float64),
                   (-.01 * torch.ones(batch, chunks, chunk, device=device, dtype=torch.float64)).cumsum(-1),
                   F.normalize(randn(batch, chunks, chunk, value_dim), dim=-1),
                   F.normalize(randn(*key.shape), dim=-1),
                   torch.full((batch, 1, 1), 8., device=device, dtype=torch.float64))
            for level in (4, 5, 6):
                period_chunks = (1 << level) // chunk
                groups = (chunks + period_chunks - 1) // period_chunks
                periods = torch.arange(batch * groups, device=device).reshape(batch, groups) % 3 != 1
                periods[0, 0] = True
                requested = torch.arange(batch * chunks, device=device).reshape(batch, chunks) % 3 != 0
                requested[0, period_chunks - 1] = True
                positions = torch.arange(chunks, device=device)
                wanted = (requested & periods[:, positions // period_chunks]
                          & (positions % period_chunks >= period_chunks // 2))
                ids = wanted.flatten().nonzero(as_tuple=False).flatten()
                actual_inputs = tuple(x.clone().requires_grad_() for x in raw)
                reference_inputs = tuple(x.clone().requires_grad_() for x in raw)
                full_inputs = tuple(x.clone().requires_grad_() for x in raw)
                actual = helper(*actual_inputs, level, periods, selected_chunks=requested)
                all_periods = torch.ones_like(periods)
                expected_full = direct_reference(*reference_inputs, level, all_periods, 1e-6)
                old_full = helper(*full_inputs, level, all_periods)

                def select(output):
                    return tuple(x.reshape(batch, groups * period_chunks, chunk, *x.shape[2:])[:, :chunks]
                                 .flatten(0, 1).index_select(0, ids) for x in output[1:])

                expected, baseline = select(expected_full), select(old_full)
                self.assertTrue(torch.equal(actual[0], ids))
                self.assertEqual(actual[1].shape, (ids.numel(), chunk, value_dim))
                for target in (expected, baseline):
                    torch.testing.assert_close(actual[1], target[0], atol=2e-12, rtol=2e-10)
                    torch.testing.assert_close(actual[2], target[1], atol=2e-8, rtol=2e-7)
                upstream = (randn(*actual[1].shape), randn(*actual[2].shape))
                got = torch.autograd.grad(sum((a * b).sum() for a, b in zip(actual[1:], upstream)), actual_inputs)
                for target, inputs in ((expected, reference_inputs), (baseline, full_inputs)):
                    want = torch.autograd.grad(sum((a * b).sum() for a, b in zip(target, upstream)), inputs)
                    for label, a, b in zip(('r', 'k', 'v', 'beta', 'gc', 'u', 'q', 'temperature'), got, want):
                        torch.testing.assert_close(a, b, atol=3e-7, rtol=3e-6,
                                                   msg=lambda m: f'chunk selection, {level}, {label}: {m}')
                if not collinear:
                    # No outputs were requested from this write chunk, but
                    # its history must still receive gradients from later reads.
                    self.assertFalse(wanted[0, 0].item())
                    for index in (1, 2, 3, 4):
                        self.assertGreater(got[index][0, 0].norm().item(), 0.)
                for mask, request in ((torch.zeros_like(periods), requested),
                                      (periods, torch.zeros_like(requested))):
                    empty = helper(*raw, level, mask, selected_chunks=request)
                    self.assertEqual(empty[0].numel(), 0)
                    self.assertEqual(empty[1].shape, (0, chunk, value_dim))
                    self.assertEqual(empty[2].shape, (0, chunk))

    def test_cpu_selected_chunks_and_prefix_gradients(self):
        with torch._dynamo.config.patch(disable=True):
            self.check_chunk_selection('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA')
    def test_cuda_selected_chunks_and_prefix_gradients(self):
        self.check_chunk_selection('cuda')

    def check_levels(self, device, collinear=False):
        helper = load_helper()
        generator = torch.Generator(device=device).manual_seed(944)
        def randn(*shape):
            return torch.randn(*shape, generator=generator, device=device, dtype=torch.float64)
        batch, chunks, chunk, key_dim, value_dim = 3, 3, 8, 5, 7
        key = F.normalize(randn(batch, chunks, chunk, key_dim), dim=-1)
        if collinear:
            key = key[:, :1, :1].expand_as(key).clone()
        raw = (F.normalize(randn(*key.shape), dim=-1), key, .1 * randn(batch, chunks, chunk, value_dim),
               torch.full((batch, chunks, chunk), .999 if collinear else .7, dtype=torch.float64, device=device),
               (-.01 * torch.ones(batch, chunks, chunk, dtype=torch.float64, device=device)).cumsum(-1),
               F.normalize(randn(batch, chunks, chunk, value_dim), dim=-1),
               F.normalize(randn(*key.shape), dim=-1), torch.full((batch, 1, 1), 8., dtype=torch.float64, device=device))
        for level in range(7):
            period = 1 << level
            groups = (chunks * chunk + period - 1) // period
            selected = torch.arange(batch * groups, device=device).reshape(batch, groups) % 3 != 1
            # Exercise selection from several heads and a partial final period.
            selected[-1, -1] = True
            candidate_inputs = tuple(x.clone().requires_grad_() for x in raw)
            reference_inputs = tuple(x.clone().requires_grad_() for x in raw)
            baseline_inputs = tuple(x.clone().requires_grad_() for x in raw)
            actual = helper(*candidate_inputs, level, selected)
            expected = direct_reference(*reference_inputs, level, selected, 1e-6)
            baseline = full_level_reference(*baseline_inputs, level, selected, 1e-6)
            self.assertTrue(torch.equal(actual[0], expected[0]))
            torch.testing.assert_close(actual[1], expected[1], atol=2e-12, rtol=2e-10)
            torch.testing.assert_close(actual[2], expected[2], atol=2e-8, rtol=2e-7)
            torch.testing.assert_close(actual[1], baseline[1], atol=2e-12, rtol=2e-10)
            torch.testing.assert_close(actual[2], baseline[2], atol=2e-8, rtol=2e-7)
            upstream_y, upstream_score = randn(*actual[1].shape), randn(*actual[2].shape)
            loss_actual = (actual[1] * upstream_y).sum() + (actual[2] * upstream_score).sum()
            loss_expected = (expected[1] * upstream_y).sum() + (expected[2] * upstream_score).sum()
            got = torch.autograd.grad(loss_actual, candidate_inputs)
            want = torch.autograd.grad(loss_expected, reference_inputs)
            baseline_loss = (baseline[1] * upstream_y).sum() + (baseline[2] * upstream_score).sum()
            baseline_gradients = torch.autograd.grad(baseline_loss, baseline_inputs)
            for target_name, target in (('token recurrence', want), ('full level', baseline_gradients)):
                if target_name == 'full level' and collinear and level == 1:
                    # The old Gram expansion loses accuracy in this case's
                    # beta derivatives. The stable rank-one implementation is
                    # checked against the independent token recurrence above.
                    continue
                for label, a, b in zip(('r', 'k', 'v', 'beta', 'gc', 'u', 'q', 'temperature'), got, target):
                    self.assertTrue(torch.isfinite(a).all(), (level, label))
                    torch.testing.assert_close(a, b, atol=3e-7, rtol=3e-6, msg=lambda m: f'{target_name}, level={level}, {label}: {m}')
                    self.assertLessEqual((a - b).norm().item(), 3e-6 * b.norm().item() + 3e-7, (level, label))
        empty = torch.zeros((batch, chunks * chunk), device=device, dtype=torch.bool)
        ids, y, score = helper(*raw, 0, empty)
        self.assertEqual(tuple(y.shape), (0, 1, value_dim))
        self.assertEqual(tuple(score.shape), (0, 1))
        self.assertEqual(ids.numel(), 0)

    def test_cpu_values_and_all_input_gradients(self):
        # Compilation is exercised separately on CUDA; eager CPU keeps this
        # mathematical oracle test independent of CPU compiler availability.
        with torch._dynamo.config.patch(disable=True):
            self.check_levels('cpu')
            self.check_levels('cpu', collinear=True)

    @unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA')
    def test_cuda_values_and_all_input_gradients(self):
        self.check_levels('cuda')
        self.check_levels('cuda', collinear=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
