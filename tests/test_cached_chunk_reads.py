"""Compact cached reads: token oracle, shared histories, and all input grads."""

import importlib
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F


_spec = importlib.util.spec_from_file_location(
    '_cached_chunk_oracle', Path(__file__).with_name('test_precise_period_reads.py'))
_oracle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_oracle)


class TestCachedChunkReads(unittest.TestCase):
    def check_device(self, device):
        previous = _oracle.load_helper()
        module = importlib.import_module(previous.__module__)
        compact = importlib.import_module(previous.__module__.rsplit('.', 1)[0] + '.cached_chunk_reads').cached_chunk_reads
        generator = torch.Generator(device=device).manual_seed(1358)

        def randn(*shape):
            return torch.randn(*shape, generator=generator, device=device, dtype=torch.float64)

        batch, chunks, chunk, key_dim, value_dim = 3, 14, 8, 5, 7
        shape = (batch, chunks, chunk)
        for kind in ('normal', 'collinear', 'strong_decay'):
            key = F.normalize(randn(*shape, key_dim), dim=-1)
            if kind == 'collinear':
                key = key[:, :1, :1].expand_as(key).clone()
            raw = (F.normalize(randn(*shape, key_dim), dim=-1), key,
                   .1 * randn(*shape, value_dim),
                   torch.full(shape, .999 if kind == 'collinear' else .7,
                              device=device, dtype=torch.float64),
                   (-torch.ones(shape, device=device, dtype=torch.float64)
                    * (10. if kind == 'strong_decay' else .01)).cumsum(-1),
                   F.normalize(randn(*shape, value_dim), dim=-1),
                   F.normalize(randn(*shape, key_dim), dim=-1), randn(batch, 1, 1).exp())
            needed = torch.zeros((batch, chunks), device=device, dtype=torch.bool)
            positions = torch.arange(chunks, device=device)
            cases = []
            for level in (4, 5, 6):
                period_chunks = (1 << level) // chunk
                groups = (chunks + period_chunks - 1) // period_chunks
                periods = torch.arange(batch * groups, device=device).reshape(batch, groups) % 3 != 1
                periods[0, 0] = True
                wanted = positions.expand(batch, -1) % 3 != 0
                wanted = wanted.clone()
                wanted[0, period_chunks - 1] = True
                needed |= periods[:, positions // period_chunks]
                cases.append((level, periods, wanted))
            actual_inputs, old_inputs, oracle_inputs = (
                tuple(x.clone().requires_grad_() for x in raw) for _ in range(3))
            cache = module.prepare_precise_chunk_cache(*actual_inputs[1:5], needed)
            old_cache = module.prepare_precise_chunk_cache(*old_inputs[1:5], needed)
            losses = [0., 0., 0.]
            for level, periods, wanted in cases:
                r, k, _, beta, gc, u, q, temp = actual_inputs
                # The compact route must not gather any complete raw history.
                with patch.object(module, '_gather_tokens', side_effect=AssertionError('unexpected period raw gather')):
                    actual = compact(r, k, beta, gc, u, q, temp, level, periods, wanted, cache)
                old = previous(*old_inputs, level, periods, selected_chunks=wanted, chunk_cache=old_cache)
                full = _oracle.direct_reference(*oracle_inputs, level, torch.ones_like(periods), 1e-6)
                period_chunks = (1 << level) // chunk
                expected = tuple(x.reshape(batch, periods.shape[1] * period_chunks, chunk, *x.shape[2:])[:, :chunks]
                                 .flatten(0, 1).index_select(0, actual[0]) for x in full[1:])
                self.assertTrue(torch.equal(actual[0], old[0]))
                for target in (old[1:], expected):
                    torch.testing.assert_close(actual[1], target[0], atol=2e-12, rtol=2e-10)
                    torch.testing.assert_close(actual[2], target[1], atol=2e-8, rtol=2e-7)
                upstream = (randn(*actual[1].shape), randn(*actual[2].shape))
                for index, outputs in enumerate((actual[1:], old[1:], expected)):
                    losses[index] = losses[index] + sum((a * b).sum() for a, b in zip(outputs, upstream))
            got = torch.autograd.grad(losses[0], actual_inputs)
            for loss, inputs in ((losses[1], old_inputs), (losses[2], oracle_inputs)):
                want = torch.autograd.grad(loss, inputs)
                for label, a, b in zip(('r', 'k', 'v', 'beta', 'gc', 'u', 'q', 'temperature'), got, want):
                    self.assertTrue(torch.isfinite(a).all(), (kind, label))
                    torch.testing.assert_close(a, b, atol=3e-7, rtol=3e-6,
                                               msg=lambda m: f'{kind}, {label}: {m}')
            if kind == 'normal':
                for index in (1, 2, 3, 4):
                    self.assertGreater(got[index][0, 0].norm().item(), 0., index)

            # Empty requests need no history, including with an empty cache.
            empty_cache = module.prepare_precise_chunk_cache(*raw[1:5], torch.zeros_like(needed))
            r, k, _, beta, gc, u, q, temp = raw
            level, periods, wanted = cases[-1]
            for selected_periods, selected_chunks in ((torch.zeros_like(periods), wanted),
                                                       (periods, torch.zeros_like(wanted))):
                empty = compact(r, k, beta, gc, u, q, temp, level, selected_periods,
                                selected_chunks, empty_cache, output_dtype=torch.bfloat16)
                self.assertEqual(empty[0].numel(), 0)
                self.assertEqual(empty[1].shape, (0, chunk, value_dim))
                self.assertEqual(empty[1].dtype, torch.bfloat16)
                self.assertEqual(empty[2].shape, (0, chunk))
            with self.assertRaisesRegex(ValueError, 'history'):
                compact(r, k, beta, gc, u, q, temp, level, periods, wanted, empty_cache)

    def test_cpu_values_shared_cache_and_all_gradients(self):
        with torch._dynamo.config.patch(disable=True):
            self.check_device('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA')
    def test_cuda_values_shared_cache_and_all_gradients(self):
        self.check_device('cuda')


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
