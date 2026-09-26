"""Independent positive contribution and dense-state oracles for repair metadata."""
import importlib
from pathlib import Path
import sys
import types
import unittest

import torch
import torch.nn.functional as F


def modules():
    package = '_radial_metadata_tests'
    if package not in sys.modules:
        root = types.ModuleType(package)
        root.__path__ = [str(Path(__file__).resolve().parents[1] / 'hattention')]
        sys.modules[package] = root
    return (importlib.import_module(package + '.radial_metadata'),
            importlib.import_module(package + '.radial_diagnostics'))


def explicit_contributions(decay, largest, remainder, period):
    """Enumerate two writes represented by every chunk; no summary recurrence."""
    maxima, rests = [], []
    for t in range(decay.shape[-1]):
        start = t // period * period
        end = min(t, start + period // 2)
        contributions = []
        for source in range(start, end):
            gain = decay[..., source + 1:t].double().prod(-1)
            contributions.extend((largest[..., source].double() * gain,
                                  remainder[..., source].double() * gain))
        if contributions:
            values = torch.stack(contributions, -1)
            maximum, selected = values.max(-1)
            other = torch.arange(values.shape[-1], device=decay.device) != selected.unsqueeze(-1)
            rest = (values * other).sum(-1)
        else:
            maximum = torch.zeros_like(decay[..., 0], dtype=torch.float64)
            rest = torch.zeros_like(maximum)
        maxima.append(maximum)
        rests.append(rest)
    return torch.stack(maxima, -1), torch.stack(rests, -1)


def dense_write_history(k, v, beta, gc, period):
    """Separate prefix/local matrices; no triangular chunk-factor identities."""
    k, v, beta, gc = (x.double() for x in (k, v, beta, gc))
    bh, chunks, tokens, key_dim = k.shape
    state = k.new_zeros((bh, key_dim, v.shape[-1]))
    projections, deltas, energies, increments, decisions = [], [], [], [], []
    for chunk in range(chunks):
        if chunk % period == 0:
            state = torch.zeros_like(state)
        prefix, local = state, torch.zeros_like(state)
        pp, dd, nn, ii, ff = [], [], [], [], []
        writing = chunk % period < period // 2
        for token in range(tokens):
            log_decay = gc[:, chunk, token]
            if token:
                log_decay = log_decay - gc[:, chunk, token - 1]
            gain = log_decay.exp()[:, None, None]
            prefix_pre, local_pre = gain * prefix, gain * local
            old = prefix_pre + local_pre
            key, value, b = k[:, chunk, token], v[:, chunk, token], beta[:, chunk, token]
            prefix_projection = torch.einsum('bk,bkv->bv', key, prefix_pre)
            local_projection = torch.einsum('bk,bkv->bv', key, local_pre)
            pp.append(prefix_projection / gc[:, chunk, token].exp().unsqueeze(-1))
            dd.append(-local_projection)
            residual = value - prefix_projection - local_projection
            tangent = key.unsqueeze(-1) * residual.unsqueeze(-2)
            updated = old + b[:, None, None] * tangent
            energy = old.square().sum((-2, -1))
            nn.append(energy if writing else torch.zeros_like(energy))
            ii.append(updated.square().sum((-2, -1)) - energy)
            product = tangent.square().sum((-2, -1)) * energy
            inner2 = (tangent * old).sum((-2, -1)).square()
            scale = torch.maximum(product, inner2)
            ff.append((scale > 0) & (product - inner2 <= torch.finfo(torch.float32).eps**.5 * scale)
                      & writing)
            prefix = prefix_pre - b[:, None, None] * key.unsqueeze(-1) * prefix_projection.unsqueeze(-2)
            local_write = value if writing else torch.zeros_like(value)
            local = local_pre + b[:, None, None] * key.unsqueeze(-1) * (local_write - local_projection).unsqueeze(-2)
        state = prefix + local
        for target, values in ((projections, pp), (deltas, dd), (energies, nn),
                               (increments, ii), (decisions, ff)):
            target.append(torch.stack(values, 1))
    return tuple(torch.stack(x, 1) for x in (projections, deltas, energies, increments, decisions))


class TestRadialMetadata(unittest.TestCase):
    def check_positive_summaries(self, device):
        metadata, _ = modules()
        generator = torch.Generator().manual_seed(974)
        for length, period in ((1, 2), (3, 2), (5, 3), (7, 16),
                               (16, 4), (129, 64), (257, 128)):
            for mode in ('random', 'ties', 'zero', 'underflow', 'positive'):
                with self.subTest(device=device, length=length, period=period, mode=mode):
                    shape = (2, 3, length)
                    largest = torch.rand(shape, generator=generator)
                    remainder = largest * torch.rand(shape, generator=generator)
                    decay = torch.rand(shape, generator=generator)
                    if mode == 'ties':
                        largest.fill_(1)
                        remainder.zero_()
                        decay.fill_(1)
                    elif mode == 'zero':
                        largest.zero_()
                        remainder.zero_()
                        decay.zero_()
                    elif mode == 'underflow':
                        largest.mul_(1e-35)
                        remainder.mul_(1e-35)
                        decay.mul_(1e-15)
                    elif mode == 'positive':
                        largest.add_(.1)
                        remainder.add_(.01)
                        decay = decay * .8 + .1
                    decay, largest, remainder = (x.to(device) for x in (decay, largest, remainder))
                    expected = explicit_contributions(decay, largest, remainder, period)
                    actual = metadata.boundary_pair(decay, largest, remainder, period)
                    for got, want in zip(actual, expected):
                        torch.testing.assert_close(got.double(), want, rtol=3e-6, atol=1e-43)
        contributions = torch.tensor([[1., 1e-20, 1e-30]], device=device)
        largest, remainder = metadata.positive_pair(contributions)
        self.assertGreater(remainder.item(), 0)
        self.assertEqual((contributions.sum(-1) - largest).item(), 0)

    def check_many_sources_per_chunk(self, device):
        metadata, _ = modules()
        generator = torch.Generator().manual_seed(416)
        for chunks, period, writes in ((7, 4, 8), (9, 8, 64)):
            for ties in (False, True):
                with self.subTest(device=device, chunks=chunks, writes=writes, ties=ties):
                    original = .5 + torch.rand(2, chunks, writes, generator=generator)
                    if ties:
                        original.fill_(1)
                    original = original.to(device)
                    decay = (.2 + .8 * torch.rand(2, chunks, generator=generator)).to(device)
                    largest, remainder = metadata.positive_pair(original)
                    self.assertTrue((remainder > largest).all())
                    actual = metadata.boundary_pair(decay, largest, remainder, period)
                    expected_max, expected_rest = [], []
                    for token in range(chunks):
                        start = token // period * period
                        end = min(token, start + period // 2)
                        contributions = [original[:, source].double()
                                         * decay[:, source + 1:token].double().prod(-1).unsqueeze(-1)
                                         for source in range(start, end)]
                        if contributions:
                            # Enumerate ALL original writes. Treating the sum of
                            # the other writes as one contribution is invalid.
                            values = torch.cat(contributions, -1).sort(-1).values
                            expected_max.append(values[:, -1])
                            expected_rest.append(values[:, :-1].sum(-1))
                        else:
                            expected_max.append(torch.zeros(2, device=device, dtype=torch.float64))
                            expected_rest.append(torch.zeros_like(expected_max[-1]))
                    for got, want in zip(actual, (torch.stack(expected_max, -1),
                                                   torch.stack(expected_rest, -1))):
                        torch.testing.assert_close(got.double(), want, rtol=3e-6, atol=1e-12)

    def test_cpu_many_original_writes_per_chunk(self):
        self.check_many_sources_per_chunk('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_many_original_writes_per_chunk(self):
        self.check_many_sources_per_chunk('cuda')

    def test_cpu_optional_prefix_metadata_preserves_norms_and_gradients(self):
        metadata, _ = modules()
        package = metadata.__package__
        shared = importlib.import_module(package + '.shared_local_norm')
        energy = importlib.import_module(package + '.energy_bucket_norm')
        generator = torch.Generator().manual_seed(452)
        shape = (2, 3, 8)
        raw = (F.normalize(torch.randn(*shape, 3, dtype=torch.float64, generator=generator), dim=-1),
               torch.randn(*shape, 5, dtype=torch.float64, generator=generator) * .1,
               torch.rand(shape, dtype=torch.float64, generator=generator) * .7,
               -torch.rand(shape, dtype=torch.float64, generator=generator) * .1)
        weights = torch.randn(4, *shape, dtype=torch.float64, generator=generator)
        for levels in (1, 3, 4):
            with self.subTest(levels=levels):
                records = []
                for requested in (False, True):
                    k, v, beta, g = inputs = tuple(x.clone().requires_grad_() for x in raw)
                    gc = g.cumsum(-1)
                    terms = energy._right_terms(k, beta, gc)
                    with torch._dynamo.config.patch(disable=True):
                        result = shared.shared_local_norms(
                            k, v, beta, gc, terms=terms, max_levels=levels,
                            return_metadata=requested)
                    if requested:
                        result, prefix, flags = result
                        self.assertFalse(prefix.requires_grad)
                        self.assertTrue(all(not flag.requires_grad for flag in flags))
                        self.assertEqual(len(flags), levels - 1)
                        # Included levels partition previous writes only inside
                        # blocks of this size; truncated metadata is not a full
                        # chunk prefix.
                        period = 1 << (levels - 1)
                        row = torch.arange(shape[-1])
                        mask = ((row[:, None] > row[None, :])
                                & (row[:, None] // period == row[None, :] // period))
                        expected_prefix = ((terms[1] * terms[2]) * mask) @ v
                        torch.testing.assert_close(prefix, expected_prefix, rtol=1e-11, atol=1e-12)
                    loss = sum((norm * weights[level]).sum()
                               for level, (norm, _) in enumerate(result))
                    gradients = torch.autograd.grad(loss, inputs, allow_unused=True)
                    records.append((result, gradients))
                for pair_a, pair_b in zip(records[0][0], records[1][0]):
                    for a, b in zip(pair_a, pair_b):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
                for a, b in zip(records[0][1], records[1][1]):
                    # Level zero writes have no dependence on decay.
                    if a is None or b is None:
                        self.assertIs(a, b)
                    else:
                        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def check_dense_write_metadata(self, device):
        _, diagnostics = modules()
        generator = torch.Generator().manual_seed(841)
        for chunks, tokens, period in ((3, 5, 2), (7, 3, 4), (9, 5, 8), (5, 7, 3)):
            for mode in ('random', 'aligned', 'zero'):
                with self.subTest(device=device, chunks=chunks, tokens=tokens, period=period, mode=mode):
                    shape = (2, chunks, tokens)
                    k = torch.randn(*shape, 3, generator=generator)
                    k = F.normalize(k, dim=-1)
                    v = torch.randn(*shape, 5, generator=generator) * .1
                    beta = torch.rand(shape, generator=generator) * .7
                    gc = (-torch.rand(shape, generator=generator).double() * .1).cumsum(-1)
                    if mode == 'aligned':
                        k.zero_()
                        k[..., 0] = 1
                        v.zero_()
                        v[..., 0] = .1
                    elif mode == 'zero':
                        beta.zero_()
                    k, v, beta, gc = (x.to(device) for x in (k, v, beta, gc))
                    prefix, delta, expected_pre, increments, expected_flags = dense_write_history(k, v, beta, gc, period)
                    actual_pre = diagnostics.write_pre_norms(increments.float(), gc, period)
                    torch.testing.assert_close(actual_pre.double(), expected_pre, rtol=3e-5, atol=2e-8)
                    with torch._dynamo.config.patch(disable=True):
                        reads, flags = diagnostics.coarse_write_history(
                            prefix.float(), delta.float(), k, v, beta, gc, period)
                    torch.testing.assert_close(flags, expected_flags, rtol=0, atol=0)
                    expected_reads = torch.zeros((shape[0], chunks), dtype=torch.bool, device=device)
                    for chunk in range(chunks):
                        start = chunk // period * period
                        if chunk % period >= period // 2:
                            expected_reads[:, chunk] = expected_flags[:, start:chunk + 1].any(dim=(1, 2))
                    torch.testing.assert_close(reads, expected_reads, rtol=0, atol=0)

    def test_cpu_dense_write_metadata(self):
        self.check_dense_write_metadata('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_dense_write_metadata(self):
        self.check_dense_write_metadata('cuda')

    def test_cpu_positive_summaries(self):
        self.check_positive_summaries('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_positive_summaries(self):
        self.check_positive_summaries('cuda')


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
