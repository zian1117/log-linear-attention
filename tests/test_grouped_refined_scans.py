"""Grouped refined scans against token-independent chunk recurrence and joins."""
import importlib
from pathlib import Path
import sys
import types
import unittest
import torch
import torch.nn.functional as F

PACKAGE = '_grouped_refined_tests'
if PACKAGE not in sys.modules:
    package = types.ModuleType(PACKAGE)
    package.__path__ = [str(Path(__file__).resolve().parents[1] / 'hattention')]
    sys.modules[PACKAGE] = package
module = importlib.import_module(PACKAGE + '.grouped_refined_scans')
grouped_refined_states = module.grouped_refined_states


def dense(factors, period):
    k, w, value, decay = factors
    current = k.new_zeros((k.shape[0], k.shape[-1], value.shape[-1]))
    states = []
    for position in range(k.shape[1]):
        if position % period == 0:
            current = torch.zeros_like(current)
        states.append(current)
        write = value[:, position] if position % period < period // 2 else 0.
        current = (decay[:, position, None, None] * current
                   + k[:, position].transpose(-1, -2) @ (write - w[:, position] @ current))
    return torch.stack(states, 1)


class TestGroupedRefinedScans(unittest.TestCase):
    def scan_cases(self, device):
        torch.manual_seed(512)
        for refinements in (1, 2):
            # Uneven/partial histories, empty batch, and odd feature dimensions.
            periods = (2, 4, 16, 4, 4)
            factors = []
            for batch, chunks in ((2, 3), (1, 8), (2, 11), (0, 4), (2, 4)):
                k = F.normalize(torch.randn(batch, chunks, 3, 5, device=device, dtype=torch.double), dim=-1)
                a = torch.full((batch, chunks), .95, device=device, dtype=torch.double)
                w = .05 * a[..., None, None] * k
                v = .1 * torch.randn(batch, chunks, 3, 7, device=device, dtype=torch.double)
                factors.append(tuple(x.detach().requires_grad_() for x in (k, w, v, a)))
            truth = tuple(tuple(x.detach().clone().requires_grad_() for x in f) for f in factors)
            got = grouped_refined_states(factors, periods, refinements=refinements)
            want = tuple(dense(f, p) for f, p in zip(truth, periods))
            used = (0, 1, 2)  # Both empty and nonempty outputs are unused.
            upstream = tuple(torch.randn_like(got[i]) for i in used)
            ga = torch.autograd.grad(tuple(got[i] for i in used), tuple(x for f in factors for x in f), upstream, allow_unused=True)
            ge = torch.autograd.grad(tuple(want[i] for i in used), tuple(x for f in truth for x in f), upstream, allow_unused=True)
            for a, e in zip(got, want):
                torch.testing.assert_close(a, e, rtol=2e-10, atol=2e-12)
            for a, e, x in zip(ga, ge, (x for f in factors for x in f)):
                a = torch.zeros_like(x) if a is None else a
                e = torch.zeros_like(x) if e is None else e
                torch.testing.assert_close(a, e, rtol=2e-9, atol=2e-11)

    def shared_case(self, device):
        # Shared roots and overlapping histories exercise gradient accumulation
        # across grouped outputs, including duplicate periods and checkpointing.
        from torch.utils.checkpoint import checkpoint
        torch.manual_seed(617)
        raw = (torch.randn(2, 9, 3, 5, device=device, dtype=torch.double) * .2,
               torch.randn(2, 9, 3, 7, device=device, dtype=torch.double) * .1,
               torch.full((2, 9), .95, device=device, dtype=torch.double))
        histories, periods = (9, 7, 9), (4, 8, 4)
        def evaluate(inputs, implementation):
            k, v, a = inputs
            factors = tuple((k[:, :n], .05 * a[:, :n, None, None] * k[:, :n], v[:, :n], a[:, :n]) for n in histories)
            return implementation(factors, periods)
        leaves = tuple(tuple(x.clone().requires_grad_() for x in raw) for _ in range(3))
        outputs = (evaluate(leaves[0], grouped_refined_states),
                   checkpoint(lambda *x: evaluate(x, grouped_refined_states), *leaves[1], use_reentrant=False),
                   evaluate(leaves[2], lambda fs, ps: tuple(dense(f, p) for f, p in zip(fs, ps))))
        weights = tuple(torch.randn_like(y) for y in outputs[0])
        gradients = tuple(torch.autograd.grad(ys, xs, weights) for ys, xs in zip(outputs, leaves))
        for a, c, e in zip(*outputs):
            torch.testing.assert_close(a, c, rtol=0, atol=0)
            torch.testing.assert_close(a, e, rtol=2e-10, atol=2e-12)
        for a, c, e in zip(*gradients):
            torch.testing.assert_close(a, c, rtol=0, atol=0)
            torch.testing.assert_close(a, e, rtol=2e-9, atol=2e-11)

    def consumer_case(self, device):
        cached = importlib.import_module(PACKAGE + '.cached_chunk_reads')
        precise = importlib.import_module(PACKAGE + '.precise_period_reads')
        torch.manual_seed(958)
        for chunks in (7, 8):
            shape = (2, chunks, 4)
            def rand(d): return torch.randn(*shape, d, device=device, dtype=torch.double)
            raw = (F.normalize(rand(3), dim=-1), F.normalize(rand(3), dim=-1), .1 * rand(5),
                   torch.full(shape, .7, device=device, dtype=torch.double),
                   torch.full(shape, -.02, device=device, dtype=torch.double).cumsum(-1),
                   F.normalize(rand(5), dim=-1), F.normalize(rand(3), dim=-1),
                   torch.full((2, 1, 1), 3., device=device, dtype=torch.double))
            leaves = tuple(tuple(x.clone().requires_grad_() for x in raw) for _ in range(2))
            outputs = []
            for grouped, inputs in enumerate(leaves):
                r, k, v, beta, gc, u, q, temp = inputs
                required = torch.ones((2, chunks), device=device, dtype=torch.bool)
                cache = precise.prepare_precise_chunk_cache(k, v, beta, gc, required)
                read_cache = cached.prepare_precise_read_input_cache(r, k, beta, gc, u, q, required, chunk_cache=cache)
                selections = []
                for level in (3, 4):
                    period = (1 << level) // 4
                    selected = required.clone(); selected[0, ::3] = False
                    selected_periods = torch.ones((2, (chunks + period - 1) // period), device=device, dtype=torch.bool)
                    selections.append((level, selected_periods, selected, period))
                prepared = cached.prepare_grouped_chunk_reads(k, selections, cache, read_cache)
                states = grouped_refined_states(tuple(p.tensors[:4] for p in prepared.values()),
                    tuple(p.plan.period_chunks for p in prepared.values())) if grouped else (None, None)
                result = []
                for state, (level, periods, selected, history) in zip(states, selections):
                    _, y, score = cached.cached_chunk_reads(r, k, beta, gc, u, q, temp, level, periods,
                        selected, cache, read_input_cache=read_cache, history_chunks=history,
                        prepared=prepared[level], boundary_state=state)
                    result.extend((y, score))
                outputs.append(tuple(result))
            weights = tuple(torch.randn_like(y) for y in outputs[0])
            gradients = tuple(torch.autograd.grad(ys, xs, weights) for ys, xs in zip(outputs, leaves))
            for a, e in zip(*outputs): torch.testing.assert_close(a, e, rtol=3e-10, atol=3e-11)
            for a, e in zip(*gradients): torch.testing.assert_close(a, e, rtol=3e-9, atol=3e-10)

    def test_cpu_contract_and_shared_gradients(self):
        self.scan_cases('cpu')
        self.shared_case('cpu')
        with torch._dynamo.config.patch(disable=True): self.consumer_case('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_grouped_recurrence_all_gradients_and_cached_consumer(self):
        self.scan_cases('cuda')
        self.shared_case('cuda')
        self.consumer_case('cuda')


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main(verbosity=2)
