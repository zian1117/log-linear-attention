"""No-repair orchestration must not construct grouped precise caches.

Execute the actual _repair_periods function body on CPU, without importing
the unrelated CUDA/FLA fast kernels. Its numerical repair helpers are real
imports replaced by failing mocks: this specifically tests caller control flow.
"""
import ast
import importlib
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import torch
import torch.nn.functional as F
import triton


def load_caller():
    root = Path(__file__).resolve().parents[1] / 'hattention'
    package_name = '_grouped_cache_orchestration_tests'
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(root)]
        sys.modules[package_name] = package
    source = root / 'fast_matrix_gdn.py'
    parsed = ast.parse(source.read_text())
    selected = [node for node in parsed.body
                if (isinstance(node, ast.FunctionDef) and node.name == '_repair_periods')
                or (isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == '_REPAIR_PREFIX_QUANTUM'
                    for target in node.targets))]
    namespace = dict(__name__=package_name+'.caller', __package__=package_name,
                     torch=torch, F=F, triton=triton)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec'), namespace)
    precise = importlib.import_module(package_name+'.precise_period_reads')
    cache = importlib.import_module(package_name+'.cached_chunk_reads')
    return namespace['_repair_periods'], precise, cache


class TestGroupedCacheOrchestration(unittest.TestCase):
    def test_no_repair_or_only_nonfinite_heads_does_not_prepare_caches(self):
        repair, precise, cache = load_caller()
        batch, chunks, chunk, key_dim, value_dim = 2, 4, 8, 3, 5
        shape = batch, chunks, chunk
        inputs = (torch.zeros(*shape, key_dim), torch.zeros(*shape, key_dim),
                  torch.zeros(*shape, value_dim), torch.zeros(shape), torch.zeros(shape),
                  torch.zeros(*shape, value_dim), torch.zeros(*shape, key_dim),
                  torch.ones(batch, 1, 1))
        for local_only in (False, True):
            levels = chunk.bit_length() if local_only else 6
            for nonfinite_head in (False, True):
                for repair_chunks, compact in ((False, False), (True, False), (True, True)):
                    with self.subTest(local_only=local_only, nonfinite_head=nonfinite_head,
                                      repair_chunks=repair_chunks, compact=compact):
                        ys = [torch.randn(*shape, value_dim) for _ in range(levels)]
                        scores = [torch.randn(shape) for _ in range(levels)]
                        original_ys, original_scores = tuple(ys), tuple(scores)
                        masks = [torch.zeros(shape, dtype=torch.bool) for _ in range(levels)]
                        nonfinite = [torch.zeros_like(x) for x in masks]
                        if nonfinite_head:
                            masks[-1][1] = True
                            nonfinite[-1][1] = True
                        fail = AssertionError('A discarded/absent repair constructed a precise cache')
                        with mock.patch.object(precise, 'precise_period_reads', side_effect=fail), \
                             mock.patch.object(precise, 'prepare_precise_chunk_cache', side_effect=fail), \
                             mock.patch.object(cache, 'prepare_precise_read_input_cache', side_effect=fail), \
                             mock.patch.object(cache, 'prepare_grouped_chunk_reads', side_effect=fail):
                            flagged = repair(ys, scores, masks, nonfinite, inputs,
                                             1e-6, torch.float32,
                                             repair_chunks=repair_chunks, compact_cache_reads=compact)
                        self.assertTrue(torch.equal(flagged, torch.tensor([False, nonfinite_head])))
                        self.assertTrue(all(a is b for a, b in zip(ys, original_ys)))
                        self.assertTrue(all(a is b for a, b in zip(scores, original_scores)))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
