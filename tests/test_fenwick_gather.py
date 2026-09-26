"""Compare deterministic Fenwick copies with independent indexed references."""
import importlib.util
from pathlib import Path
import unittest

import torch

spec = importlib.util.spec_from_file_location('tested_fenwick_gather', Path(__file__).resolve().parents[1]/'hattention/fenwick_gather.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class TestFenwickGather(unittest.TestCase):
    def check_device(self, device):
        for dtype in ((torch.float64,) if device == 'cpu' else (torch.float32, torch.float64, torch.bfloat16)):
            for chunks, period in ((0, 2), (1, 2), (7, 2), (7, 3), (9, 4), (3, 8), (16, 8), (17, 16), (257, 256)):
                for batch, tail in ((1, ()), (2, (3,)), (3, (3, 5))):
                    with self.subTest(device=device, dtype=dtype, chunks=chunks, period=period, batch=batch, tail=tail):
                        raw = torch.randn(batch, chunks, *tail, device=device, dtype=dtype)
                        if len(tail) == 2:
                            raw = raw.transpose(-1, -2)  # Exercise strided inputs.
                        x, reference = raw.detach().requires_grad_(), raw.detach().clone().requires_grad_()
                        index = torch.tensor([j for j in range(chunks) if j % period >= period//2], device=device, dtype=torch.long)
                        actual, expected = helper.active_select(x, period), reference.index_select(1, index)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        self.assertTrue(actual.is_contiguous())
                        upstream = (torch.randn(*actual.shape[:-2], actual.shape[-1], actual.shape[-2], device=device, dtype=dtype).transpose(-1, -2)
                                    if len(tail) == 2 else torch.randn_like(actual))
                        ga, = torch.autograd.grad((actual*upstream).sum(), x)
                        ge, = torch.autograd.grad((expected*upstream).sum(), reference)
                        torch.testing.assert_close(ga, ge, rtol=0, atol=0)
                        compact = actual.detach().requires_grad_()
                        compact_ref = expected.detach().clone().requires_grad_()
                        expanded = helper.active_scatter(compact, chunks, period)
                        expanded_ref = torch.zeros_like(reference).index_copy(1, index, compact_ref)
                        torch.testing.assert_close(expanded, expanded_ref, rtol=0, atol=0)
                        upstream = (torch.randn(*expanded.shape[:-2], expanded.shape[-1], expanded.shape[-2], device=device, dtype=dtype).transpose(-1, -2)
                                    if len(tail) == 2 else torch.randn_like(expanded))
                        ga, = torch.autograd.grad((expanded*upstream).sum(), compact)
                        ge, = torch.autograd.grad((expanded_ref*upstream).sum(), compact_ref)
                        torch.testing.assert_close(ga, ge, rtol=0, atol=0)

    def test_cpu(self):
        self.check_device('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA')
    def test_cuda(self):
        self.check_device('cuda')


if __name__ == '__main__':
    unittest.main(verbosity=2)
