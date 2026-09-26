"""Normalization/layout fusion preserves floors, cast joins, and gradients."""
import importlib
from pathlib import Path
import sys
import types
import unittest

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


PACKAGE = '_normalized_chunks_tests'
if PACKAGE not in sys.modules:
    package = types.ModuleType(PACKAGE)
    package.__path__ = [str(Path(__file__).resolve().parents[1] / 'hattention')]
    sys.modules[PACKAGE] = package
normalized_chunks = importlib.import_module(PACKAGE + '.normalized_chunks').normalized_chunks


def reference(raw, chunk, kind, eps):
    value = raw.float()
    if kind == 'gdn':
        value = value * torch.rsqrt(value.square().sum(-1, keepdim=True) + eps)
    else:
        value = F.normalize(value, dim=-1, eps=eps)
    # Independent permutation order: move heads before padding the time axis.
    value = value.permute(0, 2, 1, 3)
    batch, heads, length, dimension = value.shape
    padding = (-length) % chunk
    return F.pad(value, (0, 0, 0, padding)).reshape(
        batch * heads, (length + padding) // chunk, chunk, dimension).contiguous()


class TestNormalizedChunks(unittest.TestCase):
    def compare(self, raw, kind, *, eps=1e-6, chunk=8, joined=False, replay=False):
        actual = raw.detach().requires_grad_()
        expected = raw.detach().clone(memory_format=torch.preserve_format).requires_grad_()
        backend = 'inductor' if raw.is_cuda else 'eager'
        call = lambda x: normalized_chunks(x, chunk, kind, eps, backend=backend)
        output = checkpoint(call, actual, use_reentrant=False) if replay else call(actual)
        target = reference(expected, chunk, kind, eps)
        self.assertEqual(output.dtype, torch.float32)
        self.assertTrue(output.is_contiguous())
        torch.testing.assert_close(output, target, rtol=2e-6, atol=2e-7)
        weights = torch.randn_like(output)
        # Padding must produce exact zero and must contribute no raw gradient,
        # even though padded positions receive nonzero upstream weights.
        if raw.shape[1] % chunk:
            self.assertEqual(torch.count_nonzero(output[:, -1, raw.shape[1] % chunk:]).item(), 0)

        def loss(value):
            if not joined:
                return (value * weights).sum()
            promoted = value.double()  # One shared promotion, as in precise repairs.
            large = 4096 * weights.double()
            return ((promoted * large).sum()
                    + (promoted * (-large + .125 * weights.double())).sum()
                    + (.25 * value * weights).sum())

        got = torch.autograd.grad(loss(output), actual)[0]
        want = torch.autograd.grad(loss(target), expected)[0]
        self.assertEqual(got.dtype, raw.dtype)
        self.assertTrue(torch.isfinite(got).all())
        relative = .012 if raw.dtype == torch.bfloat16 else 8e-6
        self.assertLessEqual((got.double() - want.double()).norm().item(),
                             relative * want.double().norm().item() + 1e-7)

    def radial(self, device):
        generator = torch.Generator().manual_seed(921)
        value = (.05 * torch.randn(128, generator=generator)).bfloat16().to(device)
        upstream = value.float() / .05 + .003 * torch.randn(128, generator=generator).to(device)
        for kind in ('gdn', 'l2'):
            raw = value.clone().requires_grad_()
            result = normalized_chunks(raw.reshape(1, 1, 1, 128), 1, kind,
                                       backend='inductor' if device == 'cuda' else 'eager')
            got = torch.autograd.grad((result * upstream.reshape_as(result)).sum(), raw)[0]
            exact = value.double().requires_grad_()
            target = (exact * torch.rsqrt(exact.square().sum() + 1e-6)
                      if kind == 'gdn' else exact / exact.norm())
            want = torch.autograd.grad((target * upstream.double()).sum(), exact)[0]
            # Independent FP64 derivative catches premature BF16 branch casts.
            self.assertGreater(want.norm().item(), .01)
            self.assertLess((got.double() - want).norm().item(), .008 * want.norm().item())

    def floors(self, device):
        for kind in ('gdn', 'l2'):
            for eps in (1e-6, .125):
                boundary = eps ** .5 if kind == 'gdn' else eps
                raw = torch.zeros(2, 8, 3, 8, device=device)
                factors = raw.new_tensor((0., .5, 1. - 2 ** -20, 1., 1. + 2 ** -20, 2., .1, 4.))
                raw[..., 0] = (boundary * factors)[None, :, None]
                self.compare(raw, kind, eps=eps)

    def test_cpu_strides_padding_and_input_dtype_contract(self):
        torch.manual_seed(346)
        for dtype, length in ((torch.float32, 7), (torch.float64, 9)):
            raw = torch.randn(2, length, 3, 14, dtype=dtype)[..., ::2]
            for kind in ('gdn', 'l2'):
                self.compare(raw, kind)

    def test_cpu_zero_floor_and_additive_epsilon(self):
        self.floors('cpu')

    def test_cpu_bf16_radial_and_shared_consumer_gradients(self):
        self.radial('cpu')
        raw = torch.randn(2, 3, 9, 8, dtype=torch.bfloat16).transpose(1, 2)
        for kind in ('gdn', 'l2'):
            self.compare(raw, kind, joined=True)

    def test_nonreentrant_checkpoint_shared_consumers(self):
        for device in ('cpu', 'cuda') if torch.cuda.is_available() else ('cpu',):
            raw = torch.randn(2, 9, 3, 8, device=device, dtype=torch.bfloat16)
            for kind in ('gdn', 'l2'):
                self.compare(raw, kind, joined=True, replay=True)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA Inductor required')
    def test_cuda_compiled_layout_floors_and_precision(self):
        self.radial('cuda')
        self.floors('cuda')
        for dtype in (torch.float32, torch.bfloat16):
            raw = torch.randn(2, 9, 3, 14, device='cuda', dtype=dtype)[..., ::2]
            for kind in ('gdn', 'l2'):
                self.compare(raw, kind, joined=True)


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main(verbosity=2)
