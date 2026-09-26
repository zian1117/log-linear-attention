"""Compact boundary outputs against full scans and independent token recurrence."""
import importlib.util
from pathlib import Path
import unittest

import torch

_spec = importlib.util.spec_from_file_location(
    '_compact_projected_oracle', Path(__file__).with_name('test_multi_projected_states.py'))
_oracle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_oracle)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class TestCompactProjectedStates(unittest.TestCase):
    def test_outputs_all_factor_gradients_and_unused_outputs(self):
        torch.manual_seed(762943)
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            cases = (
                ((2, 7, 3, 5, 7), (2, 3, 8, 2, 32)),
                ((3, 11, 16, 32, 17), (3, 5, 8, 16)),
                ((1, 5, 64, 128, 64), (2, 4, 8)),
                ((2, 3, 5, 7, 3), (8, 16)),
            )
            for shape, periods in cases:
                for selection in ('all', 'state_only', 'projection_only', 'skip_period', 'strided'):
                    with self.subTest(shape=shape, periods=periods, selection=selection):
                        raw = _oracle.inputs(shape, 'cuda', torch.float32)
                        if selection == 'strided':
                            raw = tuple(x.detach().transpose(-1, -2).contiguous()
                                        .transpose(-1, -2).requires_grad_() for x in raw)
                        full_inputs = tuple(x.detach().clone().requires_grad_() for x in raw)
                        exact_inputs = tuple(x.detach().double().requires_grad_() for x in raw)
                        compact = _oracle.helper.multi_projected_states(*raw, periods, active_outputs=True)
                        full = _oracle.helper.multi_projected_states(*full_inputs, periods)
                        exact = _oracle.reference(*exact_inputs, periods)
                        used = [[], [], []]
                        upstream = []
                        position = torch.arange(shape[1], device='cuda')
                        for index, (period, pair, full_pair, exact_pair) in enumerate(zip(periods, compact, full, exact)):
                            active = position[position % period >= period // 2]
                            for component, (got, complete, expected) in enumerate(zip(pair, full_pair, exact_pair)):
                                full_active = complete.index_select(1, active)
                                exact_active = expected.index_select(1, active)
                                self.assertEqual(got.shape[1], active.numel())
                                torch.testing.assert_close(got, full_active, rtol=0, atol=0)
                                torch.testing.assert_close(got.double(), exact_active, rtol=3e-5, atol=2e-7)
                                if ((selection == 'skip_period' and index == 1)
                                        or (selection == 'state_only' and component == 1)
                                        or (selection == 'projection_only' and component == 0)):
                                    continue
                                for values, output in zip(used, (got, full_active, exact_active)):
                                    values.append(output)
                                gradient = torch.randn_like(got)
                                if selection == 'strided':
                                    gradient = gradient.transpose(-1, -2).contiguous().transpose(-1, -2)
                                upstream.append(gradient)
                        compact_grad = torch.autograd.grad(tuple(used[0]), raw, grad_outputs=tuple(upstream))
                        full_grad = torch.autograd.grad(tuple(used[1]), full_inputs, grad_outputs=tuple(upstream))
                        exact_grad = torch.autograd.grad(tuple(used[2]), exact_inputs,
                                                        grad_outputs=tuple(x.double() for x in upstream))
                        for name, got, old, expected in zip(('k', 'z', 'factor', 'value', 'decay'),
                                                           compact_grad, full_grad, exact_grad):
                            self.assertTrue(torch.isfinite(got).all(), name)
                            # Compact indexing supplies precisely the same
                            # zeros and active gradients to the same scan.
                            torch.testing.assert_close(got, old, rtol=0, atol=0, msg=name)
                            torch.testing.assert_close(got.double(), expected, rtol=1e-4, atol=2e-6, msg=name)
                            if all(shape[1] <= period // 2 for period in periods):
                                self.assertEqual(torch.count_nonzero(got).item(), 0, name)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main(verbosity=2)
