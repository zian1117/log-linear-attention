"""Independent checks for selective period replacement and router gradients."""
import importlib
import importlib.util
import math
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import torch


class TestPeriodReplacementCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_dynamo = torch._dynamo.config.disable
        torch._dynamo.config.disable = True
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        name = '_period_repair_tests'
        package = types.ModuleType(name)
        package.__path__ = [str(Path(__file__).resolve().parents[1] / 'hattention')]
        # Only unused legacy normalization imports require this stub on CPU.
        stub = types.ModuleType('fla.modules.l2norm')
        stub.l2_norm = lambda x, *args, **kwargs: x
        sys.modules.setdefault(name, package)
        cls.previous_stub = sys.modules.get(stub.__name__)
        sys.modules.setdefault(stub.__name__, stub)
        cls.fast = importlib.import_module(name + '.fast_matrix_gdn')
        cls.adaptive = importlib.import_module(name + '.adaptive_matrix_gdn')
        path = Path(__file__).with_name('test_bilinear_matrix_gdn.py')
        spec = importlib.util.spec_from_file_location('period_full_matrix_oracle', path)
        cls.oracle = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.oracle)

    @classmethod
    def tearDownClass(cls):
        if cls.previous_stub is None:
            sys.modules.pop('fla.modules.l2norm', None)
        torch._dynamo.config.disable = cls.previous_dynamo
        torch.set_num_threads(cls.previous_threads)

    def test_mixed_heads_partial_periods_and_replacement_gradients(self):
        generator = torch.Generator().manual_seed(188)
        batch, chunks, chunk, value_dim = 6, 7, 4, 5
        length = chunks * chunk
        for level in (0, 2, 4, 5, 6):
            period = 1 << level
            groups = (length+period-1)//period
            ids = torch.tensor([i for i in range(batch*groups) if i % 5 in (0, 3)])
            raw = (torch.randn(batch, chunks, chunk, value_dim, generator=generator, dtype=torch.float64),
                   torch.randn(batch, chunks, chunk, generator=generator, dtype=torch.float64),
                   torch.randn(len(ids), period, value_dim, generator=generator, dtype=torch.float64),
                   torch.randn(len(ids), period, generator=generator, dtype=torch.float64))
            actual_inputs = tuple(x.clone().requires_grad_() for x in raw)
            expected_inputs = tuple(x.clone().requires_grad_() for x in raw)
            actual = self.fast._replace_selected_periods(*actual_inputs[:2], ids, *actual_inputs[2:], level)
            # Token-by-token construction, without reshape/index_copy, provides
            # independent coverage of flattening across heads and padded ends.
            lookup = {index: row for row, index in enumerate(ids.tolist())}
            y, scores, ry, rs = expected_inputs
            expected_y, expected_s = [], []
            for head in range(batch):
                yy, ss = [], []
                for token in range(length):
                    row = lookup.get(head*groups + token//period)
                    yy.append(y[head, token//chunk, token%chunk] if row is None else ry[row, token%period])
                    ss.append(scores[head, token//chunk, token%chunk] if row is None else rs[row, token%period])
                expected_y.append(torch.stack(yy)); expected_s.append(torch.stack(ss))
            expected = (torch.stack(expected_y).reshape_as(y), torch.stack(expected_s).reshape_as(scores))
            upstream = tuple(torch.randn(x.shape, generator=generator, dtype=x.dtype) for x in actual)
            ga = torch.autograd.grad(sum((x*z).sum() for x,z in zip(actual,upstream)), actual_inputs)
            ge = torch.autograd.grad(sum((x*z).sum() for x,z in zip(expected,upstream)), expected_inputs, allow_unused=True)
            for a,e in zip(actual,expected): torch.testing.assert_close(a,e,rtol=0,atol=0)
            for a,e in zip(ga,ge): torch.testing.assert_close(a,torch.zeros_like(a) if e is None else e,rtol=0,atol=0)

    @staticmethod
    def float_states(k, w, value, decay, period):
        state = k.new_zeros(k.shape[0], k.shape[-1], value.shape[-1])
        result = []
        for position in range(k.shape[1]):
            if position % period == 0: state = torch.zeros_like(state)
            result.append(state)
            residual = -w[:,position] @ state
            if position % period < period//2: residual = residual + value[:,position]
            state = decay[:,position,None,None]*state + k[:,position].transpose(-1,-2) @ residual
        return torch.stack(result,1)

    @staticmethod
    def reduce(values, logits, scale):
        total, levels = values.shape[1]*values.shape[2], values.shape[-2]
        positions, level = torch.arange(total), torch.arange(levels)
        active = (level[None,:]==0)|(((positions[:,None]>>(level[None,:]-1).clamp_min(0))&1)!=0)
        active = active.reshape(values.shape[1],values.shape[2],levels)
        weight = logits.masked_fill(~active,-torch.inf).softmax(-1)
        return (weight[...,None]*values).sum(-2).mul(scale).to(values.dtype)

    @staticmethod
    def tuple_reduce(values, logits, scale):
        return TestPeriodReplacementCPU.reduce(
            torch.stack(values, -2), torch.stack(logits, -1), scale)

    def test_full_router_mixed_heads_all_eight_gradients(self):
        generator = torch.Generator().manual_seed(321)
        def randn(*shape): return torch.randn(*shape,generator=generator)
        for length in (33, 73, 129):
            batch,heads,key_dim,value_dim=2,3,8,4
            k=randn(batch,length,heads,key_dim)
            beta=torch.full((batch,length,heads),.1)
            for b,h in ((0,1),(1,2)):
                k[b,:,h]=k[b,:1,h].expand(length,-1)
                beta[b,:,h]=.999
            raw=(randn(*k.shape),k,.1*randn(batch,length,heads,value_dim),
                 torch.full_like(beta,-.02),beta,randn(batch,length,heads,value_dim),
                 randn(*k.shape),torch.full((heads,),.5*math.log(key_dim*value_dim)))
            actual_inputs=tuple(x.clone().requires_grad_() for x in raw)
            expected_inputs=tuple(x.double().clone().requires_grad_() for x in raw)
            calls=[]
            real_replace=self.fast._replace_selected_periods
            def replace(*args,**kwargs):
                calls.append(args[2].numel())
                return real_replace(*args,**kwargs)
            with mock.patch.object(self.fast._FloatStates,'apply',self.float_states), mock.patch.object(self.fast,'tuple_routing_reduce',self.tuple_reduce), mock.patch.object(self.fast,'_replace_selected_periods',replace):
                actual,flagged=self.fast.fast_matrix_gdn(*actual_inputs,repair_periods=True)
                self.assertFalse(flagged.any())
                self.assertTrue(calls)
                expected=self.oracle.full_matrix_reference(*expected_inputs,gdn_norm_dtype=torch.float32)
                upstream=randn(*actual.shape)
                ga=torch.autograd.grad((actual*upstream).sum(),actual_inputs)
                ge=torch.autograd.grad((expected*upstream.double()).sum(),expected_inputs)
            torch.testing.assert_close(actual.double(),expected,rtol=5e-4,atol=3e-6)
            for name,a,e in zip(('r','k','v','g','beta','u','q','temperature'),ga,ge):
                self.assertTrue(torch.isfinite(a).all(),name)
                relative=(a.double()-e).norm()/e.norm().clamp_min(1e-8)
                self.assertLess(relative.item(),.002,name)


if __name__ == '__main__':
    unittest.main(verbosity=2)
