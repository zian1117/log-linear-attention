"""Independent indexed references for joint Fenwick gradient accumulation."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest

import torch

# Import only the small helpers, avoiding optional model runtime dependencies.
package_name = '_multi_select_test_helpers'
package = types.ModuleType(package_name)
package.__path__ = [str(Path(__file__).resolve().parents[1]/'hattention')]
sys.modules.setdefault(package_name, package)
spec = importlib.util.spec_from_file_location(
    package_name+'.multi_active_select', Path(package.__path__[0])/'multi_active_select.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class MultiActiveSelectTests(unittest.TestCase):
    def check_device(self, device):
        torch.manual_seed(907)
        dtypes = (torch.float32, torch.float64) if device == 'cpu' else (
            torch.float32, torch.float64, torch.float16, torch.bfloat16)
        for dtype in dtypes:
            for chunks, periods in ((0,(2,4)), (1,(2,8)), (3,(2,3,8)),
                                   (7,(2,3,4)), (9,(8,2,4,2)),
                                   (17,(2,4,8,16,32)), (257,(2,16,256,512))):
                for tail in ((), (5,3)):
                    with self.subTest(device=device,dtype=dtype,chunks=chunks,periods=periods,tail=tail):
                        # Noncontiguous batch/chunk layout and noncontiguous tail.
                        raw = torch.randn(chunks,2,*tail,device=device,dtype=dtype).transpose(0,1)
                        if tail:
                            raw=raw.transpose(-1,-2)
                        x=raw.detach().requires_grad_()
                        reference=raw.detach().clone().requires_grad_()
                        actual=helper.multi_active_select(x,periods)
                        indices=[torch.tensor([i for i in range(chunks) if i%p>=p//2],
                                              device=device,dtype=torch.long) for p in periods]
                        expected=tuple(reference.index_select(1,index) for index in indices)
                        for got,want in zip(actual,expected):
                            torch.testing.assert_close(got,want,rtol=0,atol=0)
                            self.assertTrue(got.is_contiguous())
                        # Deliberately leave some outputs unused. Others have
                        # strided upstream gradients, including empty outputs.
                        used=tuple(i for i in range(len(periods)) if i%3 != 1)
                        upstream=[]
                        for index in used:
                            shape=actual[index].shape
                            grad=(torch.randn(*shape[:-2],shape[-1],shape[-2],device=device,dtype=dtype).transpose(-1,-2)
                                  if tail else torch.randn(shape,device=device,dtype=dtype))
                            upstream.append(grad)
                        got,=torch.autograd.grad(tuple(actual[i] for i in used),x,grad_outputs=tuple(upstream))
                        want,=torch.autograd.grad(tuple(expected[i] for i in used),reference,grad_outputs=tuple(upstream))
                        if dtype==torch.float64:
                            tolerance=1e-12
                        elif dtype==torch.bfloat16:
                            tolerance=.03
                        elif dtype==torch.float16:
                            tolerance=.003
                        else:
                            tolerance=1e-6
                        torch.testing.assert_close(got,want,rtol=tolerance,atol=tolerance)
                        self.assertEqual(got.dtype,dtype)

    def test_cpu(self):
        self.check_device('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda(self):
        self.check_device('cuda')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_compiled_dynamic_batch(self):
        compiled=torch.compile(helper.multi_active_select,fullgraph=True,dynamic=True)
        periods=(2,3,8,16)
        for batch in (12,48,12):
            x=torch.randn(batch,9,3,5,device='cuda',requires_grad=True)
            outputs=compiled(x,periods)
            upstream=tuple(torch.randn_like(y) for y in outputs)
            got,=torch.autograd.grad(outputs,x,grad_outputs=upstream)
            reference=torch.zeros_like(x)
            for period,grad in zip(periods,upstream):
                index=torch.tensor([i for i in range(9) if i%period>=period//2],device='cuda')
                reference.index_add_(1,index,grad)
            torch.testing.assert_close(got,reference,rtol=1e-6,atol=1e-6)

    def test_empty_periods_and_validation(self):
        self.assertEqual(helper.multi_active_select(torch.ones(2,3),()),())
        for periods in ((1,), (True,), (2.5,)):
            with self.assertRaises(ValueError):
                helper.multi_active_select(torch.ones(2,3),periods)


if __name__=='__main__':
    unittest.main(verbosity=2)
