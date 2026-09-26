"""Unpacked hierarchy reduction versus retained Triton and torch oracles."""
import importlib
from pathlib import Path
import sys
import unittest
import types
import torch
import os

ROOT=Path(__file__).resolve().parents[1]/'hattention'
package=types.ModuleType('_tuple_reduce_tests');package.__path__=[str(ROOT)]
sys.modules.setdefault(package.__name__,package)
old=importlib.import_module(package.__name__+'.bilinear_matrix_gdn')._RoutingReduce
tuple_routing_reduce=importlib.import_module(package.__name__+'.tuple_routing_reduce').tuple_routing_reduce


def reference(ys,scores,scale):
    values=torch.stack(ys,-2).float()
    logits=torch.stack(scores,-1)
    _,n,c,levels=logits.shape
    t=torch.arange(n*c,device=logits.device).reshape(n,c,1)
    level=torch.arange(levels,device=logits.device)
    active=(level==0)|((t>>((level-1).clamp_min(0)))&1).bool()
    masked=logits.masked_fill(~active,-torch.inf)
    shifted=(masked-masked.amax(-1,keepdim=True)).float()
    weights=shifted.softmax(-1)
    return (weights.unsqueeze(-1)*values).sum(-2).mul(scale).to(ys[0].dtype)


def metric(a,e):
    error=(a.double()-e.double())
    return dict(maxabs=error.abs().max().item(),relative=(error.norm()/e.double().norm().clamp_min(1e-300)).item())


def run(device):
    torch.manual_seed(91653)
    report=[]
    cases=[((2,2,3,7),5,'normal'),((1,1,13,3),6,'large_finite'),
           ((2,1,3,11),15,'inactive'),((1,1,1,64),1,'normal'),
           ((1,2,5,9),4,'equal'),((1,2,7,5),5,'strided')]
    for dtype in (torch.float32,torch.bfloat16):
        for shape,levels,kind in cases:
            ys=[torch.randn(shape,device=device,dtype=dtype)for _ in range(levels)]
            scores=[torch.randn(shape[:-1],device=device,dtype=torch.float64)*3 for _ in range(levels)]
            if kind=='large_finite':
                for level in range(levels):
                    scores[level].fill_(1e40 if level%2 else -1e40)
                    scores[level][...,::3]=1e300
                    scores[level][...,1::3]=-1e300
            elif kind=='inactive':
                # Deliberately dominant inactive levels must not enter softmax.
                for level in range(3,levels):scores[level].fill_(1e300)
            elif kind=='equal':
                for score in scores:score.fill_(1e200)
            elif kind=='strided':
                ys=[x.transpose(1,2).contiguous().transpose(1,2)for x in ys]
                scores=[x.transpose(1,2).contiguous().transpose(1,2)for x in scores]
            inputs=[tuple(x.detach().clone().requires_grad_()for x in ys+scores)for _ in range(3)]
            scale=19**-.5
            actual=tuple_routing_reduce(inputs[0][:levels],inputs[0][levels:],scale)
            previous=old.apply(torch.stack(inputs[1][:levels],-2),torch.stack(inputs[1][levels:],-1),scale)
            expected=reference(inputs[2][:levels],inputs[2][levels:],scale)
            # Tuple loads preserve the same FP32 tree reductions as the old
            # implementation; the torch oracle can use another reduction order.
            torch.testing.assert_close(actual,previous,rtol=0,atol=0)
            torch.testing.assert_close(actual.float(),expected.float(),rtol=.008 if dtype==torch.bfloat16 else 2e-6,
                                       atol=.002 if dtype==torch.bfloat16 else 2e-7)
            upstream=torch.randn_like(actual)
            if kind == 'strided':
                upstream = upstream.transpose(1, 2).contiguous().transpose(1, 2)
                assert not ys[0].is_contiguous()
                assert not upstream.is_contiguous()
            gradients=[torch.autograd.grad(o,i,upstream)for o,i in zip((actual,previous,expected),inputs)]
            for a,b,e in zip(*gradients):
                assert torch.isfinite(a).all()
                assert a.is_contiguous()
                torch.testing.assert_close(a,b,rtol=0,atol=0)
                torch.testing.assert_close(a.double(),e.double(),rtol=.01 if a.dtype==torch.bfloat16 else 2e-5,
                                           atol=.002 if a.dtype==torch.bfloat16 else 4e-7)
            row=dict(dtype=str(dtype),kind=kind,shape=shape,levels=levels,forward=metric(actual,expected),
                     values=max(metric(a,e)['maxabs']for a,e in zip(gradients[0][:levels],gradients[2][:levels])),
                     logits=max(metric(a,e)['maxabs']for a,e in zip(gradients[0][levels:],gradients[2][levels:])))
            report.append(row)
    return report

class TestTupleRoutingReduce(unittest.TestCase):
    def test_input_structure(self):
        value = torch.ones(1, 2, 3, 5)
        score = torch.ones(1, 2, 3, dtype=torch.float64)
        for values, scores in (([], []), ([value], []), ([value], [score.float()]),
                               ([value, value[..., :4]], [score, score]),
                               ([value], [score[..., :2]])):
            with self.subTest(values=len(values), scores=len(scores)):
                with self.assertRaises(ValueError):
                    tuple_routing_reduce(values, scores, 7**-.5)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_values_all_gradients_and_extreme_scores(self):
        run('cuda')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_non_reentrant_checkpoint_preserves_all_gradients(self):
        from torch.utils.checkpoint import checkpoint
        torch.manual_seed(371)
        levels, shape = 5, (2, 2, 5, 7)
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                raw = tuple(torch.randn(shape, device='cuda', dtype=dtype)
                            for _ in range(levels))
                raw += tuple(torch.randn(shape[:-1], device='cuda', dtype=torch.float64)
                             for _ in range(levels))
                actual = tuple(x.detach().requires_grad_() for x in raw)
                expected = tuple(x.detach().clone().requires_grad_() for x in raw)

                def reduce(*inputs):
                    return tuple_routing_reduce(inputs[:levels], inputs[levels:], 19**-.5)

                result = checkpoint(reduce, *actual, use_reentrant=False)
                reference_output = reduce(*expected)
                gradient = torch.randn_like(result)
                got = torch.autograd.grad(result, actual, gradient)
                want = torch.autograd.grad(reference_output, expected, gradient)
                torch.testing.assert_close(result, reference_output, rtol=0, atol=0)
                for a, b in zip(got, want):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)

    @unittest.skipUnless(os.environ.get('TRITON_INTERPRET') == '1',
                         'set TRITON_INTERPRET=1 for CPU kernel interpretation')
    def test_cpu_interpreter_values_all_gradients_and_extreme_scores(self):
        run('cpu')


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main(verbosity=2)
