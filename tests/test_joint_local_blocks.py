"""Regression tests for joint Fenwick block views and their summed adjoint.

"""
import importlib
from pathlib import Path
import sys
import types
import unittest
import torch


def module():
    name='_joint_blocks_tests'
    if name not in sys.modules:
        package=types.ModuleType(name)
        package.__path__=[str(Path(__file__).resolve().parents[1]/'hattention')]
        sys.modules[name]=package
    return importlib.import_module(name+'.joint_local_blocks')


def indices(width,period,device):
    rows=torch.arange(width//period,device=device)[:,None]*period+torch.arange(period,device=device)[None,:]
    return rows.unsqueeze(-1),rows.unsqueeze(-2)


class TestJointLocalBlocks(unittest.TestCase):
    def check_adjoint(self,device):
        implementation=module()
        generator=torch.Generator().manual_seed(816)
        for dtype in (torch.float32,torch.float64):
            for width,periods in ((2,(2,)),(8,(2,4,8)),(64,(2,8,64,8))):
                for strided in (False,True):
                    shape=(2,3,width,width*(2 if strided else 1))
                    storage=torch.randn(shape,dtype=dtype,generator=generator).to(device)
                    x=(storage[...,::2] if strided else storage).detach().requires_grad_()
                    for used in (tuple(range(len(periods))), (0,), (len(periods)-1,)):
                        with self.subTest(device=device,dtype=dtype,width=width,periods=periods,strided=strided,used=used):
                            outputs=implementation.joint_blocks(x,periods)
                            weights=[torch.randn(y.shape,dtype=dtype,generator=generator).to(device) for y in outputs]
                            for p,got in zip(periods,outputs):
                                row,column=indices(width,p,device)
                                expected=x[...,row,column]
                                torch.testing.assert_close(got,expected,rtol=0,atol=0)
                            loss=sum((outputs[i]*weights[i]).sum() for i in used)
                            actual=torch.autograd.grad(loss,x)[0]
                            # Enumerate output-to-source index mappings into an
                            # FP64 accumulation, independent of diagonal views.
                            expected=torch.zeros((*x.shape[:-2],width*width),dtype=torch.float64,device=device)
                            for i in used:
                                row,column=indices(width,periods[i],device)
                                offsets=(row*width+column).reshape(-1)
                                index=offsets.expand(*x.shape[:-2],offsets.numel())
                                expected.scatter_add_(-1,index,weights[i].double().flatten(-3))
                            expected=expected.reshape_as(x)
                            tolerance=2e-6 if dtype==torch.float32 else 2e-14
                            torch.testing.assert_close(actual.double(),expected,rtol=tolerance,atol=tolerance)
                            self.assertTrue(torch.isfinite(actual).all())

    def test_cpu_multilevel_sum_unused_and_strided(self):
        self.check_adjoint('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_multilevel_sum_unused_and_strided(self):
        self.check_adjoint('cuda')

    def test_empty_and_all_none(self):
        implementation=module()
        x=torch.randn(2,3,8,8,requires_grad=True)
        self.assertEqual(implementation.joint_blocks(x,()),())
        class Context:
            def set_materialize_grads(self,value):pass
        ctx=Context()
        implementation._JointBlocks.forward(ctx,x,(2,4,8))
        gradient,period_gradient=implementation._JointBlocks.backward(ctx,None,None,None)
        self.assertIsNone(period_gradient)
        torch.testing.assert_close(gradient,torch.zeros_like(x),rtol=0,atol=0)

    def test_checkpoint_matches_plain(self):
        from torch.utils.checkpoint import checkpoint
        implementation=module()
        torch.manual_seed(617)
        raw=torch.randn(2,3,8,8,dtype=torch.float64)
        saved=[]
        for enabled in (False,True):
            leaf=raw.detach().clone().requires_grad_()
            def call(x):return implementation.joint_blocks(x,(2,4,8,4))
            outputs=checkpoint(call,leaf,use_reentrant=False) if enabled else call(leaf)
            loss=sum((i+1)*y.square().sum() for i,y in enumerate(outputs))
            gradient=torch.autograd.grad(loss,leaf)[0]
            saved.append((outputs,gradient))
        for a,b in zip(saved[0][0],saved[1][0]):torch.testing.assert_close(a,b,rtol=0,atol=0)
        torch.testing.assert_close(saved[0][1],saved[1][1],rtol=0,atol=0)


if __name__=='__main__':
    torch.set_num_threads(4)
    unittest.main()
