"""Independent affine recurrence and chain-rule checks for shared projections."""
import importlib
import importlib.util
import math
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import torch


def reference(k, z, factor, value, decay, period):
    """Form full affine transition matrices; projection is a separate output."""
    identity = torch.eye(k.shape[-1], device=k.device, dtype=k.dtype)
    state = k.new_zeros(k.shape[0], k.shape[-1], value.shape[-1])
    states, projections = [], []
    for position in range(k.shape[1]):
        if position % period == 0: state = torch.zeros_like(state)
        states.append(state)
        projections.append(z[:,position] @ state)
        transition = (decay[:,position,None,None]*identity
                      - k[:,position].transpose(-1,-2) @ (factor[:,position,:,None]*z[:,position]))
        state = transition @ state
        if position % period < period//2:
            state = state + k[:,position].transpose(-1,-2) @ value[:,position]
    zero = sum(x.sum()*0 for x in (k,z,factor,value,decay))
    return torch.stack(states,1)+zero, torch.stack(projections,1)+zero


def load_helper():
    name = '_projected_state_tests'
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(Path(__file__).resolve().parents[1]/'hattention')]
        sys.modules[name] = package
    return importlib.import_module(name+'.projected_bucket_states')


class TestProjectedStateMathCPU(unittest.TestCase):
    def test_combined_adjoint_and_factor_formulas(self):
        helper=load_helper()
        factor_gradients=getattr(helper._factor_gradients,'_torchdynamo_orig_callable',helper._factor_gradients)
        generator=torch.Generator().manual_seed(851)
        for chunks,period in ((1,2),(7,3),(9,4),(3,8)):
            k=.03*torch.randn(2,chunks,3,5,generator=generator,dtype=torch.float64)
            raw=(k,.1*k.clone(),torch.rand(2,chunks,3,generator=generator,dtype=torch.float64),
                 torch.randn(2,chunks,3,7,generator=generator,dtype=torch.float64),
                 torch.full((2,chunks),.97,dtype=torch.float64))
            inputs=tuple(x.requires_grad_() for x in raw)
            state,projected=reference(*inputs,period)
            direct=torch.randn(state.shape,generator=generator,dtype=torch.float64)
            dp=torch.randn(projected.shape,generator=generator,dtype=torch.float64)
            expected=torch.autograd.grad((state*direct).sum()+(projected*dp).sum(),inputs)
            k,z,factor,value,decay=(x.detach() for x in inputs)
            adjoint=torch.zeros_like(state[:,0]);aa,dd,cc=[None]*chunks,[None]*chunks,[None]*chunks
            for i in range(chunks-1,-1,-1):
                if i==chunks-1 or (i+1)%period==0:adjoint=torch.zeros_like(adjoint)
                aa[i]=adjoint
                dd[i]=k[:,i]@adjoint
                cc[i]=dp[:,i]-factor[:,i,:,None]*dd[i]
                adjoint=decay[:,i,None,None]*adjoint+z[:,i].transpose(-1,-2)@cc[i]+direct[:,i]
            actual=factor_gradients(factor,value,state.detach(),projected.detach(),
                                   torch.stack(aa,1),torch.stack(dd,1),torch.stack(cc,1),period)
            for name,a,e in zip(('k','z','factor','value','decay'),actual,expected):
                torch.testing.assert_close(a,e,rtol=1e-11,atol=1e-12,msg=lambda msg:f'{name}: {msg}')


@unittest.skipUnless(torch.cuda.is_available(),'requires CUDA')
class TestProjectedStateGPU(unittest.TestCase):
    def test_outputs_and_all_factor_gradients(self):
        helper=load_helper()
        generator=torch.Generator(device='cuda').manual_seed(439)
        def randn(*shape):return torch.randn(*shape,generator=generator,device='cuda')
        for chunks,period,c,k,v in ((1,2,16,16,7),(7,3,16,16,9),(7,3,7,13,9),(9,4,32,32,33),(3,8,16,32,5),(257,256,16,16,7)):
            raw=(.03*randn(2,chunks,c,k),.03*randn(2,chunks,c,k),
                 .1*torch.rand(2,chunks,c,generator=generator,device='cuda'),
                 randn(2,chunks,c,v),torch.full((2,chunks),.99,device='cuda'))
            for loss in ('state','projection','both'):
                with self.subTest(chunks=chunks,period=period,loss=loss):
                    actual_inputs=tuple(x.clone().requires_grad_() for x in raw)
                    expected_inputs=tuple(x.double().clone().requires_grad_() for x in raw)
                    pc,pk=max(16,1<<(c-1).bit_length())-c,max(16,1<<(k-1).bit_length())-k
                    padded=(torch.nn.functional.pad(actual_inputs[0],(0,pk,0,pc)),
                            torch.nn.functional.pad(actual_inputs[1],(0,pk,0,pc)),
                            torch.nn.functional.pad(actual_inputs[2],(0,pc)),
                            torch.nn.functional.pad(actual_inputs[3],(0,0,0,pc)),actual_inputs[4])
                    state,projection=helper.ProjectedStates.apply(*padded,period)
                    actual=(state[...,:k,:],projection[...,:c,:])
                    expected=reference(*expected_inputs,period)
                    up=tuple(randn(*x.shape) for x in actual)
                    for a,e in zip(actual,expected):torch.testing.assert_close(a.double(),e,rtol=2e-4,atol=2e-6)
                    indexes=(0,) if loss=='state' else (1,) if loss=='projection' else (0,1)
                    ga=torch.autograd.grad(sum((actual[i]*up[i]).sum() for i in indexes),actual_inputs)
                    ge=torch.autograd.grad(sum((expected[i]*up[i].double()).sum() for i in indexes),expected_inputs)
                    for name,a,e in zip(('k','z','factor','value','decay'),ga,ge):
                        self.assertTrue(torch.isfinite(a).all(),name)
                        torch.testing.assert_close(a.double(),e,rtol=3e-4,atol=3e-5,msg=lambda msg:f'{name}: {msg}')
                        self.assertLess(((a.double()-e).norm()/e.norm().clamp_min(1e-7)).item(),1e-4,name)
                    suppressed=torch.arange(chunks,device='cuda')%period>=period//2
                    self.assertTrue((ga[3][:,suppressed]==0).all())

    def test_outer_factor_chain_rule_matches_old_weight(self):
        helper=load_helper()
        generator=torch.Generator(device='cuda').manual_seed(435)
        def randn(*shape):return torch.randn(*shape,generator=generator,device='cuda')
        batch,chunks,c,k,v,period=3,9,16,32,13,4
        raw=(.03*randn(batch,chunks,c,k),.03*randn(batch,chunks,c,k),
             torch.sigmoid(randn(batch,chunks,c)),-torch.rand(batch,chunks,c,generator=generator,device='cuda'),
             randn(batch,chunks,c,v),torch.full((batch,chunks),.9,device='cuda'))
        actual_inputs=tuple(x.clone().requires_grad_() for x in raw)
        expected_inputs=tuple(x.double().clone().requires_grad_() for x in raw)
        kk,zz,beta,gc,value,decay=actual_inputs
        actual=helper.ProjectedStates.apply(kk,zz,beta*gc.exp(),value,decay,period)
        kk,zz,beta,gc,value,decay=expected_inputs
        factor=beta*gc.exp()
        # Recreate the old erase-weight parameterization explicitly.
        w=factor.unsqueeze(-1)*zz
        state=kk.new_zeros(batch,k,v);states=[];projections=[]
        eye=torch.eye(k,device='cuda',dtype=torch.float64)
        for i in range(chunks):
            if i%period==0:state=torch.zeros_like(state)
            states.append(state);projections.append(zz[:,i]@state)
            state=(decay[:,i,None,None]*eye-kk[:,i].transpose(-1,-2)@w[:,i])@state
            if i%period<period//2:state=state+kk[:,i].transpose(-1,-2)@value[:,i]
        expected=(torch.stack(states,1),torch.stack(projections,1))
        up=tuple(randn(*x.shape) for x in actual)
        ga=torch.autograd.grad(sum((a*u).sum() for a,u in zip(actual,up)),actual_inputs)
        ge=torch.autograd.grad(sum((a*u.double()).sum() for a,u in zip(expected,up)),expected_inputs)
        for a,e in zip(actual,expected):torch.testing.assert_close(a.double(),e,rtol=2e-4,atol=2e-6)
        for name,a,e in zip(('k','z','beta','gc','value','decay'),ga,ge):
            torch.testing.assert_close(a.double(),e,rtol=3e-4,atol=3e-5,msg=lambda msg:f'{name}: {msg}')
            self.assertLess(((a.double()-e).norm()/e.norm().clamp_min(1e-7)).item(),1e-4,name)


if __name__=='__main__':unittest.main(verbosity=2)
