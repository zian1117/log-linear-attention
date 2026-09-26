"""Independent recurrence oracle for joint projected-state scans."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest

import torch

package_name = '_multi_projected_test_helpers'
package = types.ModuleType(package_name)
package.__path__ = [str(Path(__file__).resolve().parents[1]/'hattention')]
sys.modules.setdefault(package_name,package)
spec=importlib.util.spec_from_file_location(
    package_name+'.multi_projected_states',Path(package.__path__[0])/'multi_projected_states.py')
helper=importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def reference(k,z,factor,value,decay,periods):
    result=[]
    for period in periods:
        states,projected=[],[]
        state=k.new_zeros(k.shape[0],k.shape[-1],value.shape[-1])
        for chunk in range(k.shape[1]):
            if chunk%period == 0:
                state=torch.zeros_like(state)
            read=z[:,chunk]@state
            states.append(state)
            projected.append(read)
            write=value[:,chunk] if chunk%period < period//2 else torch.zeros_like(value[:,chunk])
            residual=write-factor[:,chunk,:,None]*read
            state=decay[:,chunk,None,None]*state+k[:,chunk].transpose(-1,-2)@residual
        result.append((torch.stack(states,1),torch.stack(projected,1)))
    return tuple(result)


def inputs(shape,device,dtype):
    b,n,c,k,v=shape
    key=torch.randn(b,n,c,k,device=device,dtype=dtype)*.05
    probe=torch.randn_like(key)*.05
    factor=torch.rand(b,n,c,device=device,dtype=dtype)*.5
    value=torch.randn(b,n,c,v,device=device,dtype=dtype)*.1
    decay=torch.rand(b,n,device=device,dtype=dtype)*.3+.6
    return tuple(x.requires_grad_() for x in (key,probe,factor,value,decay))


class MultiProjectedCPU(unittest.TestCase):
    def test_baddbmm_accumulation_formula(self):
        torch.manual_seed(227)
        data=inputs((2,7,3,5,7),'cpu',torch.float64)
        periods=(2,3,8)
        outputs=reference(*data,periods)
        gradients=tuple(tuple(torch.randn_like(x) for x in pair) for pair in outputs)
        wanted=torch.autograd.grad(tuple(x for pair in outputs for x in pair),data,
                                   grad_outputs=tuple(x for pair in gradients for x in pair))
        k,z,factor,value,decay=(x.detach() for x in data)
        accum=[torch.zeros_like(x) for x in data]
        dk,dz,df,dv,da=accum
        for period,pair,upstream in zip(periods,outputs,gradients):
            states,projected=(x.detach() for x in pair)
            direct,dp=upstream
            for start in range(0,k.shape[1],period):
                adjoint=torch.zeros_like(states[:,0])
                for chunk in range(min(start+period,k.shape[1])-1,start-1,-1):
                    write=(chunk-start)<period//2
                    residual=(value[:,chunk] if write else 0.)-factor[:,chunk,:,None]*projected[:,chunk]
                    du=k[:,chunk]@adjoint
                    combined=dp[:,chunk]-factor[:,chunk,:,None]*du
                    torch.baddbmm(dk[:,chunk],residual,adjoint.transpose(-1,-2),out=dk[:,chunk])
                    torch.baddbmm(dz[:,chunk],combined,states[:,chunk].transpose(-1,-2),out=dz[:,chunk])
                    df[:,chunk]-=(du*projected[:,chunk]).sum(-1)
                    if write:
                        dv[:,chunk]+=du
                    da[:,chunk]+=(states[:,chunk]*adjoint).sum((-2,-1))
                    adjoint=decay[:,chunk,None,None]*adjoint+z[:,chunk].transpose(-1,-2)@combined+direct[:,chunk]
        for got,want in zip(accum,wanted):
            torch.testing.assert_close(got,want,rtol=1e-12,atol=1e-12)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class MultiProjectedGPU(unittest.TestCase):
    def test_outputs_all_gradients_and_unused_outputs(self):
        torch.manual_seed(947)
        torch.backends.cuda.matmul.allow_tf32=False
        for shape,periods in (((2,5,3,7,5),(2,3,8,2)),((1,9,16,32,16),(2,4,16)),
                              ((1,5,64,128,64),(2,4,8))):
            for selection in ('all','state_only','projection_only','skip_period','strided'):
                with self.subTest(shape=shape,periods=periods,selection=selection):
                    data=inputs(shape,'cuda',torch.float32)
                    if selection=='strided':
                        data=tuple(x.detach().transpose(-1,-2).contiguous().transpose(-1,-2).requires_grad_()
                                   for x in data)
                    exact=tuple(x.detach().double().requires_grad_() for x in data)
                    expected=reference(*exact,periods)
                    actual=helper.multi_projected_states(*data,periods)
                    for pair,refpair in zip(actual,expected):
                        for got,want in zip(pair,refpair):
                            torch.testing.assert_close(got.double(),want,rtol=3e-5,atol=2e-7)
                    got_used,want_used,upstream=[],[],[]
                    for i,(pair,refpair) in enumerate(zip(actual,expected)):
                        if selection=='skip_period' and i==1:
                            continue
                        for j,(got,want) in enumerate(zip(pair,refpair)):
                            if selection=='state_only' and j==1 or selection=='projection_only' and j==0:
                                continue
                            got_used.append(got)
                            want_used.append(want)
                            upstream.append(torch.randn_like(got))
                    got_grad=torch.autograd.grad(tuple(got_used),data,grad_outputs=tuple(upstream))
                    want_grad=torch.autograd.grad(tuple(want_used),exact,grad_outputs=tuple(x.double() for x in upstream))
                    for got,want in zip(got_grad,want_grad):
                        torch.testing.assert_close(got.double(),want,rtol=1e-4,atol=2e-6)


if __name__=='__main__':
    unittest.main(verbosity=2)
