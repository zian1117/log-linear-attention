"""Independent calculus and CUDA checks for the precomputed-projection coarse core."""
import math
import unittest
import torch


def reference(projected, beta, k, gc, state, rn, qn, u, temperature, floor, dtype):
    key2 = k.square().sum(-1)
    coefficient = beta*(2-beta*key2)
    energy = coefficient*projected.square().sum(-1)
    raw = state.square().sum((-2,-1)).unsqueeze(-1)-energy.cumsum(-1)
    gain = (2*gc).exp().to(state.dtype)
    norm = gain*raw
    numerator = ((qn @ state)*u).sum(-1)
    score = temperature*numerator*norm.clamp_min(floor*floor).rsqrt()
    return (rn @ state).to(dtype), score


def analytical(inputs, dy, ds, floor):
    """Direct chain rule, tested independently against PyTorch autograd on CPU."""
    zz,beta,k,gc,h,r,q,u,temp=inputs
    qq=q@h; kn=k.square().sum(-1)
    a=beta*(2-beta*kn); e=a*zz.square().sum(-1)
    raw=h.square().sum((-2,-1)).unsqueeze(-1)-e.cumsum(-1)
    gain=(2*gc).exp(); norm=gain*raw
    f=norm.clamp_min(floor*floor).rsqrt(); numerator=(qq*u).sum(-1)
    dn=ds*temp*f; df=ds*temp*numerator
    dd=torch.where(norm>=floor*floor,-.5*df*f.pow(3),0.)
    dr=dd*gain
    de=-dr.flip(-1).cumsum(-1).flip(-1)
    da=de*zz.square().sum(-1)
    dz=2*(de*a).unsqueeze(-1)*zz
    dq=dn.unsqueeze(-1)*u
    dh=r.transpose(-1,-2)@dy+q.transpose(-1,-2)@dq
    dh=dh+2*dr.sum(-1)[...,None,None]*h
    return (dz,da*(2-2*beta*kn),-2*(da*beta.square()).unsqueeze(-1)*k,
            2*dd*raw*gain,dh,dy@h.transpose(-1,-2),dq@h.transpose(-1,-2),
            dn.unsqueeze(-1)*qq,(ds*numerator*f).sum_to_size(temp.shape))


def make_inputs(bh,n,c,kdim,vdim,device,dtype=torch.float32,scale=1.):
    k=torch.nn.functional.normalize(torch.randn(bh,n,c,kdim,device=device,dtype=dtype),dim=-1)
    beta=torch.rand(bh,n,c,device=device,dtype=dtype)*.8+.1
    gram=k@k.transpose(-1,-2)
    eye=torch.eye(c,device=device,dtype=dtype).expand_as(gram)
    inv=torch.linalg.solve_triangular(eye+(gram*beta.unsqueeze(-2)).tril(-1),eye,upper=False,unitriangular=True)
    z=inv@k
    h=torch.randn(bh,n,kdim,vdim,device=device,dtype=dtype)*scale
    r=torch.randn_like(k);q=torch.randn_like(k)
    u=torch.nn.functional.normalize(torch.randn(bh,n,c,vdim,device=device,dtype=dtype),dim=-1)
    gc=-torch.rand(bh,n,c,device=device,dtype=torch.float64).cumsum(-1)*.03
    temp=torch.full((bh,1,1),math.sqrt(kdim*vdim),device=device,dtype=torch.float64)
    return tuple(x.detach().requires_grad_() for x in (z@h,beta,k,gc,h,r,q,u,temp))


class CalculusCPU(unittest.TestCase):
    def test_all_gradients_both_floor_branches(self):
        torch.manual_seed(904)
        for scale in (1.,1e-8,0.):
            with self.subTest(scale=scale):
                inputs=make_inputs(2,3,9,7,5,'cpu',torch.float64,scale)
                y,s=reference(*inputs,1e-6,torch.float64)
                dy,ds=torch.randn_like(y),torch.randn_like(s)
                want=torch.autograd.grad((y*dy).sum()+(s*ds).sum(),inputs)
                got=analytical(inputs,dy,ds,1e-6)
                for actual,expected in zip(got,want):
                    torch.testing.assert_close(actual,expected,atol=1e-10,rtol=1e-10)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class FusedGPU(unittest.TestCase):
    def test_forward_and_nine_gradients(self):
        from hattention.fused_projected_coarse_router import fused_projected_coarse_core
        torch.backends.cuda.matmul.allow_tf32=False
        torch.manual_seed(921)
        shapes=((2,3,9,7,5),(2,2,64,128,64),(1,1,3,3,5))
        for shape in shapes:
            for output_dtype in (torch.float32,torch.bfloat16):
                for scale in (1.,1e-8,0.):
                    with self.subTest(shape=shape,dtype=output_dtype,scale=scale):
                        inputs=make_inputs(*shape,'cuda',scale=scale)
                        y,s=reference(*inputs,1e-6,output_dtype)
                        dy=torch.randn_like(y);ds=torch.randn_like(s)
                        want=torch.autograd.grad((y.float()*dy.float()).sum()+(s*ds).sum(),inputs)
                        mass=torch.ones(shape[:2],device='cuda')
                        fy,fs,flag=fused_projected_coarse_core(*inputs,mass,1e-6,output_dtype)
                        got=torch.autograd.grad((fy.float()*dy.float()).sum()+(fs*ds).sum(),inputs)
                        torch.testing.assert_close(fy,y,rtol=8e-3 if output_dtype==torch.bfloat16 else 2e-4,atol=2e-5)
                        torch.testing.assert_close(fs,s,rtol=2e-4,atol=2e-4)
                        self.assertEqual(flag.shape,(shape[0],))
                        for index,(actual,expected) in enumerate(zip(got,want)):
                            self.assertTrue(torch.isfinite(actual).all(),index)
                            scale_grad=max(1.,expected.abs().max().item())
                            torch.testing.assert_close(actual,expected,rtol=3e-4,atol=3e-5*scale_grad,msg=f'input gradient {index}')

    def test_dynamic_batch_forward_and_backward(self):
        from hattention.fused_projected_coarse_router import fused_projected_coarse_core
        compiled = torch.compile(fused_projected_coarse_core, fullgraph=True, dynamic=True)
        torch.manual_seed(712)
        # Twelve heads: batch 1 -> batch 4 -> batch 1, with one compiled wrapper.
        for bh in (12, 48, 12):
            with self.subTest(batch=bh//12):
                inputs = make_inputs(bh, 2, 64, 128, 64, 'cuda')
                mass = torch.ones(bh, 2, device='cuda')
                y, score = reference(*inputs, 1e-6, torch.float32)
                dy, ds = torch.randn_like(y), torch.randn_like(score)
                expected = torch.autograd.grad((y*dy).sum()+(score*ds).sum(), inputs)
                actual_y, actual_score, _ = compiled(*inputs, mass, 1e-6, torch.float32)
                actual = torch.autograd.grad((actual_y*dy).sum()+(actual_score*ds).sum(), inputs)
                torch.testing.assert_close(actual_y, y, rtol=2e-4, atol=2e-5)
                torch.testing.assert_close(actual_score, score, rtol=2e-4, atol=2e-4)
                for index, (got, want) in enumerate(zip(actual, expected)):
                    self.assertTrue(torch.isfinite(got).all(), index)
                    torch.testing.assert_close(got, want, rtol=3e-4,
                        atol=3e-5*max(1., want.abs().max().item()), msg=f'input gradient {index}')

    def test_diagnostic_masks(self):
        from hattention.fused_projected_coarse_router import fused_projected_coarse_core
        inputs=list(make_inputs(2,3,9,7,5,'cuda'))
        mass=torch.ones(2,3,device='cuda')
        _,_,flag=fused_projected_coarse_core(*inputs,mass,1e-6,torch.float32)
        _,_,bad,nonfinite=fused_projected_coarse_core(*inputs,mass,1e-6,torch.float32,return_masks=True)
        self.assertEqual(bad.shape,(2,3,9))
        self.assertFalse(nonfinite.any())
        torch.testing.assert_close(flag,bad.flatten(1).any(-1))
        with torch.no_grad():inputs[5].fill_(float('inf'))
        _,_,bad,nonfinite=fused_projected_coarse_core(*inputs,mass,1e-6,torch.float32,return_masks=True)
        self.assertTrue(nonfinite.all())
        self.assertTrue(bad.all())


if __name__=='__main__':unittest.main()
