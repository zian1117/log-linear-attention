"""Fused routing with recomputed normalization in backward, avoiding large FP32 saves."""
import torch
import triton
import triton.language as tl


@triton.jit
def _reduce(Y,C,U,S,O,DO,DY,DC,DU,DS,
            T:tl.constexpr,L:tl.constexpr,V:tl.constexpr,LB:tl.constexpr,
            BACKWARD:tl.constexpr,EPS:tl.constexpr):
    row=tl.program_id(0)
    bh=row//T
    t=row%T
    ll=tl.arange(0,LB)
    vv=tl.arange(0,V)
    offset=(row*L+ll[:,None])*V+vv[None,:]
    y=tl.load(Y+offset,ll[:,None]<L,0).to(tl.float32)
    c=tl.load(C+offset,ll[:,None]<L,0).to(tl.float32)
    u=tl.load(U+row*V+vv).to(tl.float32)
    cn2=tl.sum(c*c,1)
    un2=tl.sum(u*u,0)
    ci=tl.rsqrt(tl.maximum(cn2,EPS*EPS))
    ui=tl.rsqrt(tl.maximum(un2,EPS*EPS))
    ch=c*ci[:,None]
    uh=u*ui
    scale=tl.exp(tl.load(S+bh).to(tl.float32))
    cosine=tl.sum(ch*uh[None,:],1)
    logits=cosine*scale
    active=(ll<L)&((ll==0)|(((t>>tl.maximum(ll-1,0))&1)!=0))
    logits=tl.where(active,logits,-float('inf'))
    weight=tl.exp(logits-tl.max(logits,0))
    weight=weight/tl.sum(weight,0)
    if not BACKWARD:
        out=tl.sum(weight[:,None]*y,0)
        tl.store(O+row*V+vv,out)
    else:
        do=tl.load(DO+row*V+vv).to(tl.float32)
        da=tl.sum(y*do[None,:],1)
        dl=weight*(da-tl.sum(weight*da,0))
        dy=weight[:,None]*do[None,:]
        dc=scale*dl[:,None]*ci[:,None]*(uh[None,:]-tl.where(cn2[:,None]>EPS*EPS,ch*cosine[:,None],0.))
        duh=scale*tl.sum(dl[:,None]*ch,0)
        du=ui*(duh-tl.where(un2>EPS*EPS,uh*tl.sum(uh*duh,0),0.))
        ds=tl.sum(dl*cosine*scale,0)
        tl.store(DY+offset,dy,ll[:,None]<L)
        tl.store(DC+offset,dc,ll[:,None]<L)
        tl.store(DU+row*V+vv,du)
        tl.store(DS+row,ds)


class RoutingReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx,values,keys,query,temperature):
        values,keys,query,temperature=[x.contiguous() for x in (values,keys,query,temperature)]
        BH,N,C,L,V=values.shape
        out=torch.empty((BH,N,C,V),device=values.device,dtype=values.dtype)
        _reduce[(BH*N*C,)](values,keys,query,temperature,out,None,None,None,None,None,N*C,L,V,triton.next_power_of_2(L),False,1e-6,num_warps=4)
        ctx.save_for_backward(values,keys,query,temperature)
        return out

    @staticmethod
    def backward(ctx,do):
        values,keys,query,temperature=ctx.saved_tensors
        BH,N,C,L,V=values.shape
        dy,dc,du=[torch.empty_like(x) for x in (values,keys,query)]
        ds=torch.empty((BH,N,C),device=values.device,dtype=torch.float32)
        _reduce[(BH*N*C,)](values,keys,query,temperature,None,do.contiguous(),dy,dc,du,ds,N*C,L,V,triton.next_power_of_2(L),True,1e-6,num_warps=4)
        return dy,dc,du,ds.sum((1,2)).reshape_as(temperature).to(temperature.dtype)
