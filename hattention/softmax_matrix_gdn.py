"""Binary LLA-GDN with c_b = normalize(S_b p) and scalar softmax routing.

The two reads share each bucket's GDN state. Only chunk-boundary matrices are
materialized, one hierarchy level at a time; backward recomputes those states.
"""
import math
import torch
import triton
import triton.language as tl
from .matrix_routing_reduce import RoutingReduce
from fla.modules.l2norm import l2_norm


@triton.jit
def _state_fwd(KP, WP, UP, AP, HP,
               N: tl.constexpr, C: tl.constexpr, K: tl.constexpr,
               V: tl.constexpr, P: tl.constexpr):
    bh,seg=tl.program_id(0),tl.program_id(1)
    kk,vv,tt=tl.arange(0,K),tl.arange(0,V),tl.arange(0,C)
    state=tl.full((K,V),0,tl.float32)
    for i in range(P):
        n=seg*P+i
        if n<N:
            off=(bh*N+n)
            tl.store(HP+off*K*V+kk[:,None]*V+vv[None,:],state)
            if i<P-1:
                key=tl.load(KP+off*C*K+tt[:,None]*K+kk[None,:])
                w=tl.load(WP+off*C*K+tt[:,None]*K+kk[None,:])
                u=tl.load(UP+off*C*V+tt[:,None]*V+vv[None,:])
                if i>=P//2:
                    u=tl.full((C,V),0,tl.float32).to(u.dtype)
                residual=u.to(tl.float32)-tl.dot(w,state.to(w.dtype),input_precision='tf32x3')
                a=tl.load(AP+off)
                state=a*state+tl.dot(tl.trans(key),residual.to(key.dtype),input_precision='tf32x3')


@triton.jit
def _state_bwd(KP,WP,UP,AP,HP,DHP,DKP,DWP,DUP,DAP,
               N: tl.constexpr,C: tl.constexpr,K: tl.constexpr,
               V: tl.constexpr,P: tl.constexpr):
    bh,seg=tl.program_id(0),tl.program_id(1)
    kk,vv,tt=tl.arange(0,K),tl.arange(0,V),tl.arange(0,C)
    dh=tl.full((K,V),0,tl.float32)
    for reverse in range(P):
        i=P-1-reverse
        n=seg*P+i
        if n<N:
            off=bh*N+n
            key=tl.load(KP+off*C*K+tt[:,None]*K+kk[None,:])
            w=tl.load(WP+off*C*K+tt[:,None]*K+kk[None,:])
            u=tl.load(UP+off*C*V+tt[:,None]*V+vv[None,:])
            state=tl.load(HP+off*K*V+kk[:,None]*V+vv[None,:])
            a=tl.load(AP+off)
            write=i<P//2
            if not write:
                u=tl.full((C,V),0,tl.float32).to(u.dtype)
            residual=u.to(tl.float32)-tl.dot(w,state,input_precision='tf32x3')
            du=tl.dot(key,dh.to(key.dtype),input_precision='tf32x3')
            dk=tl.dot(residual.to(key.dtype),tl.trans(dh).to(key.dtype),input_precision='tf32x3')
            dw=-tl.dot(du.to(state.dtype),tl.trans(state),input_precision='tf32x3')
            da=tl.sum(tl.sum(state.to(tl.float32)*dh,1),0)
            tl.store(DKP+off*C*K+tt[:,None]*K+kk[None,:],dk)
            tl.store(DWP+off*C*K+tt[:,None]*K+kk[None,:],dw)
            tl.store(DUP+off*C*V+tt[:,None]*V+vv[None,:],tl.where(write,du,0.))
            tl.store(DAP+off,da)
            direct=tl.load(DHP+off*K*V+kk[:,None]*V+vv[None,:]).to(tl.float32)
            dh=a*dh-tl.dot(tl.trans(w),du.to(w.dtype),input_precision='tf32x3')+direct


def _states(k,w,u,a,period):
    BH,N,C,K=k.shape
    V=u.shape[-1]
    state=torch.empty((BH,N,K,V),device=k.device,dtype=k.dtype)
    _state_fwd[(BH,triton.cdiv(N,period))](k,w,u,a,state,N,C,K,V,period,num_warps=4)
    return state


class BucketReads(torch.autograd.Function):
    @staticmethod
    def forward(ctx,k,w,v,a,reads,period):
        k,w,v,a,reads=[x.contiguous() for x in (k,w,v,a,reads)]
        state=_states(k,w,v,a,period)
        result=reads@state
        ctx.save_for_backward(k,w,v,a,reads)
        ctx.period=period
        return result

    @staticmethod
    def backward(ctx,dr):
        k,w,v,a,reads=ctx.saved_tensors
        period=ctx.period
        state=_states(k,w,v,a,period)
        dreads=dr@state.transpose(-1,-2)
        dh=(reads.transpose(-1,-2)@dr).contiguous()
        dk,dw,dv,da=[torch.empty_like(x) for x in (k,w,v,a)]
        BH,N,C,K=k.shape
        V=v.shape[-1]
        _state_bwd[(BH,triton.cdiv(N,period))](k,w,v,a,state,dh,dk,dw,dv,da,N,C,K,V,period,num_warps=4)
        return dk,dw,dv,da,dreads,None


def softmax_matrix_gdn(q,k,v,g,beta,query,probe,log_temperature):
    B,T,H,K=k.shape
    V=v.shape[-1]
    C=64
    assert T % C == 0, 'Training/evaluation sequences must be padded to a multiple of 64.'
    N=T//C
    L=(T-1).bit_length()+1
    q,k=l2_norm(q),l2_norm(k)
    def chunks(x):
        return x.reshape(B,N,C,H,*x.shape[3:]).movedim(3,1).reshape(B*H,N,C,*x.shape[3:]).contiguous()
    q,k,v=map(chunks,(q,k,v))
    b=chunks(beta)
    gc=chunks(g).float().cumsum(-1)
    # The unit-diagonal triangular solve is local to a 64-token chunk.
    # Decay factors are applied after solving, avoiding exp(-g) overflow.
    with torch.autocast('cuda',enabled=False):
        kf=k.float()
        gram=kf@kf.transpose(-1,-2)
        eye=torch.eye(C,device=k.device,dtype=torch.float32)
        system=eye+(b.float().unsqueeze(-1)*gram).tril(-1)
        inv=torch.linalg.solve_triangular(system,eye.expand_as(system),upper=False,unitriangular=True)
    dt=gc.unsqueeze(-1)-gc.unsqueeze(-2)
    causal=torch.ones((C,C),device=k.device,dtype=torch.bool).tril()
    decay=dt.masked_fill(~causal,-torch.inf).exp()
    w=(inv.to(k.dtype)@(b.unsqueeze(-1)*k))
    u=((inv*decay).to(v.dtype)@(b.unsqueeze(-1)*v))
    qk=(q@k.transpose(-1,-2)).tril()
    probe=probe.to(k.dtype).unsqueeze(0).expand(B,-1,-1).reshape(B*H,1,1,K)
    pk=(probe@k.transpose(-1,-2)).expand(-1,-1,C,-1).tril()
    # Shared local linear operator for both the retrieval and routing reads.
    aq=((qk@inv.to(q.dtype)).float()*decay*b.unsqueeze(-2)).to(v.dtype)
    ap=((pk@inv.to(q.dtype)).float()*decay*b.unsqueeze(-2)).to(v.dtype)
    ge=gc.exp().unsqueeze(-1)
    qn=(ge*(q-qk@w)).to(q.dtype)
    pn=(ge*(probe-pk@w)).to(q.dtype)
    kn=(k*(gc[...,-1:]-gc).exp().unsqueeze(-1)).to(k.dtype)
    wn=(w*ge).to(k.dtype)
    end=gc[...,-1].exp().contiguous()
    pos=torch.arange(C,device=k.device)
    xor=pos[:,None]^pos[None,:]
    levels=torch.where(xor==0,0,torch.floor(torch.log2(xor.clamp_min(1).float())).long()+1)
    ys,cs=[],[]
    for level in range(min(7,L)):
        mask=(levels==level)&causal
        ys.append(aq.masked_fill(~mask,0.)@v)
        cs.append(ap.masked_fill(~mask,0.)@v)
    reads=torch.cat((qn,pn),dim=-2)
    for level in range(7,L):
        period=1<<(level-6)
        both=BucketReads.apply(kn,wn,u,end,reads,period)
        ys.append(both[...,:C,:]);cs.append(both[...,C:,:])
    values=torch.stack(ys,-2)
    keys=torch.stack(cs,-2)
    routed_query=chunks(query)
    temperature=log_temperature.float().unsqueeze(0).expand(B,-1).reshape(B*H,1,1,1)
    out=RoutingReduce.apply(values,keys,routed_query,temperature)
    return out.reshape(B,H,N,C,V).permute(0,2,3,1,4).reshape(B,T,H,V)
