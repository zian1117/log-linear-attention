"""Private FP32 TF32x3 products for the shared projected-state tree.

Two kernels cover plain/accumulating products and diagonal/interleaved forward
epilogues. They preserve the validated 64x64x32 reduction order and disable
floating-point fusion outside the tensor-core products. No global precision
setting is changed. Autograd is owned by projected_state_tree._SharedTree;
these primitives intentionally do not implement independent CUDA backward.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _bmm4_epilogue(A,B,ADD,OUT,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
                  BATCH_N:tl.constexpr,
                  A0:tl.constexpr,A1:tl.constexpr,A2:tl.constexpr,A3:tl.constexpr,
                  B0:tl.constexpr,B1:tl.constexpr,B2:tl.constexpr,B3:tl.constexpr,
                  C0:tl.constexpr,C1:tl.constexpr,C2:tl.constexpr,C3:tl.constexpr,
                  O0:tl.constexpr,O1:tl.constexpr,O2:tl.constexpr,O3:tl.constexpr,
                  ALPHA:tl.constexpr,BETA:tl.constexpr,
                  BM:tl.constexpr=64,BN:tl.constexpr=64,BK:tl.constexpr=32):
    tile,batch=tl.program_id(0),tl.program_id(1)
    row=(tile//tl.cdiv(N,BN))*BM+tl.arange(0,BM)
    col=(tile%tl.cdiv(N,BN))*BN+tl.arange(0,BN)
    kk0=tl.arange(0,BK)
    bh,bb=batch//BATCH_N,batch%BATCH_N
    acc=tl.full((BM,BN),0,tl.float32)
    for block in range(tl.cdiv(K,BK)):
        kk=block*BK+kk0
        av=tl.load(A+bh*A0+bb*A1+row[:,None]*A2+kk[None,:]*A3,
                   (row[:,None]<M)&(kk[None,:]<K),0)
        bv=tl.load(B+bh*B0+bb*B1+kk[:,None]*B2+col[None,:]*B3,
                   (kk[:,None]<K)&(col[None,:]<N),0)
        acc=tl.dot(av,bv,acc,input_precision='tf32x3')
    result=acc*ALPHA
    valid=(row[:,None]<M)&(col[None,:]<N)
    if BETA!=0:
        add=tl.load(ADD+bh*C0+bb*C1+row[:,None]*C2+col[None,:]*C3,valid,0)
        result=result+BETA*add
    tl.store(OUT+bh*O0+bb*O1+row[:,None]*O2+col[None,:]*O3,result,valid)

@triton.jit
def _forward_epilogue(A,B,D,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
                      BATCH_N:tl.constexpr,
                      A0:tl.constexpr,A1:tl.constexpr,A2:tl.constexpr,A3:tl.constexpr,
                      B0:tl.constexpr,B1:tl.constexpr,B2:tl.constexpr,B3:tl.constexpr,
                      D0:tl.constexpr,D1:tl.constexpr,
                      INTERLEAVE:tl.constexpr,
                      BM:tl.constexpr=64,BN:tl.constexpr=64,BK:tl.constexpr=32):
    tile,batch=tl.program_id(0),tl.program_id(1)
    row=(tile//tl.cdiv(N,BN))*BM+tl.arange(0,BM)
    col=(tile%tl.cdiv(N,BN))*BN+tl.arange(0,BN)
    bh,bb=batch//BATCH_N,batch%BATCH_N
    inner=tl.arange(0,BK)
    acc=tl.full((BM,BN),0,tl.float32)
    for block in range(tl.cdiv(K,BK)):
        kk=block*BK+inner
        a=tl.load(A+bh*A0+bb*A1+row[:,None]*A2+kk[None,:]*A3,
                  (row[:,None]<M)&(kk[None,:]<K),0)
        b=tl.load(B+bh*B0+bb*B1+kk[:,None]*B2+col[None,:]*B3,
                  (kk[:,None]<K)&(col[None,:]<N),0)
        acc=tl.dot(a,b,acc,input_precision='tf32x3')
    valid=(row[:,None]<M)&(col[None,:]<N)
    offset=row[:,None]*N+col[None,:]
    if INTERLEAVE:
        left=tl.load(B+bh*B0+bb*B1+row[:,None]*B2+col[None,:]*B3,valid,0)
        tl.store(Y+(2*batch)*M*N+offset,left,valid)
        tl.store(Y+(2*batch+1)*M*N+offset,acc,valid)
    else:
        decay=tl.load(D+bh*D0+bb*D1)
        value=tl.where(row[:,None]==col[None,:],decay,0.)-acc
        tl.store(Y+batch*M*N+offset,value,valid)

def _validate(a,b):
    if a.ndim!=4 or b.ndim!=4:
        raise ValueError('Expected two [BH,N,rows,cols] tensors.')
    if a.shape[:2]!=b.shape[:2] or a.shape[-1]!=b.shape[-2]:
        raise ValueError('Batch axes must match exactly; inner dimensions must match.')
    if a.dtype!=torch.float32 or b.dtype!=torch.float32 or a.device!=b.device:
        raise ValueError('BMM inputs must be FP32 on the same device.')
    if min(a.shape[-2:])<=0 or b.shape[-1]<=0:
        raise ValueError('Matrix dimensions must be positive.')
    if any(s<=0 for x in (a,b) for s in x.stride()):
        raise ValueError('Only positive strides are supported; no expanded broadcasting.')

def bmm_add(a,b,addend,*,out=None,alpha=1.,beta=1.):
    _validate(a,b)
    shape=(*a.shape[:3],b.shape[-1])
    if addend.shape!=shape or addend.dtype!=a.dtype or addend.device!=a.device or any(s<=0 for s in addend.stride()):
        raise ValueError('Addend must have the exact FP32 output shape/device and positive strides.')
    if out is None:out=torch.empty(shape,device=a.device,dtype=a.dtype)
    elif out.shape!=shape or out.dtype!=a.dtype or out.device!=a.device or any(s<=0 for s in out.stride()):
        raise ValueError('Destination must have the exact FP32 output shape/device and positive strides.')
    if torch._C._overlaps(out,a) or torch._C._overlaps(out,b):
        raise ValueError('Destination must not overlap either matrix-product operand.')
    # Exact in-place addend alias is safe. Other overlap could race across tiles.
    if torch._C._overlaps(out,addend) and (out.data_ptr()!=addend.data_ptr() or out.stride()!=addend.stride()):
        raise ValueError('Overlapping addend/destination must be the same view.')
    if not a.is_cuda:
        product=alpha*(a@b)
        out.copy_(product if beta==0 else product+beta*addend)
        return out
    bh,nb,m,k=a.shape;n=b.shape[-1]
    if bh and nb:
        _bmm4_epilogue[(triton.cdiv(m,64)*triton.cdiv(n,64),bh*nb)](
            a,b,addend,out,m,n,k,nb,*a.stride(),*b.stride(),*addend.stride(),*out.stride(),
            float(alpha),float(beta),num_warps=4,num_stages=2,enable_fp_fusion=False)
    return out

def bmm_accumulate(a,b,out,*,alpha=1.,beta=1.):
    return bmm_add(a,b,out,out=out,alpha=alpha,beta=beta)

def bmm_difference(a,b,c,d):
    """Two products with only one destination: A@B minus C@D."""
    out=mm(a,b)
    return bmm_accumulate(c,d,out,alpha=-1.,beta=1.)

def diagonal_bmm(a,b,decay):
    """Return decay*I - A@B without either dense intermediate."""
    _validate(a,b)
    bh,nb,m,k=a.shape;n=b.shape[-1]
    if m!=n or decay.shape!=(bh,nb) or decay.dtype!=a.dtype or decay.device!=a.device or any(s<=0 for s in decay.stride()):
        raise ValueError('Diagonal epilogue requires square output and matching positive-strided decay.')
    if not a.is_cuda:return decay[...,None,None]*torch.eye(m,device=a.device,dtype=a.dtype)-a@b
    out=torch.empty((bh,nb,m,n),device=a.device,dtype=a.dtype)
    if bh and nb:
        _forward_epilogue[(triton.cdiv(m,64)*triton.cdiv(n,64),bh*nb)](
            a,b,decay,out,m,n,k,nb,*a.stride(),*b.stride(),*decay.stride(),False,
            num_warps=4,num_stages=2,enable_fp_fusion=False)
    return out

def interleaved_bmm(a,incoming):
    """Return [incoming_0,A_0@incoming_0,...] along the chunk dimension."""
    _validate(a,incoming)
    bh,nb,m,k=a.shape;n=incoming.shape[-1]
    if m!=k:raise ValueError('Interleaved transition must be square.')
    if not a.is_cuda:return torch.stack((incoming,a@incoming),dim=2).flatten(1,2)
    out=torch.empty((bh,2*nb,m,n),device=a.device,dtype=a.dtype)
    if bh and nb:
        _forward_epilogue[(triton.cdiv(m,64)*triton.cdiv(n,64),bh*nb)](
            a,incoming,incoming,out,m,n,k,nb,*a.stride(),*incoming.stride(),0,0,True,
            num_warps=4,num_stages=2,enable_fp_fusion=False)
    return out


def mm(a, b):
    """Plain product through the accumulating kernel with no addend load."""
    _validate(a, b)
    if not a.is_cuda:
        return a @ b
    out = torch.empty((*a.shape[:3], b.shape[-1]), device=a.device, dtype=a.dtype)
    # A fresh destination cannot alias either input. Avoid repeating the
    # shape/overlap checks needed by the general accumulating wrapper.
    bh, nb, m, k = a.shape
    n = b.shape[-1]
    if bh and nb:
        _bmm4_epilogue[(triton.cdiv(m,64)*triton.cdiv(n,64),bh*nb)](
            a,b,out,out,m,n,k,nb,*a.stride(),*b.stride(),*out.stride(),*out.stride(),
            1.,0.,num_warps=4,num_stages=2,enable_fp_fusion=False)
    return out
