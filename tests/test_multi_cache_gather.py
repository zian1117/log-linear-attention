"""Grouped precise cache gathers preserve shared FP64 gradient cancellation."""
import importlib
import importlib.util
from pathlib import Path
import torch
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('shared_cache_test',ROOT/'tests/test_shared_precise_input_cache.py')
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)


def check_gather(device):
    fixture._oracle.load_helper()
    prototype=importlib.import_module('_precise_period_tests.multi_cache_gather')
    for dtype in (torch.float32,torch.float64):
        for scalar in (False,True):
            shape=(32,) if scalar else (32,7,11)
            raw=torch.randn(shape,device=device,dtype=dtype)
            indices=(torch.tensor([2,0,2,31],device=device),torch.empty(0,dtype=torch.long,device=device),
                     torch.tensor([[2,3],[3,7]],device=device),torch.tensor([0,3],device=device))
            for unused in (False,True):
                a=raw.clone().requires_grad_();e=raw.clone().requires_grad_()
                actual=prototype.gather_rows(a,indices)
                expected=tuple(e[idx] for idx in indices)
                chosen=[0,2]if unused else range(4)
                weights=[torch.randn_like(actual[i])for i in chosen]
                for x,y in zip(actual,expected):torch.testing.assert_close(x,y,rtol=0,atol=0)
                ga=torch.autograd.grad(tuple(actual[i]for i in chosen),a,grad_outputs=weights)[0]
                ge=torch.autograd.grad(tuple(expected[i]for i in chosen),e,grad_outputs=weights)[0]
                torch.testing.assert_close(ga,ge,rtol=3e-7 if dtype==torch.float32 else 1e-14,atol=3e-7 if dtype==torch.float32 else 1e-14)


def run(dtype,kind,device, *, prefix=False, unused_level=False):
    old=fixture._oracle.load_helper();package=old.__module__.rsplit('.',1)[0]
    preparation=importlib.import_module(package+'.precise_period_reads')
    helper=importlib.import_module(package+'.cached_chunk_reads')
    raw,history,reads,cases,generator=fixture.fixture(dtype,kind)
    raw=tuple(x.to(device)for x in raw);history=history.to(device);reads=reads.to(device)
    cases=[(l,s.to(device),w.to(device))for l,s,w in cases]
    histories = {level: None for level, _, _ in cases}
    if prefix:
        # Omit the first period's last chunk from the cache entirely while
        # preserving the original period and write half in the recurrence.
        level, selected, wanted = cases[-1]
        chunks, chunk = raw[1].shape[1:3]
        period = (1 << level) // chunk
        cutoff = period - 1
        position = torch.arange(chunks, device=device)
        wanted = wanted & (position % period < cutoff)[None, :]
        history = selected[:, position // period] & (position % period < cutoff)[None, :]
        reads = history & wanted & (position % period >= period // 2)[None, :]
        cases = [(level, selected, wanted)]
        histories = {level: cutoff}
    inputs=[tuple(x.clone().requires_grad_()for x in raw)for _ in range(3)]
    data=[fixture.normalize(x)for x in inputs]
    factors=[preparation.prepare_precise_chunk_cache(*p[1:5],history)for p in data[:2]]
    caches=[helper.prepare_precise_read_input_cache(p[0],p[1],p[3],p[4],p[5],p[6],reads,chunk_cache=factor)
            for p,factor in zip(data,factors)]
    prepared=helper.prepare_grouped_chunk_reads(data[0][1],
        [(level,selected,wanted,histories[level])for level,selected,wanted in cases],factors[0],caches[0])
    losses=[0.,0.,0.]
    for case_index, (level,selected,wanted) in enumerate(cases):
        if unused_level and case_index == 1:
            continue
        r,k,_,beta,gc,u,q,temp=data[0]
        got=helper.cached_chunk_reads(r,k,beta,gc,u,q,temp,level,selected,wanted,factors[0],
             read_input_cache=caches[0],prepared=prepared[level],history_chunks=histories[level])
        r,k,_,beta,gc,u,q,temp=data[1]
        previous=helper.cached_chunk_reads(r,k,beta,gc,u,q,temp,level,selected,wanted,factors[1],read_input_cache=caches[1],history_chunks=histories[level])
        exact=fixture._oracle.direct_reference(*(x.double()for x in data[2]),level,torch.ones_like(selected),1e-6)
        b,n,c=data[2][3].shape;period_chunks=(1<<level)//c
        expected=tuple(x.reshape(b,selected.shape[1]*period_chunks,c,*x.shape[2:])[:,:n].flatten(0,1).index_select(0,got[0])for x in exact[1:])
        assert torch.equal(got[0],previous[0])
        for a,e in zip(got[1:],previous[1:]):torch.testing.assert_close(a,e,rtol=0,atol=0)
        for a,e in zip(got[1:],expected):torch.testing.assert_close(a.double(),e,rtol=2e-6,atol=2e-8)
        weights=tuple(torch.randn(x.shape,dtype=x.dtype,generator=generator).to(device)for x in got[1:])
        for i,values in enumerate((got[1:],previous[1:],expected)):
            losses[i]+=sum((x*w).sum()for x,w in zip(values,weights))
    grads=[torch.autograd.grad(loss,raw)for loss,raw in zip(losses,inputs)]
    result={}
    for name,a,b,e in zip(('r','k','v','g','beta','u','q','log_temperature'),*grads):
        err=(a-e).norm().item();norm=e.norm().item()
        assert err<3e-7+1e-6*norm,(kind,name,err,norm)
        torch.testing.assert_close(a,b,rtol=3e-6,atol=3e-7)
        if name=='beta':torch.testing.assert_close(a,e,rtol=3e-6,atol=3e-7)
        result[name]={'relative':err/max(norm,1e-300),'maxabs':(a-e).abs().max().item(),
                      'versus_previous_maxabs':(a-b).abs().max().item()}
    if dtype==torch.float32 and kind=='collinear':assert result['beta']['maxabs']<1e-6,result['beta']
    return {'dtype':str(dtype),'kind':kind,'device':device,'gradients':result}

class TestMultiCacheGather(unittest.TestCase):
    def check_all(self, device):
        for dtype in (torch.float32, torch.float64):
            for kind in ('normal', 'collinear', 'large_decay_prefix', 'zero_beta'):
                with self.subTest(device=device, dtype=dtype, kind=kind):
                    run(dtype, kind, device)
        for kind in ('normal', 'collinear'):
            with self.subTest(device=device, kind=kind, prefix=True):
                run(torch.float32, kind, device, prefix=True)
        run(torch.float64, 'normal', device, unused_level=True)

    def test_cpu_duplicate_rows_and_unused_outputs(self):
        with torch._dynamo.config.patch(disable=True):
            check_gather('cpu')

    def test_cpu_all_raw_gradients_against_token_recurrence(self):
        with torch._dynamo.config.patch(disable=True):
            self.check_all('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_duplicate_rows_and_unused_outputs(self):
        check_gather('cuda')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_all_raw_gradients_against_token_recurrence(self):
        self.check_all('cuda')


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main(verbosity=2)
