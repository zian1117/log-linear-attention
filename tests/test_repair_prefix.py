"""Omitting states after the last read preserves outputs and prefix gradients."""
import importlib
import importlib.util
from pathlib import Path
import unittest
import torch
import torch.nn.functional as F

spec=importlib.util.spec_from_file_location('_prefix_oracle',Path(__file__).with_name('test_precise_period_reads.py'))
fixture=importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)

class TestRepairPrefix(unittest.TestCase):
    def compare(self,device):
        previous=fixture.load_helper()
        package=previous.__module__.rsplit('.',1)[0]
        factors=importlib.import_module(package+'.precise_period_reads')
        reads=importlib.import_module(package+'.cached_chunk_reads')
        generator=torch.Generator(device=device).manual_seed(82491)
        def randn(*shape):
            return torch.randn(*shape,generator=generator,dtype=torch.float64,device=device)
        for period,chunks,cutoff in ((64,150,64),(128,211,96),(256,320,160),(256,145,160)):
            with self.subTest(device=device,period=period,chunks=chunks):
                batch,chunk,key_dim,value_dim=3,2,5,7
                shape=(batch,chunks,chunk)
                raw=(F.normalize(randn(*shape,key_dim),dim=-1),
                     F.normalize(randn(*shape,key_dim),dim=-1),.1*randn(*shape,value_dim),
                     torch.full(shape,.7,dtype=torch.float64,device=device),
                     torch.full(shape,-.001,dtype=torch.float64,device=device).cumsum(-1),
                     F.normalize(randn(*shape,value_dim),dim=-1),
                     F.normalize(randn(*shape,key_dim),dim=-1),randn(batch,1,1).exp())
                groups=(chunks+period-1)//period
                selected=torch.ones(batch,groups,dtype=torch.bool,device=device)
                if groups>1:selected[1,0]=False
                requested=torch.zeros(batch,chunks,dtype=torch.bool,device=device)
                for head in range(batch):
                    for group in range(groups):
                        position=group*period+period//2+(head+group)*3
                        if position<chunks:requested[head,position]=True
                offsets=torch.arange(chunks,device=device)
                needed=selected[:,offsets//period]
                prefix_needed=needed&(offsets%period<cutoff)
                actual_inputs=tuple(x.clone().requires_grad_() for x in raw)
                expected_inputs=tuple(x.clone().requires_grad_() for x in raw)
                cache=factors.prepare_precise_chunk_cache(*actual_inputs[1:5],prefix_needed)
                reference_cache=factors.prepare_precise_chunk_cache(*expected_inputs[1:5],needed)
                level=(period*chunk).bit_length()-1
                actual=reads.cached_chunk_reads(actual_inputs[0],actual_inputs[1],*actual_inputs[3:],
                    level,selected,requested,cache,history_chunks=cutoff)
                expected=reads.cached_chunk_reads(expected_inputs[0],expected_inputs[1],*expected_inputs[3:],
                    level,selected,requested,reference_cache)
                self.assertTrue(torch.equal(actual[0],expected[0]))
                for a,e in zip(actual[1:],expected[1:]):
                    torch.testing.assert_close(a,e,rtol=1e-10,atol=1e-12)
                upstream=tuple(randn(*x.shape) for x in actual[1:])
                ag=torch.autograd.grad(sum((x*y).sum() for x,y in zip(actual[1:],upstream)),actual_inputs)
                eg=torch.autograd.grad(sum((x*y).sum() for x,y in zip(expected[1:],upstream)),expected_inputs)
                for a,e in zip(ag,eg):torch.testing.assert_close(a,e,rtol=1e-9,atol=1e-11)
                self.assertGreater(ag[2][0,:period//2].norm().item(),0.)
                for index in range(7):
                    self.assertEqual(torch.count_nonzero(ag[index][0,cutoff:min(period,chunks)]).item(),0)
                if cutoff<period:
                    with self.assertRaisesRegex(ValueError,'history|prefix'):
                        reads.cached_chunk_reads(actual_inputs[0],actual_inputs[1],*actual_inputs[3:],
                            level,selected,requested,cache,history_chunks=period//2)
    def test_cpu(self):
        self.compare('cpu')
    @unittest.skipUnless(torch.cuda.is_available(),'requires CUDA')
    def test_cuda(self):
        self.compare('cuda')

if __name__=='__main__':unittest.main(verbosity=2)
