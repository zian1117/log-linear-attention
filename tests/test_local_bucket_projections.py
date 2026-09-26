"""Independent forward/all-input-gradient checks for three shared projections."""
import unittest
import torch


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class LocalProjectionsGPU(unittest.TestCase):
    def test_outputs_and_gradients(self):
        from hattention.local_bucket_projections import local_bucket_projections
        torch.manual_seed(793)
        torch.backends.cuda.matmul.allow_tf32 = False
        for h in (1, 2, 4, 8, 16, 32):
            for v in (5, 64, 71):
                for strided in (False, True):
                    with self.subTest(h=h, v=v, strided=strided):
                        matrices = [torch.randn(2, 3, h, h, device='cuda') for _ in range(3)]
                        value = torch.randn(2, 3, h, v, device='cuda')
                        if strided:
                            matrices = [x.transpose(-1, -2) for x in matrices]
                            value = torch.randn(2, 3, v, h, device='cuda').transpose(-1, -2)
                        inputs = tuple(x.requires_grad_() for x in (*matrices, value))
                        expected = tuple(x @ inputs[-1] for x in inputs[:3])
                        upstream = tuple(torch.randn_like(x) for x in expected)
                        wanted = torch.autograd.grad(sum((x*g).sum() for x,g in zip(expected, upstream)), inputs)
                        got = local_bucket_projections(*inputs)
                        actual = torch.autograd.grad(sum((x*g).sum() for x,g in zip(got, upstream)), inputs)
                        for x,y in zip(got, expected):
                            torch.testing.assert_close(x,y,rtol=1e-5,atol=1e-5)
                        for x,y in zip(actual,wanted):
                            torch.testing.assert_close(x,y,rtol=2e-5,atol=2e-5)

    def test_unused_output(self):
        from hattention.local_bucket_projections import local_bucket_projections
        for h in (2, 16):
            inputs=tuple(x.requires_grad_() for x in (
                *(torch.randn(3,h,h,device='cuda') for _ in range(3)),
                torch.randn(3,h,5,device='cuda')))
            output=local_bucket_projections(*inputs)[0]
            grads=torch.autograd.grad(output.sum(),inputs)
            self.assertEqual(grads[1].count_nonzero().item(),0)
            self.assertEqual(grads[2].count_nonzero().item(),0)


if __name__ == '__main__':
    unittest.main()
