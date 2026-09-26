"""Real public-router regressions for cancellation of memory amplitude gradients.

The loss includes gated RMSNorm. Gate derivatives are with respect to sigmoid
logits, not unconstrained beta. All fixtures are constructed here, without saved
checkpoints or patched repair decisions.
"""
import importlib
import json
import math
import unittest

import torch
import torch.nn.functional as F


INPUT_NAMES = ('r', 'k', 'v', 'g', 'beta_logits', 'u', 'q', 'log_temperature')
CASES = ('source_128', 'source_256', 'read_half_erase', 'cross_chunk_erase',
         'write_half_aligned', 'write_half_oblique', 'two_sources_erase')


def radial_fixture(case):
    generator = torch.Generator().manual_seed(492)
    length = 256 if case in ('source_256', 'cross_chunk_erase') else 128
    batch, key_dim, value_dim = 8, 128, 64
    source_beta = .003 if length == 256 else 3e-6
    decay = -.02 if length == 256 else -1e-5
    r = torch.randn(batch, length, 1, key_dim, generator=generator)
    k, q = torch.zeros_like(r), torch.zeros_like(r)
    k[..., 1] = 1
    k[:, 0, :, 1], k[:, 0, :, 0] = 0, 1
    q[..., 0], q[..., 1] = 1, 1
    v = torch.randn(batch, 1, 1, value_dim, generator=generator).expand(
        batch, length, 1, value_dim).clone() * .1
    u = v.clone()
    beta_logits = torch.zeros(batch, length, 1)
    beta_logits[:, 0] = math.log(source_beta / (1 - source_beta))
    active_beta = torch.zeros_like(beta_logits)
    active_beta[:, 0], active_beta[:, -1] = 1, 1
    gate = F.silu(torch.randn(batch, 1, value_dim, generator=generator))
    upstream = torch.randn(batch, 1, value_dim, generator=generator)
    tracked_gates = [0]
    if case not in ('source_128', 'source_256'):
        erase = 191 if case == 'cross_chunk_erase' else 126 if case == 'read_half_erase' else 31
        active_beta[:, erase], beta_logits[:, erase] = 1, 0
        k[:, erase] = ((k[:, 0] + k[:, -1]) / math.sqrt(2)
                       if case == 'write_half_oblique' else k[:, 0])
        v = v * 2
        v[:, erase] = 0
        tracked_gates.append(erase)
    if case == 'two_sources_erase':
        for tensor in (k, v, beta_logits, active_beta):
            tensor[:, 1] = tensor[:, 0]
        tracked_gates.append(1)
    return dict(r=r, k=k, v=v, g=torch.full_like(beta_logits, decay),
                beta_logits=beta_logits, active_beta=active_beta, u=u, q=q,
                log_temperature=torch.full((1,), math.log(math.sqrt(key_dim * value_dim))),
                gate=gate, upstream=upstream, tracked_gates=tracked_gates)


def measured_error(actual, expected):
    actual, expected = actual.double(), expected.double()
    return {'error_norm': (actual - expected).norm().item(),
            'reference_norm': expected.norm().item(),
            'max_error': (actual - expected).abs().max().item()}


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA public router required')
class TestRadialRoutingPrecision(unittest.TestCase):
    def test_public_outputs_and_all_parameter_input_gradients(self):
        # Import before changing Dynamo configuration so this test cannot disable
        # compilation permanently for another test's decorated functions.
        module = importlib.import_module('hattention.bilinear_matrix_gdn')
        configurations = [(case, torch.float32) for case in CASES]
        configurations += [(case, torch.bfloat16)
                           for case in ('source_128', 'two_sources_erase')]
        for case, vector_dtype in configurations:
            with self.subTest(case=case, vector_dtype=vector_dtype):
                fixture = radial_fixture(case)
                for name in ('r', 'k', 'v', 'u', 'q'):
                    fixture[name] = fixture[name].to(vector_dtype)
                results = []
                for function in (module.precise_bilinear_matrix_gdn,
                                 module.bilinear_matrix_gdn):
                    inputs = tuple(fixture[name].cuda().detach().requires_grad_()
                                   for name in INPUT_NAMES)
                    arguments = list(inputs)
                    arguments[4] = inputs[4].sigmoid() * fixture['active_beta'].cuda()
                    with torch._dynamo.config.patch(disable=True):
                        output = function(*arguments)
                        last = F.rms_norm(output[:, -1].float(),
                                          (output.shape[-1],), eps=1e-6)
                        loss = (last * fixture['gate'].cuda() * fixture['upstream'].cuda()).sum()
                        gradients = torch.autograd.grad(loss, inputs)
                    results.append((output.detach(), tuple(x.detach() for x in gradients)))
                expected, actual = results
                metrics = {name: measured_error(a, e) for name, a, e in
                           zip(INPUT_NAMES, actual[1], expected[1])}
                print('RADIAL_ROUTING_PRECISION', case, str(vector_dtype), json.dumps(metrics), flush=True)
                # BF16 vector storage rounds both the forward output and each
                # vector VJP. Two ulps near a binade boundary can differ by
                # about 1.6%; scalar gates remain FP32 and keep strict checks.
                bf16 = vector_dtype == torch.bfloat16
                torch.testing.assert_close(actual[0], expected[0],
                                           rtol=.016 if bf16 else 2e-5, atol=2e-8)
                for name, got, want in zip(INPUT_NAMES, actual[1], expected[1]):
                    self.assertTrue(torch.isfinite(got).all(), name)
                    metric = metrics[name]
                    # Aligned fixtures have analytically zero u/temperature
                    # gradients. FP32 softmax/read arithmetic can leave ~1e-5
                    # absolute residuals; a relative-only test is meaningless.
                    absolute = 2e-5 if name in ('u', 'log_temperature') else 2e-7
                    relative = .016 if got.dtype == torch.bfloat16 else 2e-5
                    self.assertLessEqual(metric['error_norm'],
                                         relative * metric['reference_norm'] + absolute,
                                         f'{case} {name}: {metric}')
                # A large current-token gate gradient must not hide the small
                # but meaningful historical gate gradients that originally failed.
                for token in fixture['tracked_gates']:
                    metric = measured_error(actual[1][4][:, token], expected[1][4][:, token])
                    self.assertLessEqual(metric['error_norm'],
                                         3e-5 * metric['reference_norm'] + 5e-10,
                                         f'{case} sigmoid gate {token}: {metric}')


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
