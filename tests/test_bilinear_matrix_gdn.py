"""Independent full-matrix oracle for bilinear Fenwick GDN routing.

The oracle keeps each original write separately and propagates it through all
later erase/decay operations. It does not use the implementation's bucket
recurrence or chunked algebra. GPU tests compare every differentiable input.
"""
import math
import importlib
import importlib.util
from pathlib import Path
import sys
import types

import unittest
from unittest import mock
import torch
import torch.nn.functional as F


def full_matrix_reference(r, k, v, g, beta, u, q, log_temperature,
                          norm_floor=1e-6, vector_eps=1e-6,
                          gdn_norm_dtype=None):
    """Reference state orientation is [key_dimension, value_dimension]."""
    # Evaluate the oracle in its supplied precision; float64 is useful on CPU.
    recurrence_dtype = r.dtype
    if gdn_norm_dtype is not None:
        r, k = r.to(gdn_norm_dtype), k.to(gdn_norm_dtype)
    r = r * torch.rsqrt(r.square().sum(-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    r, k = r.to(recurrence_dtype), k.to(recurrence_dtype)
    u = F.normalize(u, dim=-1, eps=vector_eps)
    q = F.normalize(q, dim=-1, eps=vector_eps)
    batch, length, heads, key_dim = k.shape
    contributions = []
    output = []
    for t in range(length):
        kt, vt = k[:, t], v[:, t]
        bt, at = beta[:, t], g[:, t].exp()
        # Each past write receives exactly the same left linear operator.
        contributions = [
            at[..., None, None] * (
                state - bt[..., None, None] * kt[..., :, None]
                * torch.einsum('bhk,bhkv->bhv', kt, state)[..., None, :]
            ) for state in contributions
        ]
        contributions.append(bt[..., None, None] * kt[..., :, None] * vt[..., None, :])
        buckets = {}
        for j, state in enumerate(contributions):
            level = (t ^ j).bit_length()
            buckets[level] = state if level not in buckets else buckets[level] + state
        matrices = torch.stack([buckets[level] for level in sorted(buckets)], dim=2)
        norm = torch.linalg.vector_norm(matrices, dim=(-2, -1)).clamp_min(norm_floor)
        score = torch.einsum('bhv,bhlkv,bhk->bhl', u[:, t], matrices, q[:, t])
        score = score / norm * log_temperature.exp()[None, :, None]
        value = torch.einsum('bhlkv,bhk->bhlv', matrices, r[:, t]) / math.sqrt(key_dim)
        output.append((score.softmax(-1)[..., None] * value).sum(2))
    return torch.stack(output, dim=1)


def make_inputs(length=16, key_dim=8, value_dim=4, dtype=torch.float32,
                device='cpu', value_scale=1.0, batch=1, heads=2):
    generator = torch.Generator(device=device).manual_seed(1879)
    def randn(*shape):
        return torch.randn(*shape, generator=generator, device=device, dtype=dtype)
    key_shape = (batch, length, heads, key_dim)
    value_shape = (batch, length, heads, value_dim)
    gate_shape = (batch, length, heads)
    tensors = [
        randn(*key_shape), randn(*key_shape), value_scale * randn(*value_shape),
        -0.05 - 0.3 * torch.rand(*gate_shape, generator=generator, device=device, dtype=dtype),
        torch.sigmoid(randn(*gate_shape)), randn(*value_shape), randn(*key_shape),
        torch.full((heads,), math.log(math.sqrt(key_dim * value_dim)), device=device, dtype=dtype),
    ]
    return tuple(x.detach().requires_grad_() for x in tensors)


def _operator():
    from hattention.bilinear_matrix_gdn import bilinear_matrix_gdn
    return bilinear_matrix_gdn


def check_reference_uniform_routing_equals_full_gdn_divided_by_bucket_count():
    args = list(make_inputs(dtype=torch.float64))
    args[5] = torch.zeros_like(args[5])  # All active buckets get equal weight.
    actual = full_matrix_reference(*args)
    r, k, v, g, beta, _, _, _ = args
    r = r / (r.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    state = torch.zeros(k.shape[0], k.shape[2], k.shape[3], v.shape[3], dtype=k.dtype)
    expected = []
    for t in range(k.shape[1]):
        decayed = state * g[:, t].exp()[..., None, None]
        residual = v[:, t] - torch.einsum('bhk,bhkv->bhv', k[:, t], decayed)
        state = decayed + beta[:, t, :, None, None] * k[:, t, :, :, None] * residual[..., None, :]
        read = torch.einsum('bhk,bhkv->bhv', r[:, t], state) / math.sqrt(k.shape[-1])
        expected.append(read / (t.bit_count() + 1))
    torch.testing.assert_close(actual, torch.stack(expected, 1), rtol=1e-12, atol=1e-12)


def check_forward_and_all_gradients(length, key_dim, value_dim, value_scale):
    actual_inputs = make_inputs(length, key_dim, value_dim, device='cuda', value_scale=value_scale)
    reference_inputs = tuple(x.detach().double().requires_grad_() for x in actual_inputs)
    actual = _operator()(*actual_inputs)
    expected = full_matrix_reference(*reference_inputs)
    torch.testing.assert_close(actual.double(), expected, rtol=5e-4, atol=3e-6)
    upstream = torch.randn_like(actual)
    actual_grads = torch.autograd.grad((actual * upstream).sum(), actual_inputs)
    expected_grads = torch.autograd.grad((expected * upstream.double()).sum(), reference_inputs)
    for name, grad, reference_grad in zip(('r', 'k', 'v', 'g', 'beta', 'u', 'q', 'log_temperature'), actual_grads, expected_grads):
        assert torch.isfinite(grad).all(), f'{name} has nonfinite gradients'
        torch.testing.assert_close(grad.double(), reference_grad, rtol=2e-3, atol=2e-5, msg=lambda msg: f'{name}: {msg}')


def check_causality_and_separate_routing_query():
    args = make_inputs(length=73, key_dim=16, value_dim=8, device='cuda')
    actual = _operator()(*args)
    cut = 37
    altered = [x.detach().clone() for x in args]
    for x in altered[:-1]:
        x[:, cut:] = torch.randn_like(x[:, cut:])
    changed = _operator()(*altered)
    torch.testing.assert_close(actual[:, :cut], changed[:, :cut], rtol=0, atol=0)
    # A routing query change affects multi-bucket routing, but the first token
    # has one bucket and is independent of both routing projections.
    query_change = [x.detach().clone() for x in args]
    query_change[6].neg_()
    changed = _operator()(*query_change)
    torch.testing.assert_close(actual[:, :1], changed[:, :1], rtol=0, atol=0)
    assert (actual[:, 1:] - changed[:, 1:]).abs().max() > 1e-4


def check_zero_router_vectors_and_custom_floor():
    args = list(make_inputs(length=17, device='cuda', value_scale=1e-4))
    args[5] = torch.zeros_like(args[5], requires_grad=True)
    args[6] = torch.zeros_like(args[6], requires_grad=True)
    actual = _operator()(*args, norm_floor=1e-3, vector_eps=1e-4)
    expected = full_matrix_reference(*[x.double() for x in args], norm_floor=1e-3, vector_eps=1e-4)
    torch.testing.assert_close(actual.double(), expected, rtol=5e-4, atol=1e-8)
    for grad in torch.autograd.grad(actual.sum(), args):
        assert torch.isfinite(grad).all()


def check_bfloat16_forward_and_backward():
    args = make_inputs(length=65, key_dim=16, value_dim=8, dtype=torch.bfloat16, device='cuda')
    actual = _operator()(*args)
    expected = full_matrix_reference(*[x.float() for x in args])
    torch.testing.assert_close(actual.float(), expected, rtol=0.05, atol=0.008)
    for grad in torch.autograd.grad(actual.float().square().sum(), args):
        assert torch.isfinite(grad).all()


def check_collinear_keys_nearly_erased_buckets(dtype):
    args = list(make_inputs(length=128, key_dim=16, value_dim=8, dtype=dtype, device='cuda'))
    # All writes face the same erase direction. Older buckets become tiny
    # beside a nonzero current write, exposing norm/read cancellation errors.
    args[1] = args[1][:, :1].detach().expand_as(args[1]).clone().requires_grad_()
    args[3] = torch.full_like(args[3], -0.05, requires_grad=True)
    args[4] = torch.full_like(args[4], 0.999, requires_grad=True)
    reference_inputs = tuple(x.detach().double().requires_grad_() for x in args)
    actual = _operator()(*args)
    # Preserve GDN's FP32 normalization exactly in this cancellation stress.
    # With beta rounded to one, float64-normalized keys define a measurably
    # different near-erasure residual. Matrix propagation remains float64.
    expected = full_matrix_reference(*reference_inputs, gdn_norm_dtype=torch.float32)
    tol = 0.008 if dtype == torch.bfloat16 else 2e-5
    torch.testing.assert_close(actual.double(), expected, rtol=0.03 if dtype == torch.bfloat16 else 2e-3, atol=tol)
    upstream = torch.randn_like(actual)
    actual_grads = torch.autograd.grad((actual * upstream).sum(), args)
    expected_grads = torch.autograd.grad((expected * upstream.double()).sum(), reference_inputs)
    for name, grad, reference_grad in zip(('r', 'k', 'v', 'g', 'beta', 'u', 'q', 'log_temperature'), actual_grads, expected_grads):
        assert torch.isfinite(grad).all(), f'{name} has nonfinite gradients'
        relative_error = (grad.double() - reference_grad).norm() / reference_grad.norm().clamp_min(1e-8)
        limit = 0.1 if dtype == torch.bfloat16 else 0.01
        assert relative_error < limit, f'{name} relative gradient error: {relative_error.item()}'


def check_large_finite_logits(device):
    from hattention.bilinear_matrix_gdn import _RoutingReduce
    for scores in ((1e40, 1e40), (1e40, -1e40), (1e12, 1e12 + 1)):
        # Token zero has one bucket; token one has both. Include a tied max,
        # a saturated softmax, and a finite common offset with a small gap.
        values = torch.tensor([[[[[1., 2., 3.], [4., 5., 6.]],
                                  [[2., 3., 4.], [6., 7., 8.]]]]], device=device, requires_grad=True)
        logits = torch.tensor(scores, dtype=torch.float64, device=device).expand(1, 1, 2, 2).clone().requires_grad_()
        actual = _RoutingReduce.apply(values, logits, .5)
        reference_values = values.detach().double().requires_grad_()
        reference_logits = logits.detach().clone().requires_grad_()
        active = torch.tensor([[[[True, False], [True, True]]]], device=device)
        weights = reference_logits.masked_fill(~active, -torch.inf).softmax(-1)
        expected = .5 * (weights[..., None] * reference_values).sum(-2)
        torch.testing.assert_close(actual.double(), expected, rtol=1e-6, atol=1e-6)
        upstream = torch.tensor([[[[.2, -.7, .3], [.6, -.2, .8]]]], device=device)
        actual_grads = torch.autograd.grad((actual * upstream).sum(), (values, logits))
        expected_grads = torch.autograd.grad((expected * upstream.double()).sum(), (reference_values, reference_logits))
        for grad, reference_grad in zip(actual_grads, expected_grads):
            assert torch.isfinite(grad).all()
            torch.testing.assert_close(grad.double(), reference_grad, rtol=2e-6, atol=2e-7)


def check_numerical_boundary_inputs(case):
    args = list(make_inputs(length=73, key_dim=8, value_dim=4, device='cuda'))
    if case == 'zero_beta':
        args[4] = torch.zeros_like(args[4], requires_grad=True)
    elif case == 'exact_erasure':
        key = torch.zeros_like(args[1])
        key[..., 0] = 8.  # FP32 GDN normalization gives exactly a unit axis.
        args[1] = key.requires_grad_()
        args[4] = torch.ones_like(args[4], requires_grad=True)
    elif case == 'strong_decay':
        args[3] = torch.full_like(args[3], -100., requires_grad=True)
    elif case == 'tiny_routing_vectors':
        args[5] = (args[5].detach() * 1e-9).requires_grad_()
        args[6] = (args[6].detach() * 1e-9).requires_grad_()
    elif case == 'mixed_dtype':
        args = [x.detach().bfloat16().requires_grad_() if i in (0, 1, 2, 5, 6) else x for i, x in enumerate(args)]
    else:
        raise ValueError(case)
    reference_inputs = tuple(x.detach().double().requires_grad_() for x in args)
    actual = _operator()(*args)
    expected = full_matrix_reference(*reference_inputs, gdn_norm_dtype=torch.float32)
    mixed = case == 'mixed_dtype'
    torch.testing.assert_close(actual.double(), expected, rtol=.03 if mixed else 5e-4, atol=.003 if mixed else 3e-6)
    upstream = torch.randn_like(actual)
    actual_grads = torch.autograd.grad((actual * upstream).sum(), args)
    expected_grads = torch.autograd.grad((expected * upstream.double()).sum(), reference_inputs)
    for name, grad, reference_grad in zip(('r', 'k', 'v', 'g', 'beta', 'u', 'q', 'log_temperature'), actual_grads, expected_grads):
        assert torch.isfinite(grad).all(), f'{case}: {name} has nonfinite gradients'
        if mixed:
            error = (grad.double() - reference_grad).norm() / reference_grad.norm().clamp_min(1e-6)
            assert error < .05, f'{case}: {name} relative gradient error {error.item()}'
        else:
            torch.testing.assert_close(grad.double(), reference_grad, rtol=2e-3, atol=2e-5, msg=lambda msg: f'{case}, {name}: {msg}')


class TestCPUReference(unittest.TestCase):
    def setUp(self):
        # CPU tests verify arithmetic without triggering CPU compilation;
        # CUDA integration/benchmark jobs exercise the compiled implementation.
        self.enterContext(mock.patch.object(torch._dynamo.config, 'disable', True))

    def test_uniform_routing(self):
        check_reference_uniform_routing_equals_full_gdn_divided_by_bucket_count()

    def test_local_block_reads_match_dense_masked_reads_and_gradients(self):
        # Import only the operator and its pure-Torch helpers, avoiding model
        # package initialization and optional CUDA model dependencies on CPU.
        package_name = '_bilinear_cpu_arithmetic_tests'
        package = types.ModuleType(package_name)
        package.__path__ = [str(Path(__file__).resolve().parents[1] / 'hattention')]
        sys.modules.setdefault(package_name, package)
        operator = importlib.import_module(package_name + '.bilinear_matrix_gdn')
        generator = torch.Generator().manual_seed(389)
        batch_heads, chunks, chunk_size, key_dim, value_dim = 3, 2, 64, 7, 5
        key_shape = (batch_heads, chunks, chunk_size, key_dim)
        value_shape = (batch_heads, chunks, chunk_size, value_dim)
        scalar_shape = (batch_heads, chunks, chunk_size)
        def randn(shape):
            return torch.randn(shape, generator=generator, dtype=torch.float64)
        raw = (
            F.normalize(randn(key_shape), dim=-1),
            F.normalize(randn(key_shape), dim=-1),
            randn(value_shape),
            torch.sigmoid(randn(scalar_shape)),
            -.05 - .15 * torch.rand(scalar_shape, generator=generator, dtype=torch.float64),
            F.normalize(randn(value_shape), dim=-1),
            F.normalize(randn(key_shape), dim=-1),
            torch.tensor([1.1, 1.5, 2.3], dtype=torch.float64).reshape(batch_heads, 1, 1),
        )
        for level in range(chunk_size.bit_length()):
            with self.subTest(level=level):
                actual_inputs = tuple(x.clone().requires_grad_() for x in raw)
                reference_inputs = tuple(x.clone().requires_grad_() for x in raw)

                def evaluate(inputs, dense):
                    r, k, v, beta, g, u, q, log_temperature = inputs
                    gc = g.cumsum(-1)
                    *_, ar, aq, terms = operator._prepare(r, k, v, beta, gc, q)
                    temperature = log_temperature.exp()
                    if not dense:
                        return operator._local(ar, aq, k, v, beta, gc, u, temperature,
                                               level, terms, 1e-6, torch.float64)
                    # Original dense formulation: retain precisely the source
                    # columns with bit_length(query_position XOR source)==level.
                    mask = torch.tensor([
                        [j <= i and (i ^ j).bit_length() == level for j in range(chunk_size)]
                        for i in range(chunk_size)
                    ])
                    y = ar.masked_fill(~mask, 0.) @ v
                    read = aq.masked_fill(~mask, 0.) @ v
                    n2 = operator.local_bucket_norm2(k, v, beta, gc, level, terms=terms)
                    score = temperature * (read * u).sum(-1) / n2.clamp_min(1e-12).sqrt()
                    return y, score

                actual_y, actual_score = evaluate(actual_inputs, dense=False)
                expected_y, expected_score = evaluate(reference_inputs, dense=True)
                torch.testing.assert_close(actual_y, expected_y, rtol=1e-11, atol=1e-12)
                torch.testing.assert_close(actual_score, expected_score, rtol=1e-11, atol=1e-12)
                dy, ds = randn(value_shape), randn(scalar_shape)
                actual_grads = torch.autograd.grad((actual_y * dy).sum() + (actual_score * ds).sum(), actual_inputs)
                expected_grads = torch.autograd.grad((expected_y * dy).sum() + (expected_score * ds).sum(), reference_inputs)
                for name, grad, ref_grad in zip(('r', 'k', 'v', 'beta', 'g', 'u', 'q', 'log_temperature'), actual_grads, expected_grads):
                    torch.testing.assert_close(grad, ref_grad, rtol=1e-10, atol=1e-11, msg=lambda msg: f'{name}: {msg}')

    def test_boundary_states_and_all_gradients(self):
        path = Path(__file__).resolve().parents[1] / 'hattention' / 'bucket_states.py'
        spec = importlib.util.spec_from_file_location('tested_bucket_states', path)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        generator = torch.Generator().manual_seed(742)
        for chunks, period in ((5, 2), (7, 4), (3, 8), (9, 4), (7, 3)):
            with self.subTest(chunks=chunks, period=period):
                batch_heads, chunk_size, key_dim, value_dim = 3, 4, 3, 5
                shape = (batch_heads, chunks, chunk_size)
                raw = (
                    0.2 * torch.randn(*shape, key_dim, generator=generator, dtype=torch.float64),
                    0.2 * torch.randn(*shape, key_dim, generator=generator, dtype=torch.float64),
                    torch.randn(*shape, value_dim, generator=generator, dtype=torch.float64),
                    0.5 + 0.5 * torch.rand(batch_heads, chunks, generator=generator, dtype=torch.float64),
                )
                actual_inputs = tuple(x.clone().requires_grad_() for x in raw)
                reference_inputs = tuple(x.clone().requires_grad_() for x in raw)
                actual = helper.BucketStates.apply(*actual_inputs, period)
                keys, erase, writes, decay = reference_inputs
                state = keys.new_zeros((batch_heads, key_dim, value_dim))
                expected = []
                identity = torch.eye(key_dim, dtype=keys.dtype)
                for position in range(chunks):
                    if position % period == 0:
                        state = torch.zeros_like(state)
                    expected.append(state)
                    # Independent affine transition, with an explicit K×K
                    # matrix rather than the helper's residual/update form.
                    transition = (
                        decay[:, position, None, None] * identity
                        - keys[:, position].transpose(-1, -2) @ erase[:, position]
                    )
                    state = transition @ state
                    if position % period < period // 2:
                        state = state + keys[:, position].transpose(-1, -2) @ writes[:, position]
                expected = torch.stack(expected, dim=1)
                torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
                upstream = torch.randn(actual.shape, generator=generator, dtype=torch.float64)
                actual_grads = torch.autograd.grad((actual * upstream).sum(), actual_inputs)
                reference_grads = torch.autograd.grad((expected * upstream).sum(), reference_inputs)
                for name, grad, ref_grad in zip(('key', 'erase', 'writes', 'decay'), actual_grads, reference_grads):
                    torch.testing.assert_close(grad, ref_grad, rtol=1e-11, atol=1e-12, msg=lambda msg: f'{name}: {msg}')
                suppressed = torch.arange(chunks) % period >= period // 2
                self.assertTrue((actual_grads[2][:, suppressed] == 0).all())
                # The final chunk of each period has no dependent boundary
                # output before reset; it must receive no update gradient.
                last = torch.arange(chunks) % period == period - 1
                for grad in actual_grads:
                    self.assertTrue((grad[:, last] == 0).all())

    def test_gram_frobenius_against_matrix_recurrence(self):
        # Load this pure-Torch helper without importing optional model stacks.
        path = Path(__file__).resolve().parents[1] / 'hattention' / 'bucket_frobenius.py'
        spec = importlib.util.spec_from_file_location('tested_bucket_frobenius', path)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        torch.manual_seed(872)
        key = F.normalize(torch.randn(2, 16, 7, dtype=torch.float64), dim=-1).requires_grad_()
        value = torch.randn(2, 16, 5, dtype=torch.float64, requires_grad=True)
        beta = torch.rand(2, 16, dtype=torch.float64, requires_grad=True)
        decay = (-torch.rand(2, 16, dtype=torch.float64)).requires_grad_()
        gc = decay.cumsum(-1)
        for level in (0, 1, 2, 3, 4):
            with self.subTest(level=level):
                actual = helper.local_bucket_norm2(key, value, beta, gc, level)
                period = 1 << level
                state = torch.zeros(2, 7, 5, dtype=torch.float64)
                expected = []
                for t in range(16):
                    if t % period == 0:
                        state = torch.zeros_like(state)
                    state = decay[:, t, None, None].exp() * state
                    read = torch.einsum('bk,bkv->bv', key[:, t], state)
                    write = value[:, t] if level == 0 or t % period < period // 2 else torch.zeros_like(read)
                    state = state + beta[:, t, None, None] * key[:, t, :, None] * (write - read)[:, None, :]
                    expected.append(state.square().sum((-2, -1)))
                expected = torch.stack(expected, dim=-1)
                torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-11)
                inputs = (key, value, beta) if level == 0 else (key, value, beta, decay)
                a_grads = torch.autograd.grad(actual.sum(), inputs, retain_graph=True)
                e_grads = torch.autograd.grad(expected.sum(), inputs, retain_graph=True)
                for a, e in zip(a_grads, e_grads):
                    torch.testing.assert_close(a, e, rtol=1e-8, atol=1e-10)

        boundary = torch.randn(2, 7, 5, dtype=torch.float64, requires_grad=True)
        actual = helper.high_bucket_norm2(key, beta, gc, boundary)
        state = boundary
        expected = []
        for t in range(16):
            state = decay[:, t, None, None].exp() * state
            read = torch.einsum('bk,bkv->bv', key[:, t], state)
            state = state - beta[:, t, None, None] * key[:, t, :, None] * read[:, None, :]
            expected.append(state.square().sum((-2, -1)))
        expected = torch.stack(expected, -1)
        torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-11)
        inputs = (key, beta, decay, boundary)
        a_grads = torch.autograd.grad(actual.sum(), inputs, retain_graph=True)
        e_grads = torch.autograd.grad(expected.sum(), inputs)
        for a, e in zip(a_grads, e_grads):
            torch.testing.assert_close(a, e, rtol=1e-8, atol=1e-10)

    def test_floored_scores_after_severe_gram_cancellation(self):
        """Tiny norm errors must not become large routing-gradient errors."""
        path = Path(__file__).resolve().parents[1] / 'hattention' / 'bucket_frobenius.py'
        spec = importlib.util.spec_from_file_location('tested_bucket_frobenius_stress', path)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        generator = torch.Generator().manual_seed(946)
        batch, length, key_dim, value_dim = 2, 16, 7, 5
        key = F.normalize(torch.randn(batch, 1, key_dim, generator=generator, dtype=torch.float64), dim=-1)
        key = key.expand(batch, length, key_dim).clone()
        value = torch.randn(batch, length, value_dim, generator=generator, dtype=torch.float64)
        beta = torch.full((batch, length), 0.999, dtype=torch.float64)
        decay = torch.full((batch, length), -0.02, dtype=torch.float64)
        boundary = key[:, 0, :, None] * value[:, 0, None, :]
        query = F.normalize(torch.randn(batch, length, key_dim, generator=generator, dtype=torch.float64), dim=-1)
        output_query = F.normalize(torch.randn(batch, length, value_dim, generator=generator, dtype=torch.float64), dim=-1)
        upstream = torch.randn(batch, length, generator=generator, dtype=torch.float64)
        floor = 1e-6

        def direct_matrices(inputs, high):
            keys, values, betas, decays, initial = inputs
            current = initial if high else torch.zeros_like(initial)
            matrices = []
            identity = torch.eye(key_dim, dtype=keys.dtype)
            # Explicit transition matrices independently express the GDN rule;
            # this does not call the helper's recurrence or Gram expansion.
            for position in range(length):
                kt = keys[:, position]
                transition = decays[:, position, None, None].exp() * (
                    identity - betas[:, position, None, None] * kt[:, :, None] * kt[:, None, :]
                )
                current = transition @ current
                if not high and position < length // 2:
                    current = current + betas[:, position, None, None] * kt[:, :, None] * values[:, position, None, :]
                matrices.append(current)
            return torch.stack(matrices, dim=1)

        for high in (False, True):
            with self.subTest(high_bucket=high):
                actual_inputs = tuple(x.clone().requires_grad_() for x in (key, value, beta, decay, boundary))
                expected_inputs = tuple(x.clone().requires_grad_() for x in (key, value, beta, decay, boundary))
                ak, av, ab, ag, ast = actual_inputs
                actual_norm2 = (
                    helper.high_bucket_norm2(ak, ab, ag.cumsum(-1), ast) if high else
                    helper.local_bucket_norm2(ak, av, ab, ag.cumsum(-1), level=length.bit_length() - 1)
                )
                actual_matrices = direct_matrices(actual_inputs, high)
                expected_matrices = direct_matrices(expected_inputs, high)
                expected_norm2 = expected_matrices.square().sum((-2, -1))
                self.assertTrue((expected_norm2 < floor ** 2).any())
                self.assertTrue((expected_norm2 > floor ** 2).any())
                torch.testing.assert_close(actual_norm2, expected_norm2, rtol=1e-9, atol=1e-25)
                actual_numerator = torch.einsum('btv,btkv,btk->bt', output_query, actual_matrices, query)
                expected_numerator = torch.einsum('btv,btkv,btk->bt', output_query, expected_matrices, query)
                scale = math.sqrt(key_dim * value_dim)
                actual_score = scale * actual_numerator * actual_norm2.clamp_min(floor ** 2).rsqrt()
                expected_score = scale * expected_numerator * expected_norm2.clamp_min(floor ** 2).rsqrt()
                torch.testing.assert_close(actual_score, expected_score, rtol=1e-9, atol=1e-10)
                selected = (0, 2, 3, 4) if high else (0, 1, 2, 3)
                actual_grads = torch.autograd.grad((actual_score * upstream).sum(), tuple(actual_inputs[i] for i in selected))
                expected_grads = torch.autograd.grad((expected_score * upstream).sum(), tuple(expected_inputs[i] for i in selected))
                for name, actual_grad, expected_grad in zip((('key', 'beta', 'decay', 'boundary') if high else ('key', 'value', 'beta', 'decay')), actual_grads, expected_grads):
                    self.assertTrue(torch.isfinite(actual_grad).all(), name)
                    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-7, atol=2e-8, msg=lambda msg: f'{name}: {msg}')


@unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA')
class TestGPUOperator(unittest.TestCase):
    def test_large_finite_logits(self):
        check_large_finite_logits('cuda')

    def test_numerical_boundary_inputs(self):
        for case in ('zero_beta', 'exact_erasure', 'strong_decay', 'tiny_routing_vectors', 'mixed_dtype'):
            with self.subTest(case=case):
                check_numerical_boundary_inputs(case)

    def test_forward_and_all_gradients(self):
        for dimensions in ((16, 8, 4, 1.0), (73, 16, 8, 1.0),
                           (128, 16, 8, 1.0), (19, 13, 7, 1.0),
                           (129, 8, 4, 1.0),
                           (17, 8, 4, 1e-8),
                           (16, 8, 4, 0.0)):
            with self.subTest(dimensions=dimensions):
                check_forward_and_all_gradients(*dimensions)

    def test_causality_and_separate_routing_query(self):
        check_causality_and_separate_routing_query()

    def test_zero_router_vectors_and_custom_floor(self):
        check_zero_router_vectors_and_custom_floor()

    def test_bfloat16_forward_and_backward(self):
        check_bfloat16_forward_and_backward()

    def test_collinear_keys_nearly_erased_buckets(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                check_collinear_keys_nearly_erased_buckets(dtype)


if __name__ == '__main__':
    unittest.main(verbosity=2)
