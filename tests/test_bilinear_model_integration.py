"""Small real-model train/eval/save/load checks for the bilinear router.

Run as a script in the same HAttention runtime as the operator tests. GPU tests
exercise two optimizer steps; CPU tests cover configuration and initialization.
"""
import json
import math
from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed.tensor

from hattention.configuration_h_gated_deltanet import HGatedDeltaNetConfig
from hattention.matrix_router_module import MatrixMemoryRouter
from hattention.modeling_h_gated_deltanet import HGatedDeltaNetForCausalLM


def small_config(**overrides):
    fields = dict(
        hidden_size=64, head_dim=16, num_heads=2, expand_v=1,
        num_hidden_layers=1, intermediate_size=128, vocab_size=128,
        initializer_range=0.006, matrix_router=True,
        matrix_router_key="bilinear_frobenius", use_cache=False,
        use_short_conv=True, use_gate=True, fuse_norm=False,
        fuse_swiglu=False, fuse_cross_entropy=False,
    )
    fields.update(overrides)
    return HGatedDeltaNetConfig(**fields)


class ConfigurationTests(unittest.TestCase):
    def test_invalid_config(self):
        for fields in (
            {"matrix_router_key": "unknown"},
            {"matrix_router_norm_floor": 0},
            {"matrix_router_norm_floor": float("inf")},
            {"matrix_router_vector_eps": float("nan")},
        ):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                small_config(**fields)

    def test_config_and_legacy_state(self):
        config = small_config(matrix_router_key="single_probe")
        legacy = HGatedDeltaNetForCausalLM(config)
        fields = config.to_dict()
        for key in ("matrix_router_key", "matrix_router_norm_floor", "matrix_router_vector_eps"):
            fields.pop(key)
        restored = HGatedDeltaNetForCausalLM(HGatedDeltaNetConfig.from_dict(fields))
        self.assertEqual(restored.model.layers[0].attn.router.key_mode, "single_probe")
        restored.load_state_dict(legacy.state_dict(), strict=True)
        for name, tensor in legacy.state_dict().items():
            self.assertTrue(torch.equal(tensor, restored.state_dict()[name]), name)
        router = restored.model.layers[0].attn.router
        self.assertEqual(set(router.state_dict()), {"probe", "query.weight", "log_temperature"})
        with tempfile.TemporaryDirectory() as directory:
            new = small_config(matrix_router_norm_floor=2e-5, matrix_router_vector_eps=3e-6)
            new.save_pretrained(directory)
            loaded = HGatedDeltaNetConfig.from_pretrained(directory)
        self.assertEqual(loaded.matrix_router_key, "bilinear_frobenius")
        self.assertEqual(loaded.matrix_router_norm_floor, 2e-5)
        self.assertEqual(loaded.matrix_router_vector_eps, 3e-6)

    def test_meta_materialization_and_initialization(self):
        with torch.device("meta"):
            model = HGatedDeltaNetForCausalLM(small_config())
        model.to_empty(device="cpu")
        model.apply(model._init_weights)
        for name, value in model.named_parameters():
            self.assertTrue(torch.isfinite(value).all(), name)
        attn = model.model.layers[0].attn
        router = attn.router
        self.assertNotEqual(router.query.weight.data_ptr(), router.probe_projection.weight.data_ptr())
        self.assertNotEqual(router.probe_projection.weight.data_ptr(), attn.q_proj.weight.data_ptr())
        for projection in (router.query, router.probe_projection):
            self.assertGreater(projection.weight.std().item(), 0.004)
            self.assertLess(projection.weight.std().item(), 0.008)
        expected = math.sqrt(attn.head_k_dim * attn.head_v_dim)
        torch.testing.assert_close(router.log_temperature.exp(), torch.full_like(router.log_temperature, expected))
        # All-zero projections are legal; their routing normalization must not
        # make initialization itself undefined.
        with torch.no_grad():
            router.query.weight.zero_()
            router.probe_projection.weight.zero_()
        self.assertTrue(torch.isfinite(router.log_temperature).all())

    def test_active_coarse_gather_matches_full_partial_periods(self):
        from hattention.bilinear_matrix_gdn import _coarse, _score
        from hattention.bucket_frobenius import _terms, high_bucket_norm2
        from hattention.bucket_states import BucketStates

        torch.manual_seed(411)
        for chunks, period in ((3, 2), (3, 4), (5, 2), (5, 4), (5, 8)):
            with self.subTest(chunks=chunks, period=period):
                shape = (1, chunks, 4)
                def random(*dimensions):
                    return (torch.randn(*dimensions, dtype=torch.float64) * .1).requires_grad_()
                kn, wn, rn, qn, k = [random(*shape, 3) for _ in range(5)]
                writes, u = [random(*shape, 2) for _ in range(2)]
                end = torch.full((1, chunks), .9, dtype=torch.float64, requires_grad=True)
                beta = torch.full(shape, .3, dtype=torch.float64, requires_grad=True)
                gc = (-torch.arange(1, 5, dtype=torch.float64).expand(shape) * .1).clone().requires_grad_()
                temperature = torch.full((1, 1, 1), 2., dtype=torch.float64, requires_grad=True)
                gram, inverse, decay = _terms(k, beta, gc)
                terms = (gram, inverse, decay, inverse @ (beta.unsqueeze(-1) * k))
                arguments = (kn, wn, writes, end, rn, qn, k, beta, gc, u, temperature)
                actual_y, actual_score = _coarse(*arguments, period, terms, 1e-6, torch.float64)
                state = BucketStates.apply(kn, wn, writes, end, period)
                norm2 = high_bucket_norm2(k, beta, gc, state, terms=terms)
                active = (torch.arange(chunks) % period >= period // 2).view(1, chunks, 1)
                expected_y = (rn @ state) * active.unsqueeze(-1)
                expected_score = _score(qn @ state, u, norm2, temperature, 1e-6) * active
                torch.testing.assert_close(actual_y, expected_y, rtol=1e-11, atol=1e-12)
                torch.testing.assert_close(actual_score, expected_score, rtol=1e-11, atol=1e-12)
                output_gradient = torch.randn_like(expected_y)
                score_gradient = torch.randn_like(expected_score)
                actual_grad = torch.autograd.grad(
                    (actual_y * output_gradient).sum() + (actual_score * score_gradient).sum(),
                    arguments, retain_graph=True,
                )
                expected_grad = torch.autograd.grad(
                    (expected_y * output_gradient).sum() + (expected_score * score_gradient).sum(),
                    arguments,
                )
                for actual, expected in zip(actual_grad, expected_grad):
                    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-11)

    def test_parameter_count_for_training_config(self):
        path = Path(__file__).resolve().parents[2] / "powerssm/configs/softmax_matrix_lla_gdn_130m.json"
        if not path.exists():
            self.skipTest("Parent training configuration unavailable")
        fields = json.loads(path.read_text())
        fields["matrix_router_key"] = "bilinear_frobenius"
        with torch.device("meta"):
            model = HGatedDeltaNetForCausalLM(HGatedDeltaNetConfig.from_dict(fields))
        count = sum(parameter.numel() for parameter in model.parameters())
        print(f"bilinear_frobenius training configuration parameters: {count:,}", flush=True)
        self.assertGreater(count, 0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for full model tests")
class ModelGpuTests(unittest.TestCase):
    def test_layer_checkpointing_preserves_parameter_gradients(self):
        import copy
        torch.manual_seed(417)
        reference = HGatedDeltaNetForCausalLM(small_config()).cuda().train()
        checkpointed = copy.deepcopy(reference)
        checkpointed.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        # Cross a computational chunk boundary and end in a partial chunk.
        tokens = torch.randint(0, reference.config.vocab_size, (1, 129), device="cuda")
        losses = []
        for model in (reference, checkpointed):
            loss = model(input_ids=tokens, labels=tokens, use_cache=False).loss
            loss.backward()
            losses.append(loss.detach())
        torch.testing.assert_close(losses[0], losses[1], rtol=1e-6, atol=1e-7)
        for (name, expected), (actual_name, actual) in zip(
                reference.named_parameters(), checkpointed.named_parameters()):
            self.assertEqual(name, actual_name)
            self.assertIsNotNone(actual.grad, name)
            self.assertTrue(torch.isfinite(actual.grad).all(), name)
            torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-5, atol=2e-7,
                                       msg=lambda msg: f"{name}: {msg}")

    def test_train_eval_checkpoint_and_separate_projections(self):
        torch.manual_seed(713)
        model = HGatedDeltaNetForCausalLM(small_config()).cuda().bfloat16()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        inputs = torch.randint(0, model.config.vocab_size, (1, 65), device="cuda")
        router = model.model.layers[0].attn.router
        tracked = (router.query.weight, router.probe_projection.weight,
                   model.model.layers[0].attn.q_proj.weight, router.log_temperature)
        before = [parameter.detach().clone() for parameter in tracked]
        model.train()
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            result = model(input_ids=inputs, labels=inputs, use_cache=False)
            self.assertTrue(torch.isfinite(result.loss))
            result.loss.backward()
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            for parameter in tracked:
                self.assertGreater(parameter.grad.float().norm().item(), 0.)
            optimizer.step()
        # Projection updates exceed BF16 spacing at their initialization scale.
        # The scalar log-temperature may round away a small individual update.
        self.assertTrue(all(not torch.equal(old, new) for old, new in zip(before[:3], tracked[:3])))
        model.eval()
        with torch.no_grad():
            expected = model(input_ids=inputs, labels=inputs, use_cache=False)
        self.assertTrue(torch.isfinite(expected.loss))
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            loaded = HGatedDeltaNetForCausalLM.from_pretrained(directory, torch_dtype=torch.bfloat16).cuda().eval()
            self.assertEqual(loaded.config.matrix_router_key, "bilinear_frobenius")
            with torch.no_grad():
                actual = loaded(input_ids=inputs, labels=inputs, use_cache=False)
        torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
        torch.testing.assert_close(actual.loss, expected.loss, rtol=0, atol=0)
        # Existing fused linear CE is a separate production training branch.
        model.config.fuse_cross_entropy = True
        model.train()
        optimizer.zero_grad(set_to_none=True)
        fused = model(input_ids=inputs, labels=inputs, use_cache=False)
        self.assertTrue(torch.isfinite(fused.loss))
        fused.loss.backward()
        self.assertTrue(torch.isfinite(router.probe_projection.weight.grad).all())

    def test_zero_routing_vectors_and_unsupported_packing(self):
        model = HGatedDeltaNetForCausalLM(small_config()).cuda()
        router = model.model.layers[0].attn.router
        with torch.no_grad():
            router.query.weight.zero_()
            router.probe_projection.weight.zero_()
        inputs = torch.randint(0, 128, (1, 16), device="cuda")
        result = model(input_ids=inputs, labels=inputs, use_cache=False)
        self.assertTrue(torch.isfinite(result.loss))
        result.loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())
        with self.assertRaisesRegex(NotImplementedError, "packed sequences"):
            model(input_ids=inputs, cu_seqlens=torch.tensor([0, 8, 16], device="cuda", dtype=torch.int32))
        with self.assertRaisesRegex(NotImplementedError, "use_cache=False"):
            model(input_ids=inputs, use_cache=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
