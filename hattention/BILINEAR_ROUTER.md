# Bilinear routing over GDN matrix memories

Select `matrix_router=true` and `matrix_router_key="bilinear_frobenius"`.
The compatibility default `single_probe` still selects the earlier static-probe
router and preserves its checkpoint parameter names.

For each head, let `S_b` be the value-by-key GDN matrix for active Fenwick bucket
`b`. All subsequent tokens apply their GDN decay and erase operations to that
bucket. Three independent projections of the current token produce:

- `u`: a value-dimensional routing vector;
- `q`: a key-dimensional routing probe;
- `r`: the existing key-dimensional GDN read query.

Routing uses `u_hat = u / max(||u||_2, vector_eps)` and the same normalization
for `q`. The score and output are

```
score_b = exp(log_temperature) * u_hat.T @ S_b @ q_hat
          / max(||S_b||_F, norm_floor)
output = sum_b softmax(score)_b * S_b @ r_hat / sqrt(key_dim)
```

`r` and the GDN write keys retain the existing GDN normalization
`x / sqrt(sum(x*x) + 1e-6)`. The model's gated RMSNorm follows the output.
Both routing floors default to `1e-6` and are serialized in the configuration.
The routing projection weights use `initializer_range`; each head's learned
log-temperature starts at `log(sqrt(key_dim * value_dim))`.

Small buckets remain in softmax. Their small matrix reads are preserved; there
is no bucket mask based on matrix magnitude, additional lambda, or positional
encoding. Routing probes are independent of the value-read query.

The public `bilinear_matrix_gdn` uses the adaptive implementation; the equations
above are unchanged. `precise_bilinear_matrix_gdn` remains the FP64 reference.
The normal path uses FP32 matrix arithmetic, shared boundary states/projections,
and fused local reads. Triangular factors and energy identities compute bucket
norms without materializing a matrix at every token. Computational chunks have
64 tokens, independently of the model's key/value dimensions. Cumulative log
decays and their differences, exponentiated temperatures, and final scalar
scores retain FP64 precision.

Cancellation, proximity to the norm floor, and nonfinite results trigger
selective precise recomputation. A repaired read retains its bucket's complete
differentiable history; only the requested output chunks are evaluated.
Refined boundary scans check both forward and reverse residuals and use FP64
recurrence when those checks fail. Nonfinite heads are replaced completely,
with their discarded gradients blocked before the shared projections. These
guards are numerical diagnostics, **not a formal guarantee for every model
gradient**. They neither remove small buckets nor change the normalization.

Precise factor preparation, norms and reads share the same promoted FP64 input
tensors. This matters in backward: separately promoting an FP32 input can
round two large opposing gradient contributions before they are added. A
controlled erasure example gave a beta-gradient component of 159.125 instead
of 159.157114. Sharing the promotion permits cancellation in FP64 before one
conversion back to FP32. This fixes an arithmetic error without changing the
model equations; its effect on trained quality has not been measured.

GDN read/key normalization likewise promotes each BF16 input to FP32 once,
sharing that tensor between numerator and denominator. Separate promotions
would round their opposing gradients back to BF16 before addition. On an
almost-radial upstream-gradient regression, error against FP64 fell from
0.05103 to 0.000144 (reference gradient norm 0.07737); eager forward values are
bitwise unchanged. This is a controlled regression, not a measurement of
training-gradient error across the model.

Repair reads also share gathered inputs across hierarchy levels. State scans
stop after the last requested read, rounded up to a computational scheduling
quantum. Every requested read retains all preceding writes and erasures; the
original Fenwick periods and their write halves are unchanged. Backward also
reuses the guard's FP64 key-times-adjoint product, recomputing it for every
period whose adjoint the guard replaces; no precision check is relaxed.

The public boundary scan returns only the chunks in each Fenwick period's
active half. Backward reads those compact direct gradients and supplies zero
for the inactive positions internally, avoiding a dense gradient expansion
at each level. Every preceding write and erase still participates in the
state recurrence and receives its full gradient; complete boundary histories
remain saved for parameter-gradient computation. Precise cache rows are gathered
across repair levels together, so backward accumulates into one FP64 destination
per cache tensor. Duplicate indices and unused outputs remain supported. This
removes repeated destination allocation and summation; it does not change the
repair decisions or normalization.

Final softmax reduction reads each hierarchy level directly through a tuple
of tensor pointers. It avoids packing all reads into another tensor and
returns contiguous gradients for each level. Score centering remains FP64;
softmax and value accumulation remain FP32. The original stacked reducer is
retained for the precise reference and independent comparisons.

The performance target is **not met**. Matched attention-layer forward/backward
measurements use batch 4, context 16,384, 12 heads, key dimension 128 and value
dimension 64, versus the previous `single_probe` implementation (`e8fcd9f`):

| GPU / job | Bilinear layer | Reference layer | Ratio | Peak allocated memory, new / reference |
| --- | ---: | ---: | ---: | ---: |
| L40S / 23998204 | 0.7078 s | 0.4371 s | 1.62x | 30.11 / 13.75 GB |
| H200 / 23998100 | 0.1914 s | 0.07337 s | 2.61x | 30.13 / 13.80 GB |

These are warmed layer measurements with common parameters matched, using an
initialized first layer and saved validation tokens; they are not complete
training-step timings. Repair cost depends on the inputs. Neither throughput
parity nor the 20% slowdown target has been established. Direct per-level reduction lowered peak allocation by about 1 GB versus
the previous packed reducer; its whole-layer time reduction was modest
(about 1.2% on H200 and 1.8% on L40S in these separate measurements). Changing scan tile sizes did not improve the combined forward and
backward scan totals at the measured batch sizes on these GPUs. A separate
unsafe diagnostic disabled all repairs while retaining the current fast-path
computation and diagnostics: H200 job `23996870` measured 0.1365 s versus
0.07344 s. Thus removing repair cost alone would still miss the target in
this implementation. This diagnostic is not a supported model mode or a
mathematical lower bound on achievable runtime.

Full-sequence training and evaluation support nonmultiple lengths and unequal
key/value dimensions. Packed sequences and cached decoding are explicitly
unsupported. Tests compare outputs and every input gradient against an
independent per-write matrix recurrence, including zero/tiny matrices,
nearly complete erasure, partial buckets and unequal dimensions. Additional
checks cover shared-gradient accumulation, precise cached histories, changing
compiled batch sizes, and legacy checkpoint compatibility. Model tests cover
training, evaluation, and save/load:

```
python tests/test_bilinear_matrix_gdn.py
python tests/test_adaptive_matrix_gdn.py
python tests/test_cached_chunk_reads.py
python tests/test_multi_active_select.py
python tests/test_multi_projected_states.py
python tests/test_compact_projected_states.py
python tests/test_shared_precise_input_cache.py
python tests/test_tuple_routing_reduce.py
python tests/test_multi_cache_gather.py
python tests/test_grouped_cache_orchestration.py
python tests/test_normalization_precision.py
python tests/test_refined_projection_reuse.py
python tests/test_repair_prefix.py
python tests/test_bilinear_model_integration.py
```

Run these inside the repository's CUDA/FLA runtime.

Validation on 2026-09-26: all 81 regression tests passed on L40S (job
`23995148`). A separate fresh-process compiled public-path check at 16K
compared outputs and all eight input gradients after RMSNorm with the explicit
precise backend; the largest relative L2 error per head was `7.87e-7`.
Cold and warm results matched. H200 job `23995149` also passed the compiled
16K check, with largest relative error `6.52e-7`. Compact versus full boundary
outputs and all five factor gradients are bit-identical in 20 test cases;
an independent FP64 recurrence additionally checks the same results. The BF16 normalizer also passed a separate
fresh-process eager/compiled test against FP64 gradients. The larger regression
suite disables compilation for numerical isolation; it is not additional
compiled coverage. These checks establish agreement on the tested inputs,
not a prediction of final trained quality.

Grouped-cache validation: H200 `23996531` and L40S `23996532` passed the
new helper checks, shared-input/prefix/normalization regressions, and all model
lifecycle checks. Grouped versus separate cache gathers preserve every raw
input gradient against the independent recurrence, including cancellation,
partial histories and unused selections. Frozen-source compiled 16K checks
retain the above FP32 error bounds and identical cold/warm results.

An additional audit retained the captured native BF16 input/output dtypes,
using FP32 RMSNorm for the diagnostic loss. The largest per-head gradient
relative L2 difference against the precise backend was 0.000292 on H200 and
0.000322 on L40S, both in write-key gradients; output errors were below
0.000060. These are separate native-dtype measurements, not a relaxation of
the strict FP32 comparison. They do not establish trained-model quality.

Direct-reduction validation: H200 `23998100` and L40S `23998204` passed
GPU checks covering BF16/FP32 values, strided inputs/gradients, partial
hierarchies, padded feature dimensions, and extreme finite FP64 scores.
Outputs and every input gradient matched the retained stacked Triton reducer
bit for bit on those cases. Separate compiled 16K checks against the precise
backend passed with maximum reported relative gradient errors of 6.52e-7
and 8.40e-7, respectively; cold and warm results were identical. Three CPU
integration regressions also passed. Model lifecycle tests were unchanged
and passed on the preceding grouped-cache implementation.

A separate scratch experiment sharing affine tree summaries was accurate on
the captured input but slower than the existing state scan: H200 `23998100`
measured 69.46 ms versus 35.67 ms for compact state forward/backward, excluding
factor preparation. That tree is not used by production dispatch.

Remaining work:
- Reach the requested matched per-layer runtime target without weakening
  numerical guards or changing the routing equations.
- Continue checking numerical issues elsewhere in forward and backward,
  including cancellation before dtype conversions. Fix reproduced arithmetic
  errors without changing the conceptual model; retain independent references
  and regression cases. Initialization and stress-case checks do not certify
  every state encountered during a trained multi-layer trajectory.
