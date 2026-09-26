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

The performance target is **not met**. In L40S job `23991312`, a matched
attention-layer forward/backward at batch 4, context 16,384, 12 heads, key
dimension 128 and value dimension 64 took **0.8263 s**, versus **0.4367 s**
for the previous `single_probe` implementation (`e8fcd9f`): about 89% slower.
Peak allocated GPU memory was **32.02 GB versus 13.75 GB** (decimal GB).
These are layer measurements, not complete training-step timings. H200 job
`23990722`, before joint state-gradient accumulation, measured **0.2164 s versus
0.07334 s**, with **32.56 GB versus 13.80 GB**. The final public implementation's
H200 measurement is pending; neither throughput parity nor the 20% slowdown
target has been established.

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
python tests/test_bilinear_model_integration.py
```

Run these inside the repository's CUDA/FLA runtime.

Validation on 2026-09-26: all 70 regression tests passed on L40S (job
`23992654`). A separate fresh-process compiled public-path check at 16K
compared outputs and all eight input gradients after RMSNorm with the explicit
precise backend; the largest relative L2 error per head was `7.86e-7`
(`23992465`). Cold and warm results matched. These checks establish agreement
on the tested inputs, not a prediction of final trained quality.
