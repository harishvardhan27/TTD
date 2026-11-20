# Complexity Analysis

All numbers in §1–2 were computed from the model definitions
(`ttcompress.bench.count_macs`, verified against the analytic count in
`tests/test_research.py::test_count_macs_includes_attention`). Nothing in
this document is a latency measurement; §4 states what latency *should* do
under explicit assumptions, and Phase 5 measures it.

## 1. Where the parameters and MACs are

| | DeiT-S | ViT-B |
|---|---|---|
| total parameters | 22,050,664 | 86,540,008 |
| embedding dim d / MLP dim | 384 / 1536 | 768 / 3072 |
| tokens N (196 patches + CLS) | 197 | 197 |
| `qkv` weights (12 layers) | 5,308,416 (24.1%) | 21,233,664 (24.5%) |
| `attn_out` weights (12) | 1,769,472 (8.0%) | 7,077,888 (8.2%) |
| `mlp` weights (24) | 14,155,776 (64.2%) | 56,623,104 (65.4%) |
| `attn+mlp` weights (48) | 21,233,664 (96.3%) | 84,934,656 (98.1%) |
| MACs / image (FP32 graph) | 4.599 G | 17.564 G |
| of which attention products QKᵀ, AV (12·2·N²·d) | 0.358 G (7.8%) | 0.715 G (4.1%) |

Because every Linear is applied to all N tokens, **linear-layer MACs =
N × linear weights**. Factorising the `attn+mlp` scope at budget b therefore
removes `(1 − b) · N · (#weights in scope)` MACs, e.g. DeiT-S at b = 0.6:
0.4 × 197 × 21.23 M = 1.673 G MACs (−36% of the model). The attention
products (≈4–8%) and the patch embedding are untouched.

Note: torch's FLOP counter silently omits fused `scaled_dot_product_attention`
on CPU; `count_macs` forces the math backend to include it (found in Phase 1).

## 2. Per-layer shapes and break-even ranks

A rank-r factorisation of an `m × n` layer costs `r(m+n)` parameters and
`N r (m+n)` MACs; it is smaller iff `r < mn/(m+n)`.

| layer | DeiT-S m×n | break-even r | ViT-B m×n | break-even r |
|---|---|---|---|---|
| qkv | 1152×384 | 287 | 2304×768 | 575 |
| attn proj | 384×384 | 191 | 768×768 | 383 |
| fc1 | 1536×384 | 307 | 3072×768 | 614 |
| fc2 | 384×1536 | 307 | 768×3072 | 614 |
| head | 1000×384 | 277 | 1000×768 | 434 |

With `min_saving = 0.1` (default), ranks are capped at 90% of the dense
parameter count, e.g. DeiT-S attn proj ≤ 168 (multiple of 8 ≤ 172.8).

## 3. Algorithmic cost of the compression itself

| step | cost | DeiT-S (48 layers) |
|---|---|---|
| covariance collection | one forward pass over n calibration images + `N n_in²` per layer | n = 256: ≈ 30 s on this CPU (measured in the dev run) |
| Cholesky + triangular inverse | `O(n_in³)` per layer, float64 | n_in ≤ 1536 |
| SVD of `W L` | `O(m n min(m,n))` per layer, float64 | ≤ 1536×384×384 |
| greedy allocation | `O(K log K)`, K = #chunks | negligible |
| DP allocation | `O(L · K_l · B/u)` | ≈ 48 × 40 × 4000 cells |
| TT baseline (bisection over ranks) | `O(log R)` TT-SVDs per layer | dominant build cost of B6 |

## 4. Expected runtime behaviour on the test CPU (assumptions, to be measured)

Ryzen 7 7730U (Zen 3): two 256-bit FMA pipes per core → 16 FP32 MAC/cycle/core
peak. AVX2 has no INT8 dot-product instruction on Zen 3 (no AVX-VNNI /
AVX-512-VNNI); INT8 GEMM uses `vpmaddubsw`/`vpmaddwd` sequences, so the INT8
speed-up over FP32 is bounded and kernel-dependent.

**Arithmetic intensity at batch 1.** A dense `m × n` layer on `N = 197`
tokens performs `2Nmn` FLOPs and moves at least `4(mn + Nn + Nm)` bytes
(FP32). For DeiT-S fc1 (1536×384): 232 MFLOP over 3.9 MB ⇒ ~60 FLOP/byte;
the factorised version at r = 128 has two GEMMs with `N r (m+n)` MACs and an
intermediate `N × r` activation, intensity still ≫ 10 FLOP/byte. ViT linear
layers on 197 tokens are therefore **compute-bound** even at batch 1 on this
CPU; MAC reduction should translate to time reduction *if* the GEMM kernels
are equally efficient at the reduced inner dimension r. Two effects work
against this and are measured in the microbenchmarks:

1. *Kernel efficiency at small r*: inner dimension r = 64–256 gives fewer
   FLOPs per loaded byte of the `N × r` intermediate and less work per thread
   for 8 threads;
2. *Extra operator*: two GEMMs + one more activation tensor write/read per
   layer; in ORT also one more `DynamicQuantizeLinear` per factor in INT8
   graphs (verified: 2 MatMulInteger per factorised layer).

Under a pure MAC model, latency speed-up ≈ total MACs before / after. For
DeiT-S `attn+mlp` at b = 0.6 that is 4.599 / 2.926 = 1.57×. Measured
speed-ups below this indicate kernel inefficiency; they are reported, not
corrected.

## 5. Memory

* Parameter memory = 4 bytes × params (FP32), ~1 byte × weights + scales (INT8).
* Serialized size is measured from the actual artifact (`state_dict` buffer
  or `.onnx` file), which includes non-parameter buffers and graph metadata.
* Peak RSS during inference includes the framework runtime, the input batch
  and activations (DeiT-S batch 8: 8 × 197 × 1536 × 4 B ≈ 9.7 MB per MLP
  hidden activation) and allocator caching; it is **not** proportional to
  parameter count and is measured, not derived.
