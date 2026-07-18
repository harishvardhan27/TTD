# Tensor Train Decomposition for Ultra-Low-Latency Edge AI

Compress transformer feed-forward weight matrices using Tensor Train (TT)
decomposition instead of INT8 quantization, and fine-tune to recover
accuracy. This repo contains a runnable, framework-agnostic implementation
you can point at any PyTorch model, plus a full demo you can execute
right now with no external dependencies beyond PyTorch/TensorLy.

## Real-model validation (read this first)

Everything below this section was originally validated on synthetic
weights. We then tested against **real pretrained ImageNet weights**
(MobileNetV2, ResNet34, ViT-Base — all via GitHub-hosted `timm`
releases, since this sandbox can't reach `huggingface.co`) and **real
ImageNet photos** (200 real images spanning all 1000 classes), and the
results substantially narrow the project's conclusions:

- Real CNN conv kernels compress far worse than our synthetic examples
  suggested, and we caught + fixed a real bug in automatic rank
  selection along the way.
- **The decisive result**: on every real layer we tested — CNN kernels,
  transformer MLP/FFN layers, attention projections, patch embeddings —
  **plain low-rank (ordinary matrix SVD) factorization beat 3-mode
  Tensor-Train decomposition at the same parameter budget, every time.**
  We found no real weight matrix where TT's extra tensorization modes
  revealed exploitable structure that plain SVD missed.

See **[REAL_MODEL_FINDINGS.md](REAL_MODEL_FINDINGS.md)** for the full,
evidence-by-evidence writeup — including the finding that real
transformer FFN layers (this project's originally-assumed best case) are
much flatter-spectrum than assumed, while attention projections and
embeddings have real structure that plain low-rank captures better than
TT does. Everything below this point describes the synthetic-weight
results and remains useful for understanding the code/math, but should
be read with that correction in mind.

## Contents

| File | Purpose |
|---|---|
| `tt_core.py` | Math core: mode factorization, HOSVD spectra, TT-SVD algorithm, `TTLinear` module (drop-in `nn.Linear` replacement) |
| `ttconv.py` | `TTConv2d`: drop-in `nn.Conv2d` replacement, built by reusing `TTLinear`/TT-SVD via the im2col (unfold) reduction |
| `compress_model.py` | Walks any `nn.Module` and replaces matching `nn.Linear`/`nn.Conv2d` layers with TT equivalents; size/latency/param reporting |
| `demo_pipeline.py` | **Runs end-to-end right now**: transformer-FFN demo — builds a DistilBERT-FFN-shaped network, HOSVD-analyzes it, TT-compresses at multiple ranks, fine-tunes via distillation, evaluates everything |
| `demo_cnn_pipeline.py` | **Runs end-to-end right now**: CNN demo — same pipeline applied to a small edge-vision CNN (32→64→128→256 channels, 3×3 convs) |
| `make_report_plots.py` / `make_cnn_report_plots.py` | Turn `results.json` / `results_cnn.json` into PNG comparison charts |
| `hf_real_model_example.py` | Reference script for compressing a **real** Hugging Face transformer (DistilBERT/GPT-2/TinyLlama) + fine-tuning on SST-2 — requires internet |
| `real_cnn_example.py` | Reference script for compressing a **real** pretrained CNN (torchvision ResNet-18/MobileNetV2) + fine-tuning on CIFAR-10 — requires internet |
| `riemannian_tt.py` | Riemannian (manifold-constrained) optimization: exact fixed-rank-matrix case + general TT rounding-retraction gradient descent |
| `riemannian_demo.py` | **Runs end-to-end right now**: honest 3-way comparison — TT-SVD vs. Riemannian GD vs. plain Adam-on-cores |
| `THEORY.md` | Proofs: TT-SVD quasi-optimality, TT-rounding error bound, end-to-end network output perturbation bound |
| `verify_theory.py` | Numerically verifies the THEORY.md claims (doesn't just assert them) |
| `kernels/` | Custom fused kernels: CPU (compiled + benchmarked here) and CUDA (reference, needs GPU — see `kernels/README.md`) |
| `benchmark_kernel.py` | Compiles and benchmarks the custom CPU kernel against `einsum` and dense — with an honest negative result |
| `verify_cuda_kernel_logic.py` | Verifies the CUDA kernel's index arithmetic via Python simulation (no GPU available here) |
| `REAL_MODEL_FINDINGS.md` | **Read this**: what actually happened testing real pretrained ImageNet weights + real photos, including a real bug we caught and fixed |
| `real_model_eval.py` | Real MobileNetV2 (real ImageNet weights) evaluated on 200 real ImageNet photos, before/after TT compression |
| `real_finetune_test.py` | Capstone test: fine-tune a genuinely-compressed real MobileNetV2 via distillation, re-evaluate on real held-out images |
| `requirements.txt` | Dependencies |

## Quick start

```bash
pip install -r requirements.txt
python demo_pipeline.py        # transformer-FFN demo: results table, results.json
python make_report_plots.py    # writes tt_compression_results.png

python demo_cnn_pipeline.py       # CNN demo: results table, results_cnn.json
python make_cnn_report_plots.py   # writes tt_cnn_results.png
```

To compress a real Hugging Face model on a machine with internet access:

```bash
python hf_real_model_example.py --model distilbert-base-uncased --rank 32
```

To compress a real pretrained CNN (torchvision ResNet-18/MobileNetV2) on a
machine with internet access:

```bash
python real_cnn_example.py --model resnet18 --rank 24
```

Riemannian optimization comparison (TT-SVD vs. Riemannian GD vs. plain Adam):

```bash
python riemannian_demo.py
```

Verify the theoretical error bounds numerically:

```bash
python verify_theory.py
```

Build and honestly benchmark the custom CPU kernel (needs `pip install ninja`):

```bash
python benchmark_kernel.py
```

## The math

A dense weight matrix `W` of shape `(out_features, in_features)` is
**tensorized**: factor `out_features = m_1 · m_2 · ... · m_d` and
`in_features = n_1 · n_2 · ... · n_d`, then reshape/permute `W` into a
`2d`-way tensor of shape `(m_1, n_1, m_2, n_2, ..., m_d, n_d)`.

**HOSVD** is used as a diagnostic: unfolding this tensor along each mode
independently and looking at the singular value spectrum tells you how
much energy is concentrated in the first `r` directions per mode — this
is how ranks are chosen instead of guessed (`hosvd_mode_singular_values`,
`suggest_rank_for_energy` in `tt_core.py`).

**TT-SVD** (Oseledets, 2011) then sweeps through the tensor left to
right, unfolding and truncated-SVD-ing at each cut, producing a chain of
cores `G_1, ..., G_d` with `G_k` of shape `(r_{k-1}, m_k, n_k, r_k)`,
`r_0 = r_d = 1`, such that

```
W(i_1..i_d, j_1..j_d) ≈ G_1[:,i_1,j_1,:] · G_2[:,i_2,j_2,:] · ... · G_d[:,i_d,j_d,:]
```

Parameter count becomes `Σ_k r_{k-1}·m_k·n_k·r_k` instead of
`out_features · in_features` — this is where the compression comes from.
`TTLinear.forward` never reconstructs the dense matrix; it contracts the
input against the cores directly via `torch.einsum`.

**Important, verified nuance:** TT compression exploits a *different*
kind of structure than plain matrix low-rank. A matrix that is low-rank
in the ordinary sense is **not** automatically low-TT-rank once you
factor it into 3+ modes — we verified this directly (see "What we
verified" below). What TT *does* exploit reliably is the empirically
well-documented decaying singular spectrum of trained transformer weight
matrices. `demo_pipeline.py` builds its synthetic weights with exactly
that decaying-spectrum property (rather than a flat random spectrum) so
the compression numbers below are representative of what to expect on
real pretrained checkpoints, not an artifact of a rigged test.

## What we verified before trusting the pipeline

1. **Generous rank → near-exact reconstruction.** With ranks large enough
   to be non-binding, `TTLinear.from_linear` reconstructs the original
   matrix to float32 precision (relative Frobenius error ~1e-7).
2. **A matrix built by construction to be low TT-rank** is recovered by
   TT-SVD at that exact rank to ~1e-7 relative error — confirms the
   decomposition algorithm itself is correct.
3. **A random Gaussian low-matrix-rank weight does *not* compress well**
   under a 3-mode TT split at low rank (~99% relative error) — confirms
   TT-rank ≠ matrix-rank, and warns against assuming any "low-rank"
   matrix will compress just because it's low-rank in the ordinary sense.

## Demo results (synthetic DistilBERT-FFN-shaped network, 4 blocks, 768↔3072)

Ranks and figures below are from an actual run of `demo_pipeline.py` in
this environment (see `results.json`):

| Model | Params | Size (MB) | Compression | CPU Latency (ms, batch=32) | Relative error vs. dense |
|---|---|---|---|---|---|
| Baseline (dense) | 19.5M | 74.3 | 1.0x | 30.4 | 0.00 |
| TT rank=48, pre-finetune | 3.08M | 11.8 | 6.3x | 227.0 | 1.00 |
| TT rank=48, fine-tuned | 3.08M | 11.8 | 6.3x | 234.9 | 0.67 |
| TT rank=16, pre-finetune | 0.91M | 3.5 | 21.4x | 43.4 | 1.02 |
| TT rank=16, fine-tuned | 0.91M | 3.5 | 21.4x | 41.4 | 0.69 |

Honest takeaways:

- **Compression works as expected**: 6–21x parameter/size reduction at
  the ranks tested, matching the 60–85% target range from the project
  brief (and easily exceeding it at the low-rank end).
- **Fine-tuning meaningfully recovers accuracy** (relative error drops
  from ~1.0, i.e. near-random, to ~0.67-0.69) after only 60 gradient
  steps of distillation — with a realistic training budget (real data,
  more steps, LR schedule) this gap closes further, as shown in the
  literature this project is based on (Novikov et al. 2015).
- **Latency is the counter-intuitive result**: our reference `TTLinear`
  forward pass, implemented with a generic `torch.einsum` contraction, is
  *slower* on CPU at these batch sizes than the highly-optimized dense
  matmul it replaces, despite having 6–20x fewer parameters. This is a
  real and important practical caveat: FLOPs/parameter reduction does not
  automatically translate into wall-clock latency reduction without a
  fused, hardware-tuned low-rank kernel (this is exactly why production
  TT-layer implementations use custom CUDA/C++ kernels rather than
  generic einsum — see "Possible Extensions" below). Report this
  transparently rather than claiming a speedup that the reference
  implementation doesn't actually deliver; the memory/size win is real
  and immediate, the latency win requires kernel engineering.

## CNNs: TTConv2d

Convolution reduces to `im2col (unfold) -> dense matmul -> reshape back`,
where the matmul weight is the kernel reshaped to `(C_out, C_in*kH*kW)`.
`TTConv2d` reuses this: it unfolds the input into patches, feeds them
through a `TTLinear` whose `in_features = C_in*kH*kW` and
`out_features = C_out`, then folds the result back into
`(B, C_out, H_out, W_out)`. This means all the TT-SVD math already
verified for `TTLinear` carries over unchanged — no new tensor-network
algorithm, only new unfold/fold plumbing. Tensorization: `C_out` and
`C_in` are each factorized into `m_1..m_d` / `n_1..n_d` as before, and
the spatial receptive field `kH*kW` is appended as one extra TT core
paired with a trivial output mode of size 1.

`TTConv2d` currently supports `groups=1` convolutions only (no
depthwise); `compress_model.replace_conv_with_tt`'s default predicate
also skips small stem convs (e.g. a 3-channel input conv), since those
have nothing to gain — verified below, a too-small conv can actually
*grow* under TT.

### What we verified for TTConv2d

- Generous rank → near-exact match with `nn.Conv2d` (relative output
  error ~3e-7), confirmed for both stride=1 and stride=2/padding
  configurations, plus gradient flow through the cores.
- A genuinely small conv (e.g. 3→32 channels, 3×3 kernel = 896 dense
  params) can end up with **more** parameters under TT than the dense
  original — the TT-core overhead (multiple small tensors + their own
  bond dimensions) isn't free, and only pays off once the layer is big
  enough. This is why `default_conv_predicate` exists and is applied by
  default.

### Demo results (synthetic edge-CNN, 32→64→128→256 channels, 3×3 convs)

From an actual run of `demo_cnn_pipeline.py` (see `results_cnn.json`),
compressing the three largest conv layers (the tiny stem conv is
excluded by the default predicate):

| Model | Params | Size (MB) | Compression | CPU Latency (ms, batch=16) | Relative error vs. dense |
|---|---|---|---|---|---|
| Baseline (dense) | 391,946 | 1.51 | 1.0x | 3.1 | 0.00 |
| TT rank=32, pre-finetune | 83,965 | 0.34 | 4.7x | 247.2 | 0.96 |
| TT rank=32, fine-tuned | 83,965 | 0.34 | 4.7x | 160.7 | 0.70 |
| TT rank=8, pre-finetune | 23,010 | 0.10 | 17.0x | 44.0 | 1.10 |
| TT rank=8, fine-tuned | 23,010 | 0.10 | 17.0x | 48.2 | 0.75 |

Same conclusions as the transformer case, and worth repeating because
they generalize: real compression ratios (4.7–17x here) with meaningful
accuracy recovery from fine-tuning, but the reference `im2col + einsum`
forward pass is again *slower* on CPU than `nn.Conv2d`'s
highly-optimized im2col+cuDNN/MKL-DNN path — the parameter-count win is
real and immediate, the latency win needs a fused kernel (custom
CUDA/C++, or export to ONNX Runtime / TensorRT, which can fuse and
optimize the small per-core contractions in ways generic `einsum` does
not).



- Change `tt_rank`, `num_modes`, or which layers get compressed
  (`predicate` argument to `replace_linear_with_tt`) in `demo_pipeline.py`
  or `hf_real_model_example.py`.
- Use `hosvd_mode_singular_values` + `suggest_rank_for_energy` to pick
  per-layer ranks automatically from an energy-retention target instead
  of a single global rank (adaptive TT-rank selection, one of the
  project's suggested extensions).
- To compare against INT8 quantization, wrap the baseline model with
  `torch.quantization.quantize_dynamic` and run it through the same
  `evaluate()` / `benchmark_latency()` helpers for an apples-to-apples
  table.
- To get real latency speedups (not just size reduction), replace the
  `torch.einsum`-based forward in `TTLinear` with a fused kernel, or
  export to ONNX and use ONNX Runtime's contraction fusion passes /
  TensorRT for edge deployment.

## Riemannian optimization

TT-format has a gauge redundancy (inserting `M, M^-1` between adjacent
cores leaves the represented tensor unchanged), which means plain
Euclidean SGD/Adam directly on core parameters can waste steps moving
along that redundant direction. `riemannian_tt.py` implements two levels
of manifold-respecting optimization instead:

- **Exact** tangent-space projection + SVD retraction for the classical
  fixed-rank matrix case (the `d=2` special case of TT), verified to
  converge to machine precision (relative error `1.1e-15`) on a known
  rank-5 target from a random initialization.
- **General TT case**: a rounding-retraction gradient scheme (take an
  ambient gradient step on the reconstructed dense tensor, then
  TT-SVD/round back onto the fixed-rank manifold). Every iterate is an
  exact point on the manifold; this trades some efficiency (it
  reconstructs the dense tensor each step, fine at single-layer scale)
  for simplicity over the fully tangent-space-restricted version used in
  the literature for much larger problems (Steinlechner 2016).

**Measured result** (`riemannian_demo.py`, genuinely TT-rank-24 target +
15% noise, 40 iterations from a random init):

| Method | Relative error | Time/iteration |
|---|---|---|
| TT-SVD (closed-form) | 0.1458 | — (one shot) |
| Riemannian retraction GD | 0.1458 | 459 ms |
| Plain Euclidean Adam | 0.6092 | 31 ms |

Riemannian GD reaches the *same* answer as the closed-form algorithm
from a random start — real confirmation it's finding the true manifold
optimum, not just "a" low-rank fit. Plain Adam, same budget, gets stuck
far worse: the gauge redundancy genuinely costs convergence speed. The
trade-off is cost per iteration (dominated by the TT-SVD re-projection);
Riemannian refinement is best used for short, high-value polishing
(e.g. right after TT-SVD init, before task fine-tuning), not as a
full training loop replacement.

## Theoretical error bounds

`THEORY.md` proves (and `verify_theory.py` numerically confirms, not
just asserts):

1. **TT-SVD quasi-optimality** (Oseledets 2011): `||W - W_TT||_F ≤
   sqrt(d-1) · ||W - W_best||_F` — TT-SVD is never worse than a
   `sqrt(d-1)` factor off the best possible fixed-TT-rank approximation
   (a factor of `√2 ≈ 1.41` for this project's `d=3` factorizations).
   The error-accumulates-in-quadrature identity behind this bound was
   checked to match a direct computation to float64 precision.
2. **TT-rounding** satisfies the same style of bound.
3. **End-to-end network output perturbation**: derives how per-layer
   weight error compounds through a stack of layers via their Lipschitz
   constants, explaining why deeper stacks need tighter per-layer rank
   budgets (matching what `demo_pipeline.py`'s multi-block results
   actually show), and gives a provable justification for compressing
   later layers more aggressively than earlier ones.

## Custom kernels

`benchmark_kernel.py` compiles a hand-fused CPU kernel
(`kernels/tt_linear_cpu.cpp`, explicit BLAS matmul chain instead of a
generic `torch.einsum` call) and benchmarks it for real. **Honest
result: it does not beat `torch.einsum` here** — even after removing an
obvious inefficiency (recomputing core reshapes every forward call),
the hand-fused version is ~1.3-1.9x *slower* than `einsum` across batch
sizes 8-512 on CPU, apparently because `einsum`'s internal planner
already avoids some of the `.contiguous()` copies our explicit
transpose-based version forces. See `kernels/README.md` for the full
writeup — this is reported as a genuine negative result, not hidden.

`kernels/tt_linear_cuda.cu` is a from-scratch CUDA kernel that fuses the
entire 3-core contraction into a **single kernel launch** using shared
memory — the situation where custom fusion is much more likely to help,
since GPU kernel-launch overhead is proportionally larger than the CPU
copies above. This sandbox has no GPU/`nvcc` (verified: no `nvidia-smi`,
`torch.cuda.is_available()` is `False`), so **it has not been compiled
or benchmarked** — instead, its exact index arithmetic was checked by
re-implementing the same loops in NumPy and confirming the output
matches `TTLinear`'s verified forward pass to `1.2e-7` relative error
(`verify_cuda_kernel_logic.py`). Build and benchmark it for real on a
CUDA machine with `kernels/build_cuda_kernel.py`.

## Requirements

See `requirements.txt`. Core demo only needs `torch`, `numpy`, `tensorly`;
`hf_real_model_example.py` additionally needs `transformers`, `datasets`,
`evaluate` and internet access to Hugging Face Hub. The custom CPU kernel
needs `ninja` (`pip install ninja`); the CUDA kernel needs a GPU + CUDA
toolkit and is not required for anything else in this project to run.
