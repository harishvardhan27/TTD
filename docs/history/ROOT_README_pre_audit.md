# Tensor Train Decomposition for Edge AI — Full Project

This repository documents a complete, evidence-driven investigation into
compressing neural network weight matrices using **Tensor Train (TT)
decomposition** as an alternative to INT8 quantization, tested across
transformers (DistilBERT-FFN, ViT), CNNs (ResNet, MobileNetV2), and
custom CUDA/C++ kernels — with honest reporting of what worked and what
didn't on real pretrained models.

---

## Repository Structure

```
MAIN PROJECT/
├── CNN/              # TTConv2d: TT compression for convolutional layers
├── DistilBERT/       # TTLinear: TT compression for transformer FFN layers
├── FNN/              # Real-model validation: MobileNetV2 + real ImageNet images
├── HO/               # Finalized scope: low-rank SVD beats TT on real models
├── THEORY_VERIFY/    # Theory proofs, Riemannian optimization, custom kernels
└── VIT/              # ViT-Base real-model test: compression + latency validation
```

---

## Submodule Summaries

### `CNN/` — TTConv2d
Drop-in `nn.Conv2d` replacement via TT decomposition. Convolution is
reduced to `im2col → TTLinear → fold`. Achieves 4.7–17x parameter
compression on synthetic edge-CNN (32→64→128→256 channels). Latency
caveat: `im2col + einsum` is slower than `nn.Conv2d`'s optimized path
on CPU — the size win is real, the latency win needs a fused kernel.

→ See [`CNN/README.md`](CNN/README.md)

---

### `DistilBERT/` — TTLinear (Transformer FFN)
Core TT-SVD implementation: `TTLinear` (drop-in `nn.Linear`), HOSVD
rank diagnostics, distillation fine-tuning. Achieves 6–21x compression
on synthetic DistilBERT-FFN-shaped weights (768↔3072, 4 blocks).
Same latency caveat as CNN — `torch.einsum` forward is slower than
dense matmul at these batch sizes.

→ See [`DistilBERT/README.md`](DistilBERT/README.md)

---

### `FNN/` — Real-Model Validation
Tests TT compression against **real pretrained ImageNet weights**
(MobileNetV2, ResNet34, ViT-Base) and **200 real ImageNet photos**.
Key finding: **plain low-rank (SVD) beat 3-mode TT at matched parameter
budgets on every real layer tested.** Includes a real bug caught and
fixed in automatic rank selection. Fine-tuning (distillation) on a
genuinely compressed real MobileNetV2 is validated end-to-end.

→ See [`FNN/README.md`](FNN/README.md) and [`FNN/REAL_MODEL_FINDINGS.md`](FNN/REAL_MODEL_FINDINGS.md)

---

### `HO/` — Finalized Project Scope
Evidence-based scope correction: after real-model testing, the project
pivots from TT to **plain low-rank (SVD) factorization of attention
projections and embeddings** — the only layers that showed real,
exploitable redundancy. Validated on ViT-Base with 100 real ImageNet
images: **1.2x compression, 2–3% accuracy drop, 12% latency improvement
— no fine-tuning, no custom kernel required.** First result in the
project where compression, accuracy, and latency all improved together.

→ See [`HO/FINALIZED_PROJECT.md`](HO/FINALIZED_PROJECT.md)

---

### `THEORY_VERIFY/` — Theory, Riemannian Optimization, Custom Kernels
The most complete submodule. Includes:
- **THEORY.md**: proofs of TT-SVD quasi-optimality, TT-rounding error
  bound, and end-to-end network output perturbation bound
- **Riemannian optimization**: manifold-constrained gradient descent
  (exact for `d=2`, rounding-retraction for general TT); Riemannian GD
  matches closed-form TT-SVD from a random init; plain Adam gets stuck
- **Custom CPU kernel** (`tt_linear_cpu.cpp`): hand-fused BLAS matmul
  chain, compiled and benchmarked — honest negative result: 1.3–1.9x
  *slower* than `torch.einsum` on CPU
- **Custom CUDA kernel** (`tt_linear_cuda.cu`): single-launch fused
  kernel using shared memory; index arithmetic verified via NumPy
  simulation (no GPU in this environment)

→ See [`THEORY_VERIFY/README.md`](THEORY_VERIFY/README.md)

---

### `VIT/` — ViT Real-Model Test
Applies low-rank attention compression to **ViT-Base** (real ImageNet
weights via `timm`) on 100 real ImageNet photos. Catches and fixes a
real preprocessing bug (wrong normalization for this model variant).
Validates the finalized scope: attention-layer SVD compression gives
real, positive results without fine-tuning or custom kernels.

→ See [`VIT/README.md`](VIT/README.md) and [`VIT/REAL_MODEL_FINDINGS.md`](VIT/REAL_MODEL_FINDINGS.md)

---

## Key Findings (Honest Summary)

| Claim | Verdict |
|---|---|
| TT-SVD correctly decomposes weight matrices | ✅ Verified (relative error ~1e-7 at generous rank) |
| TT compression reduces parameter count | ✅ 4.7–21x on synthetic weights |
| TT compression reduces latency | ❌ Reference `einsum` forward is *slower* than dense on CPU |
| TT beats plain SVD on real models | ❌ Plain SVD won on every real layer tested |
| Fine-tuning recovers accuracy after compression | ✅ Meaningful recovery in 60 distillation steps |
| Low-rank SVD on attention layers improves latency | ✅ 12% faster on ViT-Base CPU inference (no fine-tuning) |
| Riemannian GD finds the true manifold optimum | ✅ Matches closed-form TT-SVD from random init |
| Custom CPU kernel beats `torch.einsum` | ❌ 1.3–1.9x slower (reported as genuine negative result) |

---

## Quick Start

Each submodule is self-contained. Start with the one most relevant to
your use case:

```bash
# Synthetic transformer FFN demo (no internet needed)
cd DistilBERT
pip install -r requirements.txt
python demo_pipeline.py

# Synthetic CNN demo
cd CNN
python demo_cnn_pipeline.py

# Real-model validation (needs timm + real weights)
cd FNN
python real_model_eval.py

# Theory verification + Riemannian optimization
cd THEORY_VERIFY
python verify_theory.py
python riemannian_demo.py

# Custom CPU kernel benchmark (needs ninja)
cd THEORY_VERIFY
pip install ninja
python benchmark_kernel.py

# ViT real-model test
cd VIT
python real_vit_test.py
```

---

## Dependencies

Core: `torch`, `numpy`, `tensorly`
Real-model scripts: `timm` (weights fetched from GitHub releases)
HuggingFace scripts: `transformers`, `datasets`, `evaluate` + internet
Custom CPU kernel: `ninja` (`pip install ninja`)
CUDA kernel: GPU + CUDA toolkit (not required for anything else)

---

## Reference

Novikov et al., "Tensorizing Neural Networks", NeurIPS 2015
Oseledets, "Tensor-Train Decomposition", SIAM J. Sci. Comput. 2011
Steinlechner, "Riemannian Optimization for Fixed-Rank Matrix Approximation", SIAM 2016
