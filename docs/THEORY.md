# Theory Overview

This page summarises *what the method is, why it should work, and what is
not guaranteed*. Proofs: [`MATHEMATICAL_DERIVATIONS.md`](MATHEMATICAL_DERIVATIONS.md).
Costs: [`COMPLEXITY_ANALYSIS.md`](COMPLEXITY_ANALYSIS.md). Attribution:
[`LITERATURE_REVIEW.md`](LITERATURE_REVIEW.md). It supersedes
`THEORY_VERIFY/THEORY.md` (audit H5).

## 1. The pipeline

```
pretrained ViT ──► calibration (training-side images) ──► C_l = E[x xᵀ] per Linear l
        │                                                    │
        │                         ┌──────────────────────────┘
        ▼                         ▼
  W_l ──► factorise:  plain SVD(W_l)   or   whitened SVD(W_l L_l),  C_l + λμI = L_l L_lᵀ
        │
        ▼
  rank allocation under a parameter budget (greedy | exact DP | per-layer target | uniform)
        │
        ▼
  LowRankLinear(up ∘ down)  ──► optional dynamic INT8 (torch / ONNX Runtime)
        │                     ──► optional KD fine-tuning (factors + LayerNorms)
        ▼
  evaluation on ImageNetV2 (dev for selection, test for reporting) + CPU latency
```

| Component | Code | Guarantee | Not guaranteed |
|---|---|---|---|
| plain SVD | `lowrank.factorize(W)` | optimal weight error at each rank (Eckart–Young) | anything about outputs |
| whitened SVD | `lowrank.factorize(W, C)` | optimal *output* error on the calibration distribution for λ = 0; for λ > 0, optimal for output error + λμ·weight error (derivations §2.1) | generalisation to other inputs; effect of upstream compression on this layer's inputs |
| greedy allocation | `lowrank.allocate_ranks` | within one chunk's error of the optimum of the summed-relative-error objective (5.2) | that the objective predicts accuracy |
| DP allocation | `lowrank.allocate_ranks_dp` | exact for the discretised objective | same |
| INT8 | `quant.dynamic_int8`, `quant.ort_dynamic_int8` | weight rounding error ≤ Δ/2 per entry | that quantization and factorisation commute |
| KD | `distill.distill` | none (non-convex training) | recovery; measured on dev/test |

## 2. Why weight SVD should fail and whitened SVD should do better on ViTs

A ViT layer's inputs are LayerNorm outputs (qkv, fc1) or attention / GELU
outputs (proj, fc2). If their covariance is far from isotropic (a few
directions carry most of the energy — quantified for both models from the
calibration covariances in the analysis, "input anisotropy"), weight SVD spends rank
on directions of `W` that may receive almost no input energy; whitened SVD
weights directions by `C`, so it keeps what the layer actually transmits.
This predicts (i) a large gap between plain and whitened SVD at equal
parameters, (ii) a larger gap for layers with more anisotropic inputs, and
(iii) that weight-space "compressibility" (singular-value decay of `W`)
underestimates how compressible the layer is in function space. All three
are tested (A1, layer-coverage suite, per-layer error logs).

## 3. What the per-layer objective cannot tell us

Per-layer output errors are measured against the **uncompressed** network's
activations; the compressed network's later layers see perturbed inputs.
There is no useful end-to-end bound (derivations §3: the Lipschitz bound is
vacuous for ViTs). Consequently:

* rank allocation minimises a proxy, and the proxy's value is reported next
  to accuracy so the correlation can be judged;
* sequential re-calibration (compress layer l, then recollect C_{l+1} from
  the partially compressed model) is a known improvement (cf. SVD-LLM's
  sequential update) and is **not** implemented — a stated limitation.

## 4. Quantization interaction

Dynamic INT8 quantizes each factor separately, so a factorised layer suffers
two rounding errors plus the truncation error (derivations §6). Because
`U = P_r S_r` carries the singular values, column scales in `U` vary by the
singular-value spread; per-channel weight scales mitigate this. The order
"quantize then factorise" vs "factorise then quantize" is compared
empirically.

## 5. Hardware-aware rank selection (status: open)

A hardware-aware objective `L_rec + λ₁ L_lat + λ₂ L_mem` requires a latency
model per layer shape, rank and backend. On this CPU the MAC model predicts
speed-up ≈ MAC ratio for the compute-bound ViT GEMMs (complexity §4), so a
hardware term only changes the allocation if measured latency deviates from
MACs (e.g. poor efficiency at small ranks, or the extra INT8 quantize ops).
Phase 5 measures `t(m, n, r, backend)`; a hardware-aware allocator will be
implemented **only if** the measured deviation is large enough to change
allocations. Prior work (FLAR-SVD, CVPRW 2025) already does latency-aware
SVD for ViTs with a learned latency predictor.

## 6. Corrections to earlier project theory

* `THEORY_VERIFY/THEORY.md` §1 mixed two error definitions in the TT-SVD
  quasi-optimality proof; corrected in derivations §4.
* Its §3 Lipschitz propagation bound was presented as a rank-selection
  criterion; it is vacuous for ViTs (derivations §3).
* `riemannian_tt_fit` is projected gradient descent with a TT-SVD
  "retraction", not tangent-space Riemannian optimisation; on the
  reconstruction loss TT-SVD is already the quasi-optimal closed form, so
  matching it from a random start is expected and not evidence of benefit.
