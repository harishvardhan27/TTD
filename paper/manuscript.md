# Activation-Whitened Low-Rank Factorization and INT8 Quantization of Vision Transformers on a Laptop CPU: A Controlled, Contamination-Checked Study

*Working title; to be finalised from the results (Phase 8).*
*Status markers: every number in this draft comes from a file under
`results/`; `[PENDING: …]` marks results not yet produced. No placeholder may
remain at submission.*

---

## Abstract

[PENDING — written last, from §9–§11 only.]

**Keywords:** Vision Transformer, low-rank factorization, activation-aware SVD,
post-training quantization, knowledge distillation, CPU inference, model compression.

---

## 1. Introduction

Vision Transformers (ViTs) are increasingly deployed on CPUs without
accelerators, where memory and latency, not accuracy alone, decide what is
usable. Two families of post-training compression are attractive in that
setting because they need little or no training data: low-rank factorization
of linear layers and integer quantization. Prior work has shown that
transformer weights are close to full rank while their activations are not
[yu2023compressing, chen2021drone], and that whitening the factorization
objective with input statistics makes truncation error track the true output
error [wang2025svdllm]. For ViTs, SVD has been reported to beat Tucker and
Tensor-Train decompositions [xiao2023comcat], activation-aware mixed-rank
schemes have been proposed [azizi2024memory], latency-aware SVD with a learned
latency predictor exists [thoma2025flarsvd], and rank and bit-width have been
optimised jointly [gordon2025mlorq].

This paper does **not** propose a new compression technique. It asks how
these established components behave *together* on a common laptop CPU, under
an evaluation protocol designed to make each conclusion falsifiable:

* **Q1** Does activation-whitened SVD beat weight SVD at matched parameter
  budgets on ViTs, by how much, and where in the network?
* **Q2** Which layer types can be factorised, and how should a parameter
  budget be spread across them?
* **Q3** Do factorization and INT8 dynamic quantization compose: in accuracy,
  in deployable size, and in measured CPU latency (PyTorch and ONNX Runtime)?
* **Q4** How much of the factorization loss can short CPU-only distillation recover?

**Contributions.** (i) A controlled comparison of plain SVD, whitened SVD,
Tensor-Train, INT8 and their combinations on DeiT-S and ViT-B, with paired
statistical tests on a held-out set checked for training-data overlap;
(ii) CPU measurements on an AVX2 laptop processor without VNNI, including
verification of which operators actually execute in INT8 in ONNX Runtime;
(iii) an audit of our own earlier evaluation, which found that commonly used
small ImageNet subsets (Imagenette, one-image-per-class samples) consist
largely of ImageNet-*train* images and silently inflate accuracy;
(iv) an open, test-covered implementation.

## 2. Motivation and Problem Definition

Given a pretrained ViT `f` with linear layers `{W_l}`, a parameter budget
`b ∈ (0,1]` (fraction of the targeted layers' weights retained) and an
optional INT8 constraint, find replacements `Ŵ_l = U_l V_l` that minimise the
accuracy loss of `f̂` relative to `f` on unseen images, and report the
resulting deployable size, MACs and measured latency on the target CPU. Only
training-side images may be used for calibration and distillation.

## 3. Related Work

See `docs/LITERATURE_REVIEW.md` for the verified table; summary paragraphs:
*low-rank compression* [eckart1936approximation, denton2014exploiting,
novikov2015tensorizing, oseledets2011tensor]; *data/activation-aware
factorization* [chen2021drone, hsu2022language, yuan2023asvd,
wang2025svdllm, yu2023compressing]; *ViT-specific* [xiao2023comcat,
azizi2024memory, thoma2025flarsvd]; *rank allocation*
[liebenwein2021compressing]; *quantization and hybrids* [jacob2018quantization,
yao2023zeroquantv2, gordon2025mlorq]; *distillation* [hinton2015distilling].

## 4. Mathematical Background

(From `docs/MATHEMATICAL_DERIVATIONS.md` §1–§2.) Truncated SVD and
Eckart–Young; whitened objective `‖(W−Ŵ)X‖_F` ⇒ SVD of `W L` with
`C = LLᵀ`; damping `C + λμI` as interpolation between output-error and
weight-error objectives (eq. 2.6).

## 5. Proposed Methodology (pipeline)

Figure 1 [PENDING: diagram] — calibration → whitened SVD → budgeted rank
allocation → LowRankLinear → optional INT8 → optional KD.

### 5.1 Activation-aware factorization
Algorithm 1:
```
input: W ∈ R^{m×n}, calibration inputs {x_t}, damping λ, rank r
C ← (1/N) Σ_t x_t x_tᵀ                    (float64, forward hooks)
C_λ ← C + λ · tr(C)/n · I
L ← chol(C_λ);  P S Qᵀ ← svd(W L)
U ← P[:, :r] · diag(S[:r]);   V ← Q[:, :r]ᵀ L⁻¹
return U (m×r), V (r×n)       # layer computes U (V x) + b
```

### 5.2 Global rank allocation
Algorithm 2 (greedy) and the exact multiple-choice-knapsack DP; guarantee
(5.2) in the derivations: greedy is within one chunk's normalised error of
the optimum of the summed relative-error objective.

### 5.3 Hybrid quantization
Dynamic INT8 on every `nn.Linear` including both factors (verified operator
coverage: two `MatMulInteger` per factorised layer in ORT; attention products
stay FP32).

### 5.4 Knowledge distillation
`L = α CE + (1−α) T² KL`, factors + LayerNorms trainable, teacher logits cached
(image and flip), checkpoint selection on ImageNetV2-dev.

## 6. Experimental Setup

**Models.** DeiT-S (`deit_small_patch16_224.fb_in1k`, 22.05 M params) and
ViT-B/16 (`vit_base_patch16_224_miil.in21k_ft_in1k`, 86.54 M); checkpoint
SHA-256 in every result file.

**Data.** ImageNetV2 matched-frequency [recht2019imagenet], split per class
into dev (2/class) and test (8/class). Calibration and distillation use only
Imagenette and a 1,000-image ImageNet sample (training-side). Automated
checks: 0 dev/test overlap, 0 byte-identical and 0 near-duplicate
eval/training images.

**Statistics.** Wilson intervals, paired Δ with 95% CI, exact McNemar with
Holm correction per question family; ±1 pp practical-equivalence margin.

## 7. Hardware Configuration

AMD Ryzen 7 7730U (Zen 3, 8C/16T, AVX2+FMA, no AVX-512/VNNI), 15.3 GB RAM,
Windows 11, PyTorch 2.11 CPU, ONNX Runtime 1.23 CPU EP, 8 intra-op threads.
Latency protocol: interleaved shuffled rounds, bootstrap CI of the median
speed-up, AC power required (recorded per run).

## 8. Results

### 8.1 Plain vs whitened SVD (Q1)
[PENDING: `results/tables/deit_small_v2_test_core.md`, rows `plain_attn+mlp_b*` / `data_attn+mlp_b*`; figure `results/figures/deit_small_v2_test_core_acc_vs_budget.png`]

### 8.2 Layer coverage and allocation (Q2)
[PENDING: core rows `data_{qkv,attn_out,attn,mlp,all}_b0.7`, `*_ub*`, `*_dp`; per-layer figure]

### 8.3 Tensor-Train baseline
[PENDING: core rows `tt_attn+mlp_b*`]

### 8.4 INT8 and hybrid compression (Q3)
[PENDING: core rows `int8`, `*_int8`, `*_int8_qf`, `ort_*`; ONNX op histograms]

### 8.5 Distillation (Q4)
[PENDING: `results/raw/deit_small/distill/…/result.json`, test rows `*__kd-*`]

### 8.6 CPU latency, memory and MACs
[PENDING: `results/raw/*/bench/*.json`, layer microbenchmarks]

## 9. Ablation Studies
[PENDING: A1–A12 from dev and core tables]

## 10. Statistical Analysis
[PENDING: paired tables; multiplicity]

## 11. Discussion
[PENDING]

## 12. Limitations
* CPU-only; no GPU or mobile-SoC measurements.
* ImageNetV2 has 10 images/class: per-class conclusions are not possible.
* Per-layer covariances come from the uncompressed model (no sequential re-calibration).
* Distillation transfer set is small (14k images) and class-imbalanced.
* Dynamic INT8 only; no static/PTQ calibration or mixed precision.

## 13. Threats to Validity
* *Internal:* single hardware platform and OS; thermal state uncontrolled beyond interleaving; one calibration seed for the main grid (seed variability measured on dev, A12).
* *Construct:* top-1 on ImageNetV2 is a proxy for deployment quality; agreement/KL are reported as fidelity measures.
* *External:* two ViT variants; results may not transfer to hierarchical ViTs (Swin) or LLMs.
* *Contamination:* ImageNet-21k vs ImageNetV2 overlap for the ViT-B checkpoint is unverified.

## 14. Conclusion
[PENDING]

## References
`paper/references.bib` (all entries verified 2026-11-20).
