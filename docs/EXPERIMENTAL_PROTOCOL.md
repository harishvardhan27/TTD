# Experimental Protocol

Written and committed to the tracker **before** any ImageNetV2 result was
inspected (2026-11-20, 13:50). Changes after that time are listed in §9 with
the reason.

## 1. Hardware and software

| Component | Value |
|---|---|
| CPU | AMD Ryzen 7 7730U (Zen 3 "Barcelo"), 8 cores / 16 threads, AVX2 + FMA, **no AVX-512, no AVX-VNNI** |
| Memory | 15.3 GB |
| GPU | integrated Radeon; **not used** (no CUDA; no GPU result is claimed) |
| OS | Windows 11 |
| Software | Python 3.10.0, PyTorch 2.11.0+cpu, timm 1.0.31, ONNX 1.23.2, ONNX Runtime 1.23.2 (CPU EP) |
| Threads | `torch.set_num_threads(8)` (physical cores) for all evaluation and timing; ORT `intra_op_num_threads=8` |

Every result JSON embeds the environment, git commit (+ dirty flag), config
SHA-1, checkpoint SHA-256 and the evaluation split's manifest SHA-256.

## 2. Models

| Role | timm identifier (pinned tag) | Params | Preprocessing (timm data config) |
|---|---|---|---|
| primary | `deit_small_patch16_224.fb_in1k` | 22,050,664 | 224, crop 0.875, bicubic, ImageNet mean/std |
| confirmatory | `vit_base_patch16_224_miil.in21k_ft_in1k` | 86,540,008 | 224, crop 0.875, bilinear, mean 0 / std 1 |

Each model is always evaluated with **its own** timm data config; within a
model every variant sees bit-identical input tensors (one uint8 cache).

## 3. Data and roles

| Split | Source | Size | Role |
|---|---|---|---|
| `v2_dev` | ImageNetV2 matched-frequency, 2 images/class (seeded per-class permutation) | 2,000 | hyperparameter and checkpoint selection **only** |
| `v2_test` | ImageNetV2 matched-frequency, remaining 8/class | 8,000 | final reporting **only** |
| `transfer` | Imagenette2-320 train+val (13,394) + ImageNet sample (1,000) | 14,394 | calibration, distillation |

Rationale and checks: Imagenette and the sample set are ImageNet-train
provenance (RESEARCH_AUDIT C1), so they are used only on the training side.
Manifests with per-file SHA-1 are in `data/manifests/`. Automated checks
(2026-11-20): V2 dev∩test = 0; byte-identical V2/training files = 0; same-class
near-duplicates (16×16 fingerprint correlation ≥ 0.98) = 0
(`data/manifests/overlap_v2_vs_transfer.json`). `get_items(..., role=...)`
raises if an eval split is requested for training or vice versa.

ImageNetV2 is a *harder* distribution than ImageNet-val (Recht et al. report
an 11–14 pp drop for most models); absolute accuracies are therefore lower
than published ImageNet numbers and must not be compared to them.

Calibration: `n` images drawn from `transfer` with a seeded permutation; the
next `heldout_n` = 128 images of the same permutation form the held-out set
used only to *report* per-layer output errors.

## 4. Terminology

* **budget b** = fraction of the *targeted layers'* weight parameters kept
  (b = 0.6 keeps 60%). Results also report whole-model parameter ratio,
  serialized size ratio, MACs and measured latency, which are different
  quantities and are never substituted for each other.
* **scope** = which nn.Linear layers are targeted: `qkv`, `attn_out`, `attn`,
  `mlp`, `attn+mlp`, `all` (adds the classifier head).
* **plain** = weight SVD; **data** = activation-whitened SVD.

## 5. Pre-registered selection rules (fixed before seeing dev results)

Operating point for all hyperparameter studies: `data`, `attn+mlp`, b = 0.6,
greedy allocation, DeiT-S, `v2_dev`.

1. **Damping λ** ∈ {0, 1e-4, 1e-3, 1e-2, 1e-1}: choose the λ with the highest
   dev **top-1 agreement with the FP32 baseline**; ties (within 0.25 pp, i.e.
   5 images) go to the larger λ (more conservative). Agreement is used rather
   than accuracy because it has lower variance and measures fidelity, which is
   what compression should preserve.
2. **Calibration size n** ∈ {32, 64, 128, 256, 512}: smallest n whose dev
   agreement is within 0.25 pp of the best n.
3. **Calibration source** and **seed** are *reported*, not selected; the
   default (`transfer`, seed 1) is kept regardless.
4. **min_saving** ∈ {0, 0.1, 0.25}: reported; the default 0.1 is kept unless
   0 is better by > 0.5 pp agreement.
5. **Distillation** (α, T, lr, steps): chosen by dev top-1 accuracy of the
   student (§7); test is evaluated once per selected configuration.

The same λ and n are used for ViT-B (no separate tuning).

## 6. Metrics and statistics

Per variant, on the evaluation split (never pooled across datasets):

* top-1 and top-5 accuracy; **Wilson 95% interval** for top-1;
* paired vs FP32 baseline on the same images: Δtop-1 with 95% CI (paired
  normal approximation), **exact two-sided McNemar** p-value, top-1
  agreement, mean KL(baseline ‖ variant);
* per-class shift distribution (number of classes losing ≥2 images, …);
  with 8 images/class per-class accuracies are never interpreted individually;
* for method comparisons (e.g. plain vs data at the same budget) the same
  paired tests are applied between the two variants (analysis script).

**Multiplicity.** The core grid has ~40 variants. McNemar p-values are
reported raw and with Holm–Bonferroni correction within each research
question family (A1…A12). Conclusions are stated from the effect size and its
interval; a significant p-value alone is never presented as practically
important. A practical-equivalence margin of **±1.0 pp top-1** is used when
calling two methods "equivalent".

## 7. Distillation protocol

Teacher = FP32 pretrained model; student = compressed model. Trainable
parameters: low-rank factors + LayerNorms. Data: `transfer` (teacher logits
cached for the image and its horizontal flip). Model selection on `v2_dev`
top-1 every `eval_every` steps with early stopping; the selected checkpoint is
evaluated **once** on `v2_test`. α ∈ {0, 0.5}, T ∈ {1, 2, 4} explored on dev
with a fixed step budget; seeds reported.

## 8. Latency and memory protocol

* Separate process per backend family; model loading excluded from timing.
* Fixed thread count (8), identical input tensor per batch size; batch 1 and 8.
* ≥ 10 warm-up calls, then ≥ 50 interleaved rounds in shuffled order per round.
* Reported: median, mean, std, p90, p95, throughput; speed-up vs FP32 dense
  with bootstrap 95% CI (2,000 resamples of the per-round times).
* Peak RSS sampled at 5 ms during the timed loop; serialized size of the
  deployable artifact (`state_dict` or `.onnx`).
* Layer microbenchmarks additionally in a **cold-cache** mode (64 MB buffer
  sweep before each call; L3 = 16 MB).
* **Latency runs require AC power.** The power source, battery charge and
  active Windows power scheme are recorded in every result's `env` block
  (`env.power`) and latency results measured on battery are flagged invalid.
  Thermal state is not controlled beyond interleaving. (At protocol time the
  machine was on battery, Balanced scheme; accuracy runs are unaffected.)

## 9. Deviations log

| Date | Change | Reason |
|---|---|---|
| 2026-11-20 14:07 | **Exploratory dev finding registered as confirmatory hypotheses for v2_test** (before any test run): H-iso: at equal whole-model params, `data_mlp_b0.7` > `data_attn+mlp_b0.8` (dev: +4.20 pp [+2.8, +5.6]); H-type: the data-vs-plain gain is larger for `mlp` than `attn` scope (dev: +13.45 vs +3.45 pp). Both variants are already in the core grid; tested with exact McNemar, α = 0.05, one test each. | dev results showed the equal-weight allocation objective misallocates across layer types |
| 2026-11-20 14:07 | Added a *sensitivity-weighted* allocator variant (`allocator_weights: sensitivity`, §5.2 weights w_i estimated from training-side images) to a second dev suite `dev2`, evaluated on v2_dev only; it enters the core grid only if it beats greedy on dev agreement by > 0.5 pp at b = 0.6. | follow-up to H-iso; post-hoc, therefore labelled exploratory |
