# Research Audit

Audit date: 2026-11-20. Scope: the full repository at commit `b10c8c1` plus the
uncommitted restructuring done in this session (`ttcompress/`, `tests/`,
`experiments/`). Every finding below cites the evidence that supports it.
Severity: **Critical** = invalidates a reported conclusion; **High** = a
reported number or claim is wrong or unsupported; **Medium** = engineering or
methodology defect with limited impact on conclusions; **Low** = hygiene.

Status vocabulary used throughout the project:
*Implemented* (code exists) → *Tested* (unit/integration test passes) →
*Validated* (experiment run under the protocol in `docs/EXPERIMENTAL_PROTOCOL.md`)
→ *Paper-ready* (validated, artifact-traceable, reviewed) · *Unverified*.

---

## 1. Current experiment state (Phase 0 snapshot)

| Item | State |
|---|---|
| Running processes | None. The smoke run (`experiments/vit_lowrank.py --quick`) was stopped by the user at 13:25 after 6 of 8 configs. |
| Smoke artifacts | `results/vit_lowrank_vit_base_quick.{json,log}`, `results/base_logits_vit_base_quick.pt`. Kept as-is; superseded by the protocol below (see C1, C2). |
| Data on disk | `data/imagenette2-320` (9,469 train / 3,925 val), `data/imagenet-sample-images` (1,000), `data/cache/*.pt`, ImageNetV2 download in progress. |
| Weights | `vit_base_patch16_224_miil.in21k_ft_in1k` (86,540,008 params), `deit_small_patch16_224.fb_in1k` (22,050,664 params) in the HF cache. |
| Tests | 11/11 pass (`pytest`). |
| Git | Restructuring uncommitted; `CNN/results_cnn.json`, `DistilBERT/results.json` modified by an earlier, unrecorded re-run (see M7). |

---

## 2. Findings

### Critical

**C1. Evaluation data overlaps the models' training data.**
*Evidence:* Imagenette "val" contains 3,791/3,925 files named `nXXXXXXXX_NNNN.JPEG`
(ImageNet-1k **train** naming); only 134 are `ILSVRC2012_val_*`. The
`imagenet-sample-images` set's 10 Imagenette-class images are pixel-identical
(16×16 fingerprint correlation 1.000) to Imagenette files with train-style
names, 8 of them inside Imagenette *train*.
*Impact:* every accuracy reported so far, the old HO table (93%) and this
session's smoke run, was measured largely on images the pretrained model was
trained on. Absolute accuracies are inflated; compression-induced degradation
on memorised images is not guaranteed to match degradation on unseen images.
The sample set also overlaps the calibration source.
*Correction:* final evaluation moves to **ImageNetV2 matched-frequency**
(10,000 images, 1,000 classes, collected after ImageNet; Recht et al., 2019),
split per class into a dev split (2/class, 2,000 images, for hyperparameter
selection only) and a test split (8/class, 8,000 images, final reporting only).
Imagenette and the sample set are re-designated as **training-side data**
(calibration, distillation), where ImageNet-train provenance is appropriate.
Overlap checks are automated (see §4).
*Residual risk:* ViT-B/miil is pretrained on ImageNet-21k; whether any
ImageNetV2 image is in ImageNet-21k is **unverified** (V2 was collected from
Flickr after 2012 with de-duplication against ImageNet; 21k overlap was not
checked by its authors as far as we know).

**C2. Old HO "target error" results are one uniform-rank configuration.**
*Evidence:* `results/vit_lowrank_vit_base_quick.log`: replicating
`HO/low_rank_compress.suggest_rank_by_target_error` exactly, 0/24 layers meet
0.15 and 3/24 meet 0.30; all others fall back silently to rank 256. Param
count of uniform rank 256 = 72.41 M = the HO table's 72.4 M. The "0.15" config
gives bit-identical predictions to `uniform_rank256` (same accuracy, agreement,
KL). Achieved mean weight error 0.455 (max 0.661), not ≤0.15.
*Correction:* `ttcompress.lowrank.rank_for_target_error` keeps unreachable
layers dense and reports why. Regression test
`test_rank_for_target_error_reports_unreachable_targets`.

**C3. The old HO accuracy claim does not reproduce.** HO reported a 2–3 point
drop for the "0.15" config; the identical configuration gives −9.75 pp
(95% CI [−12.9, −6.6], McNemar p = 8.7e−10, n = 400) on the smoke subset.
Possible contributors: different timm checkpoint resolution (see H1), 100 vs
400 images, contaminated data in both. To be re-measured on ImageNetV2.

### High

**H1. `vit_base_patch16_224_miil` silently resolves to the ImageNet-21k head
in timm ≥ 0.9** (11,221 classes). *Evidence:* first download in this session
fetched `…miil.in21k`. *Correction:* all configs pin
`…miil.in21k_ft_in1k` and record the checkpoint SHA-256 (§4).

**H2. Incomparable error metrics in the new experiment log.** `CompressionReport`
prints weight error for plain SVD and calibration *output* error for
data-aware SVD under the same name `rel. error` (0.416 vs 0.175 in the smoke
log). *Correction:* report both methods' **output error on the same held-out
activations** plus the weight error, under distinct names.

**H3. Reported data-aware error is for the damped covariance.** `factorize`
whitens with `C + λ·mean(diag C)·I` (λ = 1e−2) but `rel_error` is quoted as
the output error; the identity holds exactly only for λ = 0 (the unit test
uses λ = 0). *Correction:* measured output error is computed directly; λ is
ablated (A9) and selected on calibration-held-out activations.

**H4. Greedy allocation is described as optimal.** `allocate_ranks`
docstring says greedy "is optimal". It is optimal for the continuous
relaxation of a separable concave knapsack, not for the integer problem with
unequal chunk costs and a dense-fallback option. *Correction:* docstring and
theory restated with the actual guarantee; exact DP solver added for
comparison (§3.3 of the theory doc).

**H5. Theory document errors** (`THEORY_VERIFY/THEORY.md`). §1 Step 3 mixes
two definitions of ε_k (tails of remainder unfoldings vs. original
unfoldings); the √(d−1) bound is Oseledets' result with the latter. §3's bound
uses a Lipschitz product that is vacuous for real ViTs (‖W‖₂ ≫ 1 per layer)
and is not used by any code. *Correction:* rewritten in `docs/THEORY.md`.

**H6. "Riemannian" TT fitting is projected gradient descent.**
`riemannian_tt_fit` takes a Euclidean step on the dense tensor and re-applies
TT-SVD; no tangent-space projection. On the quadratic reconstruction loss,
TT-SVD of the target is already the (quasi-)optimal closed form, so
"matches TT-SVD from random init" is expected, not evidence. *Correction:*
documentation corrected; method retained only as a historical artifact.

**H7. Latency claims from 5 sequential iterations.** HO's "12% faster" timed
the compressed model and then the baseline, 5 iterations each, no
interleaving or intervals. Unverified. *Correction:* `bench.compare_latency`
(interleaved, shuffled, bootstrap CI). Re-measured in Phase 5.

**H8. CPU kernel benchmark favours the dense layer** (hot-cache repeated
calls on a 9.4 MB weight) and loads a non-existent path
(`THEORY_VERIFY/kernels/tt_linear_cpu.cpp`). *Correction:* path fixed; cold/
large-working-set mode added (Phase 5).

**H9. CUDA kernel cannot run the project's own shapes.**
`MAX_SHARED_FLOATS = 4096`; for 768→3072 at r=8, `state_a = m1·n2·n3·r1`
already exceeds it. One block per sample, no tiling. No NVIDIA GPU here:
**untested, excluded from all results**, documented as such.

### Medium

| ID | Finding | Evidence | Correction |
|---|---|---|---|
| M1 | Experiment script combines datasets of different distributions into an `all` accuracy | `vit_lowrank.py` `scores["all"]` | removed; per-split only |
| M2 | Distillation has no CE term, no validation-based checkpoint selection, no resumable checkpoints | `ttcompress/distill.py` | α·CE + (1−α)T²·KD, dev-split selection, checkpointing |
| M3 | Benchmark lacks mean/std/p90/p95, RSS, batch-1/8 protocol | `bench.py` | extended |
| M4 | ONNX INT8 path never exercised; operator coverage unknown | `quant.py` | graph inspection test; per-op report |
| M5 | No checkpoint hashes, dataset manifests, config files, experiment IDs | — | `configs/`, manifests, run IDs |
| M6 | Legacy scripts have broken imports (now point to moved modules), duplicated READMEs (`FNN`=`THEORY_VERIFY`=`VIT` README; `FNN`=`VIT` findings) | md5 audit | imports routed to `ttcompress`; duplicates replaced by pointers |
| M7 | Tracked result files silently overwritten by an unrecorded re-run; DistilBERT dense latency changed 30.4 → 14.8 ms between runs | `git diff DistilBERT/results.json` | left untouched (provenance unknown, not used by any new analysis); new results go to `results/raw/<run_id>/` and are never overwritten |
| M8 | Rank allocation ignores hardware: several `proj` layers get rank 376 of break-even 383 (2% saving, extra GEMM) | smoke JSON | minimum-saving threshold; latency-aware variant (§3.4) evaluated, not assumed |
| M9 | `setup_threads` assumes 2-way SMT | `experiments/common.py` | explicit thread count in configs, recorded |
| M10 | Calibration covered only 10 classes (Imagenette) | `vit_lowrank.py` | calibration-source ablation (A8) incl. 1,000-class sample set |

### Low
No `.gitignore` (caches, `__pycache__`, `*.egg-info`, data); `.vscode` with
machine-specific paths tracked; `THEORY_VERIFY/mnt/user-data/...` stray sandbox path.

---

## 3. Issues confirmed fixed and guarded against regression

| Historical issue | Guard |
|---|---|
| Silent rank-256 fallback | `rank_for_target_error` + test |
| 100-image evaluation | ImageNetV2 test split, 8,000 images; Wilson CI + paired McNemar on every number |
| 5-iteration sequential latency | interleaved bootstrap protocol |
| Hot-cache kernel benchmark | cold mode in kernel benchmark |
| Riemannian over-claim | documentation corrected |
| Broken imports / duplicates | package + import smoke test over legacy scripts |

## 4. Remaining limitations (known now)
* CPU-only; no GPU latency claim can be made.
* ImageNetV2 is 10 images/class: per-class metrics have very wide intervals and are reported as distributions, not per-class claims.
* ImageNet-21k vs ImageNetV2 overlap unverified (C1).
* Distillation transfer set (Imagenette + sample, ~14k images, 1,000-class coverage only via the 1k sample) is far smaller than ImageNet; recovery results are a lower bound.

## 5. Changes made (log)
Maintained in `EXPERIMENT_TRACKER.md` and git history.
