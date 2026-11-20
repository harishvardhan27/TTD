# Experiment Tracker

Status legend: ⬜ planned · 🔄 running · ✅ done · ❌ failed (diagnosed) · ⏸ superseded.
Claims legend: **Implemented → Tested → Validated → Paper-ready**, or **Unverified**.

## Execution plan (prioritised, compute-budgeted)

Measured eval cost on this laptop (Ryzen 7 7730U, 8 threads, batch 32):
DeiT-S 55 ms/img, ViT-B 174 ms/img. ImageNetV2-test (8,000 imgs):
**DeiT-S ≈ 7 min/config, ViT-B ≈ 23 min/config.** V2-dev (2,000): 2 / 6 min.

| Phase | Content | Est. CPU time | Gate to next phase |
|---|---|---|---|
| 0 | Audit, data provenance, plan | done | `RESEARCH_AUDIT.md` |
| 1 | Fix audit items H2–H4, M1–M5, M8–M9; ImageNetV2 loader + dev/test split + manifests + overlap checks; configs; tests (unit, integration, ONNX) | ~1 h | all tests pass |
| 2 | Re-answer Q1–Q3 on V2-**dev** (DeiT-S), small grid | ~30 min | sanity of protocol |
| 3 | Core grid on V2-**test**: DeiT-S full (≈40 configs); ViT-B confirmatory subset (≈12 configs) | ≈ 5 h + 4.5 h | results CSV/JSON |
| 4 | Distillation: α/T chosen on V2-dev; DeiT-S 3 runs, ViT-B 1 run | ≈ 4 h | before/after on V2-test |
| 5 | Latency/memory: batch 1 & 8, torch + ORT, FP32/INT8, layer microbenchmarks, kernel cold/hot | ≈ 1.5 h | bootstrap CIs |
| 6 | Remaining ablations (calib size/source, λ, seeds, quant order) | ≈ 3 h | — |
| 7 | Analysis: tables/figures generated from `results/raw` only | — | — |
| 8 | Docs + literature review + manuscript | — | every number traceable |
| 9 | Final validation: tests, clean install, ONNX checks, reproducibility re-run of 2 configs | ~1 h | completion report |

Model roles: **DeiT-S is the primary model for the full grid and ablations**
(4× cheaper, a realistic edge model); **ViT-B is confirmatory** on the key
configurations, so model-dependence (A11) is still tested. Every config is
checkpointed per result (resume skips finished configs), so long runs survive
interruption.

## Experiments

| ID | Question | Model | Data | Method / config | Seed | Status | Result | Artifacts | Interpretation / limits |
|---|---|---|---|---|---|---|---|---|---|
| P1 | Phase 1 foundations | — | — | audit fixes H2–H4, M1–M5, M8, M9; ImageNetV2 dev/test + manifests + overlap checks; runner, analysis, distillation, benchmark scripts | — | ✅ | 48/48 tests pass (unit, integration, ONNX graph/parity, legacy-import regression, resume-exactness); 2 new bugs found & fixed: FLOP counter omits fused SDPA; INT8 models under-report params | `tests/`, `data/manifests/` | Implemented + Tested |
| E1 | Q2/Q3 + hyperparameters (A1, A3, A4, A8, A9, A12, min_saving) | DeiT-S | v2_dev (2,000) | suite `dev` (25 variants), greedy allocation | calib seeds 1–3 | ✅ 13:46–14:36, 0 failures | Baseline 68.25%. **A1** data − plain: +10.25 pp (b=0.8), +17.20 (0.7), +30.75 (0.6); attn-only +3.45, mlp-only +13.45 (all Holm p ≤ 1e−5). **Iso-params**: data_mlp_b0.7 − data_attn+mlp_b0.8 = +4.20 pp [+2.8, +5.6]. **A9** λ: agreement 42.3/42.3/42.2/42.6/43.8% for 0…0.1 (boundary → extended in dev2). **A8** n: 35.1/38.9/40.5/42.6/43.9% for 32…512 (boundary → extended); source: sample 47.9%, transfer 42.6%, imagenette 42.1%. **A12** seeds: 42.4–42.8%. min_saving: 42.8/42.6/43.3%. Input anisotropy: top-1% of directions hold 17–33% of energy (fc2 highest). | `results/raw/deit_small/v2_dev/dev/`, `results/tables/deit_small_v2_dev_dev.md`, `results/figures/deit_small_v2_dev_dev_*` | Dev split: selection only. Registered H-iso, H-type, H-src for test (protocol §9). Even data-aware SVD loses 11–32 pp at b ≤ 0.8 without recovery. |
| E2 | Follow-ups: sensitivity-weighted allocation, exact DP, extended λ and n grids | DeiT-S | v2_dev | suite `dev2` (9 variants) | 1 | 🔄 started 14:36 | first: sens b=0.8 58.65% (agree 72.8%) vs greedy 57.10% (70.9%) | `results/raw/deit_small/v2_dev/dev2/` | exploratory (post-hoc) |
| E0 | Smoke: pipeline runs end-to-end | ViT-B | Imagenette-val 200 + sample 200 (**contaminated**, see audit C1) | legacy HO, uniform 256, attn budget 0.5 plain/data | 0 | ⏸ stopped by user after 6/8 configs | Legacy "0.15" ≡ uniform r=256 (identical predictions); −9.75 pp [−12.9, −6.6]; plain vs data at budget 0.5: 87.25% vs 86.75% (n.s.) | `results/vit_lowrank_vit_base_quick.*` | Superseded: eval data overlaps training data; error metrics not comparable (audit H2). Q1 answered qualitatively; Q2/Q3 to be re-run. |
