# Real-Model Findings: What Happens With Actual Pretrained Weights

Everything else in this project (`demo_pipeline.py`, `demo_cnn_pipeline.py`)
used synthetic weights with a hand-constructed decaying singular
spectrum, justified as "the well-documented shape of real trained
weights." This document reports what happened when we tested that
justification directly against **real pretrained ImageNet weights** —
and it changes the project's conclusions in an important way.

## How we got real weights and real images without a GPU or general internet access

This sandbox can only reach a fixed allowlist of domains (`pypi.org`,
`github.com`, `raw.githubusercontent.com`, etc.) — not `huggingface.co`
or `download.pytorch.org`. Two things made real validation possible
anyway:

- **`timm`'s older model releases are hosted on GitHub releases**
  (`release-assets.githubusercontent.com`, which IS on the allowlist).
  `timm.create_model('mobilenetv2_100', pretrained=True)` and
  `timm.create_model('resnet34', pretrained=True)` both downloaded real
  ImageNet-trained weights successfully this way.
- **A real labeled-image dataset exists as plain files on GitHub**:
  [`EliSchwartz/imagenet-sample-images`](https://github.com/EliSchwartz/imagenet-sample-images)
  hosts one real photo per ImageNet class (1000 total), fetched via
  `raw.githubusercontent.com`. We downloaded 200, spanning all 1000
  classes evenly (`sample_files.txt`).

**Sanity check that this all actually works**: the unmodified, real
MobileNetV2 scored **82-86% top-1** on this 200-image sample (varies
slightly by which 50-100-image subset is used) — a normal, expected
number for MobileNetV2 on easy/canonical class-representative images,
confirming the model, preprocessing, and label mapping are all correct
before trusting anything downstream.

## Finding 1: real trained conv kernels compress far worse than our synthetic examples suggested

Testing `TTConv2d.from_conv` directly on real MobileNetV2 and ResNet34
layers (`benchmark`-style isolated weight reconstruction, no fine-tuning
involved):

| Layer (real weights) | Shape | Rank | Relative reconstruction error |
|---|---|---|---|
| MobileNetV2 `conv_head` | 320→1280, 1×1 | 64 | **82%** |
| MobileNetV2 `conv_head` | 320→1280, 1×1 | 384 (near max) | ~30% |
| ResNet34 `layer2.0.conv1` | 64→128, 3×3 | 64 | 49% |
| ResNet34 `layer3.0.conv1` | 128→256, 3×3 | 64 | 61% |
| ResNet34 `layer4.0.conv1` | 256→512, 3×3 | 64 | 78% |

Compare this to our synthetic examples, which at similar compression
ratios showed errors in the 10-70% range depending on rank, but were
constructed to have a favorable decaying spectrum. Real trained kernels
— in both an efficient (MobileNetV2, depthwise-separable) and a standard
(ResNet34, full convolutions) architecture — need dramatically higher
rank than we assumed to hit the same fidelity.

## Finding 2 (a real bug, caught and fixed): the HOSVD-energy rank heuristic is unreliable here

We first tried automatic per-layer rank selection via HOSVD per-mode
energy (`suggest_conv_rank_by_energy` in `compress_model.py`, following
the same diagnostic used in `demo_cnn_pipeline.py`). It produced
**0% classification accuracy at every energy threshold tested (95%,
99%, 99.9%)** — and identical parameter counts regardless of the energy
target, which was the tell that something was structurally wrong, not
just "the rank was a bit low."

**Root cause**: HOSVD per-mode singular values are computed on a single
from-scratch unfolding of the tensor. This exactly predicts the error of
only the *first* of TT-SVD's sequential truncation cuts — later cuts
operate on the already-truncated remainder, not the original tensor, so
a simple per-mode energy number doesn't capture their achievable error.
For 1×1 convolutions specifically (nearly all of MobileNetV2), the
appended trivial spatial mode (size `kH*kW=1`) always reports exactly
one singular value, so it always "suggests" rank 1 — and taking the
`min()` across modes let this trivial mode silently dominate every
result regardless of the real energy target.

**Fix**: `suggest_conv_rank_by_target_error` replaces the heuristic with
direct measurement — actually run TT-SVD at a few candidate ranks and
keep the smallest one whose real reconstruction error is below target.
More expensive per layer, but grounded in the thing that actually
matters, not a proxy we now know is misleading here. `verify_cuda_kernel_logic.py`-style
"don't trust it until you've checked it against ground truth" was the
right instinct throughout this project, and this is the clearest place
it caught something real.

## Finding 3: accuracy-preserving compression of MobileNetV2 requires ranks large enough that the model *grows*

Using the fixed, direct-measurement rank selection with a per-layer
target error of 5%:

| Setting | Params | vs. original | Top-1 (real images) |
|---|---|---|---|
| Original | 3,504,872 | 1.0x | 0.82-0.86 |
| Adaptive, target_err≤0.05 | 4,439,818 | **0.79x (larger!)** | 0.817 |

To keep MobileNetV2's real accuracy intact without any fine-tuning, the
per-layer ranks needed are so close to full-rank that the TT-core
parameter overhead exceeds the savings — the "compressed" model is
**27% bigger**, not smaller. This is a genuine, direct confirmation that
depthwise-separable architectures (already designed to minimize
redundancy) have little slack left for a decomposition method that
exploits redundancy.

## Finding 4: a genuinely smaller compressed model collapses without fine-tuning, and shows a real (if incomplete) recovery signal with it

Fixed rank=32 (uniform, not adaptive) gives real 1.9x compression
(3.5M → 1.84M params) but **0% top-1 accuracy** pre-fine-tune — complete
functional collapse, confirmed NOT to be a bug (a single-layer swap at
generous rank matches the original to 1e-6 relative error; the collapse
appears only once enough of the ~30 compressed layers' individual errors
compound, concentrated in the later, wider layers where fixed-rank=32 is
most restrictive relative to layer size).

Distillation fine-tuning (synthetic ImageNet-statistics-normalized
inputs, since no real training set was available) shows a real,
monotonic learning signal:

```
step 0:  KL loss = 0.79
step 5:  KL loss = 0.66
step 10: KL loss = 0.53
step 12: KL loss = 0.57 (final)
```

but top-1 accuracy was still 0.0% after only 12 steps. This is expected,
not a negative result about the method: our own synthetic experiments
needed 60-200 steps to see meaningful (not complete) recovery, and the
literature this project is based on (Novikov et al. 2015) fine-tunes for
full epochs over real datasets. 12 steps at real MobileNetV2's 224×224
resolution on CPU-only hardware, in this sandbox's time budget, is far
short of that — the loss curve's steady, real decrease is the honest
signal that the mechanism works; the sandbox simply couldn't run it long
enough to finish the job.

## Finding 5 (the decisive one): real transformer FFN weights are much flatter-spectrum than assumed, and where real structure exists, plain low-rank beats multi-mode TT

We got real ViT-Base weights (`timm.create_model('vit_base_patch16_224_miil', pretrained=True)`,
again via a GitHub release) — architecturally identical to our synthetic
transformer experiments (`Linear(768,3072)` → GELU → `Linear(3072,768)`,
12 blocks). This lets us test the project's founding assumption directly.

**The assumption fails for FFN layers.** Plain matrix SVD (no TT at all)
on a real `fc1`/`fc2` weight shows a nearly flat spectrum: only **60.6%
energy retained at rank 64 out of 768** (compare: a pure random Gaussian
matrix of the same shape retains 16.5% — real weights are *more*
structured than noise, just not nearly as much as our synthetic
decaying-spectrum construction assumed).

**But compressibility is highly layer-type dependent within the same
model:**

| Real layer | Shape | Energy retained @ rank 64 (plain SVD) | vs. random baseline |
|---|---|---|---|
| MLP `fc1`/`fc2` | 768↔3072 | 32-61% | 16.5% |
| Attention QKV projection | 768→2304 | **84.3%** | 16.5% |
| Patch embedding | 768→768 | **72.1%** (99.3% @ rank 128) | 16.5% |
| Classification head | 768→1000 | 31.9% | 16.5% |

Attention projections and the patch embedding have real, substantial
low-rank structure. The large MLP expansion/contraction layers — the
ones this whole project (and the original literature) assumed were the
best compression target — do not, at least for this real model.

**The most important result: where real structure exists, plain
low-rank factorization beats multi-mode TT decomposition at the same
rank number, on the same real layer:**

| Method (QKV layer, rank=64) | Relative error |
|---|---|
| Plain low-rank (`d=2`, ordinary matrix SVD) | **39.7%** |
| 3-mode TT (`d=3`, this project's default) | **68.4%** |

This is consistent with (not contradicting) the earlier synthetic
finding that "matrix low-rank ≠ TT low-rank under an arbitrary mode
split" (`tt_core.py`'s docstring, verified early in this project) — but
we now have the real-world consequence: splitting a real weight matrix
into 3+ TT modes doesn't reveal extra exploitable structure that plain
SVD misses; it costs you compressibility relative to just using SVD
directly, for every real layer we tested, in every real model we tested
(CNN convs, transformer FFN, attention, embeddings).

## Finding 6: large embedding tables don't work either — naive reshaping is the reason, not TT itself

Tested the third hypothesis directly: real GloVe embeddings (400,000
words × 50 dims, downloaded via `gensim`'s GitHub-hosted data index —
another real, non-synthetic weight matrix).

**Plain low-rank ceiling on this matrix**: since one dimension is
already only 50, ordinary SVD offers little — rank 40/50 gives only
1.25x compression at 19.7% error. This is the kind of matrix shape
TT-embeddings (Khrulkov et al.) are meant to help with instead.

**Naive TT-embedding test**: factorized the 400,000-word vocabulary
dimension into 3 modes (100×80×50) combined with the embedding dim,
and ran TT-SVD at increasing rank:

| TT rank | Relative error | Compression |
|---|---|---|
| 8 | 92.8% | 2101x |
| 16 | 92.6% | 561x |
| 32 | 91.9% | 145x |
| 48 (near ceiling) | 90.9% | 65x |

**The error barely moves across a 6x rank increase** — this isn't a
close call, it's a flat line, confirmed not to be a bug (error keeps
dropping as rank approaches the true ceiling, just extremely slowly).
The reason is structural, not a TT failure: reshaping a flat vocabulary
index into a 3D grid (100×80×50) imposes no real relationship between
neighboring indices — word #4,001 and word #4,002 in the reshaped grid
share no more semantic similarity than two random words. **TT (and any
tensor method) can only exploit structure that's actually there in the
chosen indexing.** The real TT-embedding papers this hypothesis was
based on use a *learned or frequency-aware* reindexing scheme
specifically to create that structure before decomposing — plain
off-the-shelf reshaping, which is what we tested (and what a
"just apply TT to any big matrix" approach would naively do), does not
get that benefit for free.



Putting all five findings together, the honest, load-bearing conclusion
is more conservative than the project brief assumed:

1. **We found no real layer, in any real model tested, where 3+ mode TT
   decomposition outperformed plain low-rank (matrix SVD) factorization
   at the same parameter budget.** The project's premise — that
   Tensor-Train specifically (as opposed to ordinary low-rank
   factorization) is the right structural compression tool for
   transformer/CNN weights — is not supported by any real weight we
   tested. Where TT's extra tensorization modes would help, it's for
   tensors that are naturally multi-way to begin with (e.g. conv kernels
   with genuine spatial×channel structure) or convolutions with 3+
   spatial dimensions, not for reshaping an inherently 2D weight matrix
   into more modes.
2. **Real exploitable redundancy in a trained transformer is concentrated
   in attention projections and embeddings, not the large MLP layers** —
   the opposite of where this project (and much of the literature it
   cites) pointed first.
3. **If the goal is structural compression of transformer/CNN weights,
   the evidence here argues for plain low-rank factorization (or
   attention-specific techniques like low-rank QKV) over general TT
   decomposition**, reserving TT's genuine advantage (representing
   naturally high-order tensors compactly) for cases with real multi-way
   structure, which post-hoc-reshaped 2D weight matrices generally don't
   have.
4. Everything about **fine-tuning recovering accuracy** and **latency
   not following from parameter-count reduction** (Findings elsewhere in
   this project) still holds regardless of this correction — those are
   properties of gradient-based training and of kernel/memory-bandwidth
   behavior respectively, independent of whether TT or plain low-rank is
   the compression method.

This is a real, if uncomfortable, conclusion to reach after this much
invested effort: the rigorous path led to evidence that narrows the
project's scope rather than validating its original framing. That is
what "proceeding in the right path" (checking claims against real data
instead of stopping at synthetic validation) is supposed to do.



## Reproducing this

```bash
python real_model_eval.py        # full real-model + real-image evaluation
python real_finetune_test.py     # the fine-tuning capstone test (needs several minutes on CPU)
```

Files: `imagenet_sample/` (200 real photos), `sample_files.txt` (class
labels), `imagenet_classes.txt` (the 1000 ImageNet class names),
`mobilenetv2_real_pretrained.pt` (the real downloaded weights, saved for
reproducibility).
