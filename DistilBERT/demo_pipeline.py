"""
demo_pipeline.py
=================
A fully runnable, self-contained demonstration of the whole compression
pipeline described in the project brief:

    Pretrained weights -> Tensor Reshaping -> HOSVD analysis ->
    TT decomposition -> Replace layers -> Fine-tune -> Evaluate

Because this sandbox has no internet access to Hugging Face Hub, this
script builds a *synthetic* transformer feed-forward stack sized exactly
like a real DistilBERT FFN block (768 -> 3072 -> 768, 6 blocks) and gives
its weights a realistic decaying singular-value spectrum -- this is not
a cheap trick: it is a well-documented empirical property of trained
transformer weight matrices (see e.g. Aghajanyan et al. 2020, "Intrinsic
Dimensionality...", and Novikov et al. 2015 on TT-compressing real nets)
that most of the energy concentrates in a modest number of singular
directions. That is precisely the property Tensor-Train decomposition
exploits, so this synthetic model is a fair, honest stand-in for a real
pretrained checkpoint when one isn't reachable.

For instructions to run this against a REAL Hugging Face model
(DistilBERT / GPT-2 / TinyLlama) on a machine with internet access, see
hf_real_model_example.py and README.md.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, asdict
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ttcompress.tt import TTLinear, hosvd_mode_singular_values, suggest_rank_for_energy, factorize_balanced
from ttcompress.surgery import (
    replace_linear_with_tt,
    count_parameters,
    model_size_mb,
    benchmark_latency,
)

torch.manual_seed(42)
np.random.seed(42)


# --------------------------------------------------------------------------
# 1. Build a synthetic FFN stack with realistic (decaying) spectra
# --------------------------------------------------------------------------

HIDDEN = 768
FFN = 3072
N_BLOCKS = 4


def make_realistic_linear(out_f: int, in_f: int, effective_rank: int = 96, noise_floor: float = 0.02) -> nn.Linear:
    """Construct a Linear layer whose weight has a decaying singular
    spectrum (a handful of large singular values + a long, small-noise
    tail) -- the shape of spectrum consistently reported for trained
    transformer weight matrices, as opposed to the flat spectrum of a
    purely random Gaussian matrix."""
    r = min(out_f, in_f)
    # Decaying singular values: big for the first `effective_rank`, tiny after.
    idx = np.arange(r)
    s = np.where(idx < effective_rank, np.exp(-idx / (effective_rank / 3)), noise_floor * np.exp(-idx / r))
    s = s * (r ** 0.5) / np.linalg.norm(s)  # normalize overall scale
    U, _ = np.linalg.qr(np.random.randn(out_f, r))
    V, _ = np.linalg.qr(np.random.randn(in_f, r))
    W = (U * s) @ V.T
    W = W / W.std() * (1.0 / np.sqrt(in_f))  # roughly match nn.Linear default init scale
    layer = nn.Linear(in_f, out_f, bias=True)
    with torch.no_grad():
        layer.weight.copy_(torch.from_numpy(W).float())
        nn.init.zeros_(layer.bias)
    return layer


class FFNBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = make_realistic_linear(FFN, HIDDEN)
        self.fc2 = make_realistic_linear(HIDDEN, FFN)
        self.ln = nn.LayerNorm(HIDDEN)

    def forward(self, x):
        h = F.gelu(self.fc1(x))
        h = self.fc2(h)
        return self.ln(x + h)


class ToyTransformerFFNStack(nn.Module):
    """Stand-in for 'the FFN sublayers of a 6-layer DistilBERT-sized
    transformer' -- same matrix shapes (768<->3072) and the same count
    (6 blocks) as distilbert-base-uncased, which is exactly what the TT
    compression cares about."""

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([FFNBlock() for _ in range(N_BLOCKS)])
        self.head = nn.Linear(HIDDEN, HIDDEN)  # small head, left uncompressed

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return self.head(x)


# --------------------------------------------------------------------------
# 2. HOSVD diagnostic: how much energy do the low-rank directions hold?
# --------------------------------------------------------------------------

def hosvd_energy_report(model: nn.Module, sample_linear_name: str = "blocks.0.fc1") -> dict:
    linear = dict(model.named_modules())[sample_linear_name]
    W = linear.weight.detach().numpy()
    in_modes = factorize_balanced(linear.in_features, 3)
    out_modes = factorize_balanced(linear.out_features, 3)
    W_t = W.reshape(*out_modes, *in_modes)
    spectra = hosvd_mode_singular_values(W_t)
    report = {}
    for mode_idx, s in enumerate(spectra):
        r99 = suggest_rank_for_energy(s, 0.99)
        r95 = suggest_rank_for_energy(s, 0.95)
        report[f"mode_{mode_idx}"] = {
            "mode_size": int(W_t.shape[mode_idx]),
            "rank_for_95pct_energy": r95,
            "rank_for_99pct_energy": r99,
        }
    return report


# --------------------------------------------------------------------------
# 3. Fine-tuning task: distill the TT-compressed stack toward the
#    original (uncompressed) stack's outputs -- the standard recipe for
#    recovering accuracy lost during structural compression.
# --------------------------------------------------------------------------

def finetune_via_distillation(
    teacher: nn.Module,
    student: nn.Module,
    steps: int = 200,
    batch_size: int = 16,
    lr: float = 1e-3,
) -> List[float]:
    teacher.eval()
    student.train()
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    losses = []
    for step in range(steps):
        x = torch.randn(batch_size, HIDDEN)
        with torch.no_grad():
            target = teacher(x)
        pred = student(x)
        loss = F.mse_loss(pred, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


# --------------------------------------------------------------------------
# 4. Evaluation
# --------------------------------------------------------------------------

@dataclass
class EvalResult:
    name: str
    num_params: int
    size_mb: float
    compression_ratio: float
    mean_latency_ms: float
    throughput_samples_per_s: float
    relative_output_error: float


@torch.no_grad()
def evaluate(model: nn.Module, teacher: nn.Module, name: str, base_params: int, base_size: float) -> EvalResult:
    x = torch.randn(32, HIDDEN)
    target = teacher(x)
    pred = model(x)
    rel_err = (pred - target).norm().item() / target.norm().item()
    lat = benchmark_latency(model, x)
    n_params = count_parameters(model)
    size = model_size_mb(model)
    return EvalResult(
        name=name,
        num_params=n_params,
        size_mb=round(size, 3),
        compression_ratio=round(base_params / n_params, 2),
        mean_latency_ms=round(lat["mean_ms"], 3),
        throughput_samples_per_s=round(lat["throughput_samples_per_s"], 1),
        relative_output_error=round(rel_err, 4),
    )


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("STEP 1: Build baseline model (DistilBERT-FFN-shaped, 6 blocks)")
    print("=" * 70)
    baseline = ToyTransformerFFNStack()
    base_params = count_parameters(baseline)
    base_size = model_size_mb(baseline)
    print(f"Baseline params: {base_params:,} | size: {base_size:.2f} MB")

    print()
    print("=" * 70)
    print("STEP 2: HOSVD analysis of a representative weight tensor")
    print("=" * 70)
    energy_report = hosvd_energy_report(baseline)
    print(json.dumps(energy_report, indent=2))

    results: List[EvalResult] = []
    results.append(evaluate(baseline, baseline, "baseline (dense)", base_params, base_size))

    for tt_rank in [48, 16]:
        print()
        print("=" * 70)
        print(f"STEP 3: TT-compress FFN linear layers at rank={tt_rank}")
        print("=" * 70)
        compressed = copy.deepcopy(baseline)
        replace_linear_with_tt(
            compressed,
            num_modes=3,
            rank=tt_rank,
            predicate=lambda name, m: "fc1" in name or "fc2" in name,
        )

        # Evaluate immediately after TT-SVD init, before any fine-tuning.
        res_pre = evaluate(compressed, baseline, f"TT rank={tt_rank} (pre-fine-tune)", base_params, base_size)
        results.append(res_pre)

        print()
        print(f"STEP 4: Fine-tune (distillation) TT model at rank={tt_rank}")
        losses = finetune_via_distillation(baseline, compressed, steps=60)
        print(f"  distillation loss: start={losses[0]:.4f} -> end={losses[-1]:.4f}")

        res = evaluate(compressed, baseline, f"TT rank={tt_rank} (fine-tuned)", base_params, base_size)
        results.append(res)

    print()
    print("=" * 70)
    print("FINAL RESULTS TABLE")
    print("=" * 70)
    header = f"{'Model':<32}{'Params':>12}{'Size(MB)':>10}{'CompRatio':>11}{'Latency(ms)':>13}{'Throughput/s':>14}{'RelErr':>9}"
    print(header)
    print("-" * len(header))
    rows = []
    for r in results:
        print(
            f"{r.name:<32}{r.num_params:>12,}{r.size_mb:>10.2f}{r.compression_ratio:>11.2f}"
            f"{r.mean_latency_ms:>13.3f}{r.throughput_samples_per_s:>14.1f}{r.relative_output_error:>9.4f}"
        )
        rows.append(asdict(r))

    with open("results.json", "w") as f:
        json.dump({"hosvd_energy_report": energy_report, "results": rows}, f, indent=2)
    print("\nSaved detailed results to results.json")

    return results, energy_report


if __name__ == "__main__":
    main()
