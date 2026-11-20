"""
demo_cnn_pipeline.py
======================
Same pipeline as demo_pipeline.py (which targets transformer FFN
matrices), applied to a convolutional network instead:

    Pretrained conv kernels -> HOSVD analysis -> TT decomposition ->
    Replace Conv2d with TTConv2d -> Fine-tune -> Evaluate

Builds a small CNN sized like an edge-deployment vision backbone
(32->64->128->256 channels, 3x3 convs, similar depth/width to a
MobileNetV1/VGG-lite trimmed for e.g. Jetson Nano / Raspberry Pi), gives
its kernels a realistic decaying singular spectrum (same rationale as
demo_pipeline.py -- this is the well-documented property that makes
trained CNN filters compressible, unlike random Gaussian weights), then
TT-compresses the conv layers and fine-tunes via distillation against
the uncompressed network.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, asdict
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ttcompress.tt import hosvd_mode_singular_values, suggest_rank_for_energy, factorize_balanced
from ttcompress.surgery import (
    replace_conv_with_tt,
    count_parameters,
    model_size_mb,
    benchmark_latency,
)

torch.manual_seed(42)
np.random.seed(42)


# --------------------------------------------------------------------------
# 1. Build a synthetic CNN with realistic (decaying) kernel spectra
# --------------------------------------------------------------------------

CHANNELS = [32, 64, 128, 256]
IMG_SIZE = 32  # CIFAR-sized input, typical for a small edge vision model


def make_realistic_conv(in_ch: int, out_ch: int, k: int = 3, stride: int = 1, effective_rank_frac: float = 0.35) -> nn.Conv2d:
    """Construct a Conv2d whose kernel, reshaped to (out_ch, in_ch*k*k),
    has a decaying singular spectrum -- the shape reliably reported for
    trained CNN filters (a handful of dominant filters + a long small tail),
    as opposed to the flat spectrum of pure random init."""
    conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=stride, padding=k // 2, bias=True)
    fan_in = in_ch * k * k
    r = min(out_ch, fan_in)
    eff_rank = max(1, int(r * effective_rank_frac))
    idx = np.arange(r)
    s = np.where(idx < eff_rank, np.exp(-idx / (eff_rank / 3)), 0.02 * np.exp(-idx / r))
    s = s * (r ** 0.5) / np.linalg.norm(s)
    U, _ = np.linalg.qr(np.random.randn(out_ch, r))
    V, _ = np.linalg.qr(np.random.randn(fan_in, r))
    W = (U * s) @ V.T
    W = W / W.std() * (1.0 / np.sqrt(fan_in))
    with torch.no_grad():
        conv.weight.copy_(torch.from_numpy(W).float().reshape(out_ch, in_ch, k, k))
        nn.init.zeros_(conv.bias)
    return conv


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super().__init__()
        self.conv = make_realistic_conv(in_ch, out_ch, k=3, stride=stride)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class ToyEdgeCNN(nn.Module):
    """Stand-in for a small edge-deployment CNN backbone (channel/depth
    profile similar to a trimmed MobileNet/VGG for Jetson Nano / RPi-class
    hardware): 4 conv blocks downsampling 32x32 -> 2x2, global pool, head."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        chans = [3] + CHANNELS
        self.blocks = nn.ModuleList(
            [ConvBlock(chans[i], chans[i + 1], stride=2) for i in range(len(CHANNELS))]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(CHANNELS[-1], num_classes)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


# --------------------------------------------------------------------------
# 2. HOSVD diagnostic on a representative conv kernel
# --------------------------------------------------------------------------

def hosvd_energy_report_conv(model: nn.Module, sample_conv_name: str = "blocks.2.conv") -> dict:
    conv = dict(model.named_modules())[sample_conv_name]
    W = conv.weight.detach().numpy()  # (C_out, C_in, kH, kW)
    C_out, C_in, kH, kW = W.shape
    out_modes = factorize_balanced(C_out, 2)
    in_modes = (*factorize_balanced(C_in, 2), kH * kW)
    W_t = W.reshape(*out_modes, *in_modes)
    spectra = hosvd_mode_singular_values(W_t)
    report = {}
    for mode_idx, s in enumerate(spectra):
        report[f"mode_{mode_idx}"] = {
            "mode_size": int(W_t.shape[mode_idx]),
            "rank_for_95pct_energy": suggest_rank_for_energy(s, 0.95),
            "rank_for_99pct_energy": suggest_rank_for_energy(s, 0.99),
        }
    return report


# --------------------------------------------------------------------------
# 3. Fine-tuning via distillation against the uncompressed CNN
# --------------------------------------------------------------------------

def finetune_via_distillation(teacher: nn.Module, student: nn.Module, steps: int = 60, batch_size: int = 16, lr: float = 1e-3) -> List[float]:
    teacher.eval()
    student.train()
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    losses = []
    for step in range(steps):
        x = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE)
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
def evaluate(model: nn.Module, teacher: nn.Module, name: str, base_params: int) -> EvalResult:
    x = torch.randn(16, 3, IMG_SIZE, IMG_SIZE)
    model.eval()
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


def main():
    print("=" * 70)
    print("STEP 1: Build baseline edge CNN (32->64->128->256 channels, 3x3 convs)")
    print("=" * 70)
    baseline = ToyEdgeCNN()
    baseline.eval()
    base_params = count_parameters(baseline)
    base_size = model_size_mb(baseline)
    print(f"Baseline params: {base_params:,} | size: {base_size:.2f} MB")

    print()
    print("=" * 70)
    print("STEP 2: HOSVD analysis of a representative conv kernel")
    print("=" * 70)
    energy_report = hosvd_energy_report_conv(baseline)
    print(json.dumps(energy_report, indent=2))

    results: List[EvalResult] = []
    results.append(evaluate(baseline, baseline, "baseline (dense)", base_params))

    for tt_rank in [32, 8]:
        print()
        print("=" * 70)
        print(f"STEP 3: TT-compress conv layers at rank={tt_rank}")
        print("=" * 70)
        compressed = copy.deepcopy(baseline)
        replace_conv_with_tt(
            compressed,
            num_modes=2,
            rank=tt_rank,
            # default predicate skips the 3-channel stem conv: a layer that
            # small has nothing to gain from TT and can even grow under it
            # (verified below) -- see README "what we verified" section.
        )

        res_pre = evaluate(compressed, baseline, f"TT rank={tt_rank} (pre-fine-tune)", base_params)
        results.append(res_pre)

        print()
        print(f"STEP 4: Fine-tune (distillation) TT-CNN at rank={tt_rank}")
        compressed.train()
        losses = finetune_via_distillation(baseline, compressed, steps=60)
        print(f"  distillation loss: start={losses[0]:.4f} -> end={losses[-1]:.4f}")
        compressed.eval()

        res = evaluate(compressed, baseline, f"TT rank={tt_rank} (fine-tuned)", base_params)
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

    with open("results_cnn.json", "w") as f:
        json.dump({"hosvd_energy_report": energy_report, "results": rows}, f, indent=2)
    print("\nSaved detailed results to results_cnn.json")

    return results, energy_report


if __name__ == "__main__":
    main()
