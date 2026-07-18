"""
compress_model.py
==================
Model-agnostic utilities for walking a PyTorch `nn.Module` (e.g. a
HuggingFace transformer) and replacing selected `nn.Linear` layers with
`TTLinear` layers, plus size/param reporting helpers.
"""

from __future__ import annotations

import copy
import io
from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from tt_core import TTLinear, factorize_balanced
from ttconv import TTConv2d


def default_conv_predicate(min_channels: int = 16) -> Callable[[str, nn.Conv2d], bool]:
    """Only TT-compress Conv2d layers with enough channels to be worth it
    (a 3x3x3->16 stem conv gains nothing and risks accuracy for no size win)."""

    def predicate(name: str, module: nn.Conv2d) -> bool:
        return (
            module.in_channels >= min_channels
            and module.out_channels >= min_channels
            and module.groups == 1
        )

    return predicate


def replace_conv_with_tt(
    model: nn.Module,
    num_modes: int = 2,
    rank: int | Sequence[int] = 16,
    predicate: Optional[Callable[[str, nn.Conv2d], bool]] = None,
    verbose: bool = True,
) -> nn.Module:
    """Walk `model` and replace every nn.Conv2d matching `predicate` with
    an equivalent TTConv2d initialized via TT-SVD of the pretrained
    kernel (using the im2col/unfold reduction to reuse the verified
    TTLinear machinery). Works on any nn.Module, including real CNN
    backbones (ResNet, MobileNet, VGG, etc.) since it only inspects
    named_modules(), not the model class.
    """
    predicate = predicate or default_conv_predicate()

    targets: List[str] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and predicate(name, module):
            targets.append(name)

    for name in targets:
        parent, attr = _resolve_parent(model, name)
        conv: nn.Conv2d = getattr(parent, attr)
        d = num_modes + 1  # +1 for the spatial (kH*kW) core
        ranks = [rank] * (d - 1) if isinstance(rank, int) else list(rank)
        tt_layer = TTConv2d.from_conv(conv, num_modes=num_modes, ranks=ranks)
        setattr(parent, attr, tt_layer)
        if verbose:
            kH, kW = conv.kernel_size
            orig_p = conv.in_channels * conv.out_channels * kH * kW + (
                conv.out_channels if conv.bias is not None else 0
            )
            print(
                f"[TT] {name}: Conv2d({conv.in_channels},{conv.out_channels},k={conv.kernel_size}) "
                f"({orig_p:,} params) -> TTConv2d ranks={ranks} "
                f"({tt_layer.num_parameters():,} params, {tt_layer.compression_ratio():.2f}x)"
            )

    return model


def default_predicate(min_features: int = 256) -> Callable[[str, nn.Linear], bool]:
    """Only TT-compress Linear layers big enough to be worth it (small
    layers like classification heads or biases-only projections gain
    nothing and just add overhead / accuracy risk)."""

    def predicate(name: str, module: nn.Linear) -> bool:
        return module.in_features >= min_features and module.out_features >= min_features

    return predicate


def replace_linear_with_tt(
    model: nn.Module,
    num_modes: int = 3,
    rank: int | Sequence[int] = 16,
    predicate: Optional[Callable[[str, nn.Linear], bool]] = None,
    verbose: bool = True,
) -> nn.Module:
    """Walk `model` and replace every nn.Linear matching `predicate` with
    an equivalent TTLinear initialized via TT-SVD of the pretrained
    weight. Works on any nn.Module, including a full HuggingFace
    `transformers` model (e.g. DistilBERT, GPT-2) since it only inspects
    named_modules()/named_children(), not the model class.

    Returns the same model object, mutated in place, for convenience.
    """
    predicate = predicate or default_predicate()

    # Build a list first: mutating named_modules() while iterating it is unsafe.
    targets: List[str] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and predicate(name, module):
            targets.append(name)

    for name in targets:
        parent, attr = _resolve_parent(model, name)
        linear: nn.Linear = getattr(parent, attr)
        in_modes = factorize_balanced(linear.in_features, num_modes)
        out_modes = factorize_balanced(linear.out_features, num_modes)
        d = len(in_modes)
        ranks = [rank] * (d - 1) if isinstance(rank, int) else list(rank)
        tt_layer = TTLinear.from_linear(
            linear, in_modes=in_modes, out_modes=out_modes, ranks=ranks
        )
        setattr(parent, attr, tt_layer)
        if verbose:
            orig_p = linear.in_features * linear.out_features + (
                linear.out_features if linear.bias is not None else 0
            )
            print(
                f"[TT] {name}: {linear.in_features}x{linear.out_features} "
                f"({orig_p:,} params) -> TT modes in={in_modes} out={out_modes} "
                f"ranks={ranks} ({tt_layer.num_parameters():,} params, "
                f"{tt_layer.compression_ratio():.2f}x)"
            )

    return model


def _resolve_parent(model: nn.Module, dotted_name: str):
    parts = dotted_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


# --------------------------------------------------------------------------
# Reporting helpers
# --------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model: nn.Module) -> float:
    """Serialize the state_dict to an in-memory buffer and measure bytes.
    This mirrors what `torch.save` would actually write to disk."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / (1024 ** 2)


@torch.no_grad()
def benchmark_latency(model: nn.Module, example_input, n_warmup: int = 3, n_iters: int = 15) -> Dict[str, float]:
    import time

    model.eval()
    for _ in range(n_warmup):
        model(example_input)
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        model(example_input)
        times.append(time.perf_counter() - t0)
    times = torch.tensor(times)
    return {
        "mean_ms": float(times.mean() * 1000),
        "p50_ms": float(times.median() * 1000),
        "p95_ms": float(times.quantile(0.95) * 1000),
        "throughput_samples_per_s": float(example_input.shape[0] / times.mean()),
    }
