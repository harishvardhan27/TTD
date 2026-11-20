"""
surgery.py (formerly compress_model.py, previously copied into CNN/,
DistilBERT/ and FNN/)
==================
Model-agnostic utilities for walking a PyTorch `nn.Module` and replacing
selected `nn.Linear` / `nn.Conv2d` layers with Tensor-Train layers
(`TTLinear`, `TTConv2d`). The low-rank equivalents live in lowrank.py.
"""

from __future__ import annotations

import copy
import io
from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from .tt import TTLinear, factorize_balanced
from .ttconv import TTConv2d


def suggest_conv_rank_by_target_error(
    conv: nn.Conv2d, num_modes: int = 2, target_rel_error: float = 0.15,
    candidate_ranks: Sequence[int] = (8, 32, 128, 384),
) -> int:
    """Pick the smallest TT-rank (from a small geometrically-spaced
    candidate list, for speed across a real many-layer CNN) whose ACTUAL
    weight-reconstruction relative Frobenius error is <= target_rel_error,
    measured directly by really running TT-SVD at each candidate --
    not via the HOSVD per-mode-energy heuristic (see
    suggest_conv_rank_by_energy's docstring: that proxy is only exact for
    the FIRST of TT-SVD's sequential cuts and was found empirically to
    under-suggest rank badly enough to zero out real classification
    accuracy). We also avoid a tempting "compute the SVD spectra once at
    a generous cap, then predict smaller ranks' error from the same
    spectra" shortcut: since each cut's remainder depends on the actual
    truncation chosen at the previous cut, that shortcut has the same
    kind of cross-cut blind spot we just found and fixed -- so this
    function really re-runs TT-SVD at each of the few candidates instead.
    """
    from .ttconv import TTConv2d

    for rank in candidate_ranks:
        d = num_modes + 1
        ttc = TTConv2d.from_conv(conv, num_modes=num_modes, ranks=[rank] * (d - 1))
        W_dense = ttc.tt.to_dense_weight()
        W_flat = conv.weight.detach().reshape(conv.out_channels, -1)
        rel_err = (W_flat - W_dense).norm().item() / (W_flat.norm().item() + 1e-12)
        if rel_err <= target_rel_error:
            return rank
    return candidate_ranks[-1]


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


def suggest_conv_rank_by_energy(conv: nn.Conv2d, num_modes: int = 2, energy_keep: float = 0.9) -> int:
    """HOSVD-informed per-layer rank suggestion: unfold the (tensorized)
    kernel and pick the rank that retains `energy_keep` fraction of the
    dominant mode's singular-value energy, instead of using one uniform
    rank for every layer regardless of how compressible it actually is.

    CAVEAT (found empirically -- see suggest_conv_rank_by_target_error's
    docstring): this heuristic is only exact for the FIRST of TT-SVD's
    sequential cuts and can substantially UNDER-suggest the rank needed
    for later cuts on real trained conv weights, to the point of
    catastrophic accuracy loss if trusted blindly. Prefer
    suggest_conv_rank_by_target_error, which measures actual
    reconstruction error directly instead of relying on this proxy.
    Kept here for the diagnostic value of comparing the two."""
    import numpy as np
    from .tt import hosvd_mode_singular_values, suggest_rank_for_energy

    W = conv.weight.detach().numpy()
    C_out, C_in, kH, kW = W.shape
    out_modes = factorize_balanced(C_out, num_modes)
    in_modes = (*factorize_balanced(C_in, num_modes), kH * kW)
    W_t = W.reshape(*out_modes, *in_modes)
    spectra = hosvd_mode_singular_values(W_t)
    ranks = [suggest_rank_for_energy(s, energy_keep) for s, dim in zip(spectra, W_t.shape) if dim > 1]
    if not ranks:
        ranks = [1]
    return max(4, min(ranks))


def replace_conv_with_tt_adaptive(
    model: nn.Module,
    num_modes: int = 2,
    target_rel_error: float = 0.15,
    predicate: Optional[Callable[[str, nn.Conv2d], bool]] = None,
    verbose: bool = True,
) -> nn.Module:
    """Like replace_conv_with_tt, but picks each layer's TT-rank
    individually so its ACTUAL reconstruction error meets
    `target_rel_error`, instead of using one global rank for every layer.
    This is the "adaptive per-layer rank selection" extension from the
    project brief. Matters a lot in practice: real conv layers vary
    enormously in how compressible they are (verified directly on real
    MobileNetV2/ResNet34 weights), so a single global rank is either
    wasteful on easy layers or catastrophic on hard ones. Uses
    suggest_conv_rank_by_target_error (measured directly), not the
    HOSVD-energy heuristic (which we found is unreliable for this)."""
    predicate = predicate or default_conv_predicate()

    targets: List[str] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and predicate(name, module):
            targets.append(name)

    for name in targets:
        parent, attr = resolve_parent(model, name)
        conv: nn.Conv2d = getattr(parent, attr)
        rank = suggest_conv_rank_by_target_error(conv, num_modes=num_modes, target_rel_error=target_rel_error)
        d = num_modes + 1
        ranks = [rank] * (d - 1)
        tt_layer = TTConv2d.from_conv(conv, num_modes=num_modes, ranks=ranks)
        setattr(parent, attr, tt_layer)
        if verbose:
            kH, kW = conv.kernel_size
            orig_p = conv.in_channels * conv.out_channels * kH * kW + (
                conv.out_channels if conv.bias is not None else 0
            )
            print(
                f"[TT-adaptive] {name}: Conv2d({conv.in_channels},{conv.out_channels},k={conv.kernel_size}) "
                f"({orig_p:,} params) -> rank={rank} (target err<={target_rel_error}) "
                f"({tt_layer.num_parameters():,} params, {tt_layer.compression_ratio():.2f}x)"
            )

    return model


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
        parent, attr = resolve_parent(model, name)
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
        parent, attr = resolve_parent(model, name)
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


def resolve_parent(model: nn.Module, dotted_name: str):
    parts = dotted_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


# --------------------------------------------------------------------------
# Reporting helpers (single implementation lives in bench.py)
# --------------------------------------------------------------------------

from .bench import benchmark_latency, count_parameters, model_size_mb  # noqa: E402,F401


_resolve_parent = resolve_parent  # backwards-compatible alias
