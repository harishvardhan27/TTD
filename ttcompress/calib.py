"""Calibration: collect per-layer input second moments E[x x^T] via forward hooks."""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List

import torch
import torch.nn as nn


@torch.no_grad()
def collect_input_covariances(model: nn.Module, layer_names: List[str], batches: Iterable[torch.Tensor],
                              forward: Callable[[nn.Module, torch.Tensor], object] = None) -> Dict[str, torch.Tensor]:
    """Run `batches` through `model` and return {layer_name: E[x x^T]} for
    the *inputs* of the named nn.Linear layers (every token / spatial
    position counts as one sample). Accumulates in float64."""
    modules = dict(model.named_modules())
    sums: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {n: 0 for n in layer_names}
    handles = []

    def make_hook(name):
        def hook(mod, inputs, output):
            x = inputs[0].detach()
            x = x.reshape(-1, x.shape[-1]).float()
            xtx = (x.T @ x).double()
            if name in sums:
                sums[name] += xtx
            else:
                sums[name] = xtx
            counts[name] += x.shape[0]
        return hook

    for n in layer_names:
        handles.append(modules[n].register_forward_hook(make_hook(n)))
    was_training = model.training
    model.eval()
    try:
        for xb in batches:
            (forward or (lambda m, x: m(x)))(model, xb)
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)
    return {n: sums[n] / counts[n] for n in layer_names if counts[n] > 0}
