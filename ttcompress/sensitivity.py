"""
Layer sensitivity weights for rank allocation.

The default allocation objective sum_i E_i(r_i) assumes a unit of relative
squared error costs the same in every layer. Dev results contradicted this
(EXPERIMENT_TRACKER E1, "H-iso"). Here each layer's weight is measured:

    w_i = KL( f(x) || f_i(x) ) / E_i(r_probe)

where f_i is the model with *only* layer i truncated to a probe rank and
E_i the layer's relative squared error at that rank. Under a first-order
model, KL(f || f_compressed) ~= sum_i w_i E_i(r_i), so allocating with these
weights minimises predicted output divergence. The first-order model ignores
interactions between layers; it is an approximation, evaluated empirically.

Images must be training-side (calibration hold-out); never evaluation data.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lowrank import Factorization, _resolve_parent, max_useful_rank


@torch.no_grad()
def layer_sensitivities(model: nn.Module, facs: Dict[str, Factorization], batches: Iterable[torch.Tensor],
                        probe_frac: float = 0.5, granularity: int = 8, min_saving: float = 0.1,
                        floor: float = 1e-6) -> Dict[str, dict]:
    """Return {layer: {"weight", "kl", "err2", "probe_rank"}}. `batches` is
    materialised once (normalised input tensors)."""
    model.eval()
    xs = [x for x in batches]
    base = [F.log_softmax(model(x).float(), 1) for x in xs]
    out = {}
    for name, fac in facs.items():
        lin = getattr(*_resolve_parent(model, name))
        o, i = lin.weight.shape
        cap = max_useful_rank(i, o, min_saving, granularity)
        r = max(granularity, int(probe_frac * cap) // granularity * granularity)
        U, V = fac.factors(r)
        W0 = lin.weight.data.clone()
        lin.weight.data.copy_((U @ V).to(W0.dtype))
        kl = 0.0
        n = 0
        for x, lb in zip(xs, base):
            lp = F.log_softmax(model(x).float(), 1)
            kl += float(F.kl_div(lp, lb, reduction="sum", log_target=True))
            n += len(x)
        lin.weight.data.copy_(W0)
        e2 = fac.rel_error(r) ** 2
        out[name] = {"kl": kl / n, "err2": e2, "probe_rank": r, "weight": max(kl / n, floor) / max(e2, 1e-12)}
    mean_w = sum(v["weight"] for v in out.values()) / len(out)
    for v in out.values():
        v["weight"] /= mean_w  # normalise to mean 1 (scale does not change the argmin)
    return out
