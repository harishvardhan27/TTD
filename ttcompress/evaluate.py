"""
Accuracy evaluation with uncertainty.

Small eval sets make accuracy differences between a compressed model and
its baseline easy to over-read (the old HO result reported a 2-3 point drop
on 100 images, i.e. 2-3 images, inside a +-5 point interval). Every number
here comes with:

* a Wilson 95% interval for each accuracy,
* a *paired* comparison against the baseline on the same images: the
  accuracy delta with a 95% interval (paired normal approximation) and an
  exact McNemar test p-value,
* top-1 agreement with the baseline's predictions (a lower-variance fidelity
  metric than accuracy itself) and mean KL(baseline || model).

Interpretation: the McNemar test asks whether the compressed model fixes and
breaks *different numbers* of images than chance would allow; it says nothing
about whether a difference matters in practice. A 0.3 pp drop can be
significant on 8,000 images and still be acceptable; report the interval of
the delta and judge practical significance against a stated tolerance.
Accuracies from different datasets are never pooled.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import batches


@torch.no_grad()
def predict_logits(model: nn.Module, x_uint8: torch.Tensor, data_cfg: dict, batch_size: int = 32) -> torch.Tensor:
    model.eval()
    return torch.cat([model(xb).float() for xb in batches(x_uint8, data_cfg, batch_size)])


def wilson_interval(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value; b, c are the discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


@dataclass
class EvalResult:
    n: int
    top1: float
    top1_ci: tuple
    top5: float
    delta_vs_base: Optional[float] = None
    delta_ci: Optional[tuple] = None
    mcnemar_p: Optional[float] = None
    agreement: Optional[float] = None
    kl_to_base: Optional[float] = None

    def to_dict(self):
        d = asdict(self)
        return {k: (round(v, 5) if isinstance(v, float) else ([round(t, 5) for t in v] if isinstance(v, tuple) else v))
                for k, v in d.items()}

    def short(self) -> str:
        s = f"top1={self.top1*100:.2f}% [{self.top1_ci[0]*100:.1f},{self.top1_ci[1]*100:.1f}] (n={self.n})"
        if self.delta_vs_base is not None:
            s += (f"  delta={self.delta_vs_base*100:+.2f}pp [{self.delta_ci[0]*100:+.1f},{self.delta_ci[1]*100:+.1f}]"
                  f" p={self.mcnemar_p:.3g} agree={self.agreement*100:.1f}% KL={self.kl_to_base:.4f}")
        return s


def score(logits: torch.Tensor, labels: torch.Tensor, base_logits: Optional[torch.Tensor] = None) -> EvalResult:
    n = len(labels)
    pred = logits.argmax(1)
    correct = pred == labels
    k = int(correct.sum())
    top5 = float((logits.topk(5, 1).indices == labels[:, None]).any(1).float().mean())
    res = EvalResult(n=n, top1=k / n, top1_ci=wilson_interval(k, n), top5=top5)
    if base_logits is not None:
        base_correct = base_logits.argmax(1) == labels
        d = correct.float() - base_correct.float()
        mean = float(d.mean())
        se = float(d.std(unbiased=True)) / math.sqrt(n) if n > 1 else 0.0
        b = int((base_correct & ~correct).sum())
        c = int((~base_correct & correct).sum())
        res.delta_vs_base = mean
        res.delta_ci = (mean - 1.96 * se, mean + 1.96 * se)
        res.mcnemar_p = mcnemar_exact_p(b, c)
        res.agreement = float((pred == base_logits.argmax(1)).float().mean())
        res.kl_to_base = float(F.kl_div(F.log_softmax(logits, 1), F.log_softmax(base_logits, 1),
                                        reduction="batchmean", log_target=True))
    return res


def per_class_accuracy(logits: torch.Tensor, labels: torch.Tensor, n_classes: int = 1000) -> torch.Tensor:
    """Per-class top-1 accuracy (NaN for classes absent from `labels`).
    With ImageNetV2-test (8 images/class) each entry has a +-35 pp interval,
    so use these as a distribution (e.g. how many classes lost >= 2 images),
    never as per-class claims."""
    correct = (logits.argmax(1) == labels).float()
    tot = torch.bincount(labels, minlength=n_classes).float()
    hit = torch.bincount(labels, weights=correct, minlength=n_classes)
    return torch.where(tot > 0, hit / tot.clamp_min(1), torch.full_like(tot, float("nan")))


def per_class_shift(logits: torch.Tensor, base_logits: torch.Tensor, labels: torch.Tensor) -> dict:
    """Distribution of per-class correct-count changes vs. the baseline."""
    c_new = (logits.argmax(1) == labels).long()
    c_base = (base_logits.argmax(1) == labels).long()
    delta = torch.bincount(labels, weights=(c_new - c_base).double(), minlength=1000)
    present = torch.bincount(labels, minlength=1000) > 0
    d = delta[present]
    return {"classes": int(present.sum()), "worse_by_2plus": int((d <= -2).sum()), "worse_by_1": int((d == -1).sum()),
            "unchanged": int((d == 0).sum()), "better": int((d > 0).sum()), "min_delta": int(d.min())}
