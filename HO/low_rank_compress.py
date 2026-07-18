"""
low_rank_compress.py
======================
The finalized project's compression method: plain low-rank (SVD)
factorization, targeted at attention/embedding layers where real
pretrained models were shown to have exploitable redundancy (see
REAL_MODEL_FINDINGS.md). Supersedes tt_core.py's Tensor-Train approach,
which was outperformed by this simpler method on every real layer tested.

W (out_features, in_features) ~= U @ V
    U: (out_features, rank)
    V: (rank, in_features)
Params: rank*(out_features + in_features) instead of out_features*in_features.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import torch
import torch.nn as nn


class LowRankLinear(nn.Module):
    """Drop-in nn.Linear replacement using W ~= U @ V."""

    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.U = nn.Parameter(torch.empty(out_features, rank))
        self.V = nn.Parameter(torch.empty(rank, in_features))
        nn.init.kaiming_uniform_(self.U, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.V, a=5 ** 0.5)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = (x @ self.V.T) @ self.U.T
        if self.bias is not None:
            out = out + self.bias
        return out

    def num_parameters(self) -> int:
        n = self.U.numel() + self.V.numel()
        if self.bias is not None:
            n += self.bias.numel()
        return n

    def compression_ratio(self) -> float:
        dense = self.in_features * self.out_features
        if self.bias is not None:
            dense += self.out_features
        return dense / self.num_parameters()

    @classmethod
    def from_linear(cls, linear: nn.Linear, rank: int) -> "LowRankLinear":
        U, S, Vt = torch.linalg.svd(linear.weight.detach(), full_matrices=False)
        r = min(rank, len(S))
        layer = cls(linear.in_features, linear.out_features, r, bias=linear.bias is not None)
        with torch.no_grad():
            layer.U.copy_(U[:, :r] * S[:r])
            layer.V.copy_(Vt[:r, :])
            if linear.bias is not None:
                layer.bias.copy_(linear.bias.detach())
        return layer

    def to_dense_weight(self) -> torch.Tensor:
        return self.U @ self.V


def suggest_rank_by_target_error(linear: nn.Linear, target_rel_error: float = 0.1,
                                  candidate_ranks: Sequence[int] = (4, 8, 16, 32, 64, 96, 128, 192, 256)) -> int:
    """Pick the smallest rank whose ACTUAL reconstruction error meets the
    target, measured directly from the real singular value spectrum
    (one SVD call, then just summing tails -- cheap, no re-decomposition
    needed per candidate, since plain SVD has no cross-cut dependency
    issue the way sequential TT-SVD cuts do)."""
    W = linear.weight.detach()
    S = torch.linalg.svdvals(W)
    total = (S ** 2).sum()
    for rank in candidate_ranks:
        r = min(rank, len(S))
        tail = (S[r:] ** 2).sum()
        rel_err = (tail / total).sqrt().item()
        if rel_err <= target_rel_error:
            return r
    return min(candidate_ranks[-1], len(S))


def default_predicate(name: str, module: nn.Linear) -> bool:
    """Target attention projections by name convention (qkv, q_proj,
    k_proj, v_proj, out_proj, proj) -- the layer type shown to have real
    exploitable redundancy. Skip MLP/FFN layers (fc1/fc2/mlp) since
    real testing showed those are nearly full-rank."""
    name_lower = name.lower()
    is_attn = any(tok in name_lower for tok in ["attn", "attention", "qkv", "q_proj", "k_proj", "v_proj", "out_proj"])
    is_mlp = any(tok in name_lower for tok in ["mlp", "fc1", "fc2", "intermediate", "output.dense"])
    return is_attn and not is_mlp


def replace_linear_with_lowrank(
    model: nn.Module,
    target_rel_error: float = 0.1,
    predicate: Optional[Callable[[str, nn.Linear], bool]] = None,
    verbose: bool = True,
) -> nn.Module:
    predicate = predicate or default_predicate

    targets: List[str] = [name for name, m in model.named_modules() if isinstance(m, nn.Linear) and predicate(name, m)]

    for name in targets:
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        linear: nn.Linear = getattr(parent, parts[-1])

        rank = suggest_rank_by_target_error(linear, target_rel_error)
        lr_layer = LowRankLinear.from_linear(linear, rank)
        setattr(parent, parts[-1], lr_layer)

        if verbose:
            dense_p = linear.in_features * linear.out_features + (linear.out_features if linear.bias is not None else 0)
            print(f"[low-rank] {name}: {linear.in_features}x{linear.out_features} ({dense_p:,} params) "
                  f"-> rank={rank} ({lr_layer.num_parameters():,} params, {lr_layer.compression_ratio():.2f}x)")

    return model
