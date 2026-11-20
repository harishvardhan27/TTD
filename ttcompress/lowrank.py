"""
lowrank.py
==========
Low-rank factorization of nn.Linear layers: the project's finalized method
(it beat 3-mode Tensor-Train on every real layer tested; see
FNN/REAL_MODEL_FINDINGS.md).

    W (out, in)  ~=  U (out, r) @ V (r, in)

Two ways to choose U, V:

1. Weight SVD ("plain"): truncated SVD of W. Optimal for ||W - W'||_F, i.e.
   it treats every input direction as equally important.

2. Data-aware SVD ("whitened", SVD-LLM / ASVD style): what matters is the
   layer's *output* error on real activations, ||(W - W') X||_F, not the
   weight error (THEORY_VERIFY/THEORY.md section 3 makes the same point).
   With C = E[x x^T] = L L^T (Cholesky), ||(W - W') X||_F^2 = N ||(W - W') L||_F^2,
   so the optimal rank-r W' is obtained from the truncated SVD of W L:
       W L = P S Q^T   ->   W' = P_r S_r Q_r^T L^{-1}
   Its output error is exactly the discarded tail of S, so per-layer output
   error is known for *every* rank from a single SVD.

Rank selection:
  * `rank_for_target_error` returns the smallest rank meeting a relative
    error target and *reports* when no rank below break-even can (the old
    HO/low_rank_compress.py silently fell back to rank 256, which made its
    "target_err<=0.15" and "<=0.30" rows the same uniform-rank-256 model).
  * `allocate_ranks` (greedy) and `allocate_ranks_dp` (exact multiple-choice
    knapsack on a discretised budget) distribute a global parameter budget to
    minimise sum_i w_i * relerr_i(r_i)^2. Each layer's error curve is convex
    and decreasing in r (sorted singular values), so greedy by marginal
    gain per parameter is optimal for the *continuous relaxation*; for the
    integer problem with unequal per-layer chunk costs it can leave up to one
    chunk's worth of budget unused per layer. The DP is exact up to the
    budget discretisation and is used to measure that gap, not assumed.

`LowRankLinear` is built from two plain nn.Linear modules so that
torch dynamic INT8 quantization and ONNX export handle it natively.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# Layer
# --------------------------------------------------------------------------

class LowRankLinear(nn.Module):
    """Drop-in nn.Linear replacement computing up(down(x)), W ~= U @ V."""

    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.down = nn.Linear(in_features, rank, bias=False)  # V: (r, in)
        self.up = nn.Linear(rank, out_features, bias=bias)    # U: (out, r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))

    # Convenience views ---------------------------------------------------
    @property
    def U(self) -> torch.Tensor:
        return self.up.weight

    @property
    def V(self) -> torch.Tensor:
        return self.down.weight

    @property
    def bias(self):
        return self.up.bias

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def compression_ratio(self) -> float:
        dense = self.in_features * self.out_features + (self.out_features if self.up.bias is not None else 0)
        return dense / self.num_parameters()

    def to_dense_weight(self) -> torch.Tensor:
        return self.up.weight @ self.down.weight

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, rank={self.rank}"

    @classmethod
    def from_factors(cls, U: torch.Tensor, V: torch.Tensor, bias: Optional[torch.Tensor]) -> "LowRankLinear":
        out_f, r = U.shape
        r2, in_f = V.shape
        assert r == r2
        layer = cls(in_f, out_f, r, bias=bias is not None)
        with torch.no_grad():
            layer.up.weight.copy_(U)
            layer.down.weight.copy_(V)
            if bias is not None:
                layer.up.bias.copy_(bias)
        return layer

    @classmethod
    def from_linear(cls, linear: nn.Linear, rank: int, cov: Optional[torch.Tensor] = None) -> "LowRankLinear":
        """Factorize `linear` at `rank`. With `cov` (E[x x^T] of the layer's
        input), uses data-aware SVD; otherwise plain weight SVD."""
        fac = factorize(linear.weight.detach(), cov)
        U, V = fac.factors(rank)
        b = linear.bias.detach() if linear.bias is not None else None
        return cls.from_factors(U.to(linear.weight.dtype), V.to(linear.weight.dtype), b)


def break_even_rank(in_features: int, out_features: int) -> int:
    """Largest rank at which the factorized layer is still strictly smaller
    than the dense one: r (m + n) < m n."""
    return max(0, (in_features * out_features - 1) // (in_features + out_features))


# --------------------------------------------------------------------------
# Factorization (one SVD per layer, any rank afterwards)
# --------------------------------------------------------------------------

@dataclass
class Factorization:
    """SVD of W (plain) or of W L (data-aware), kept so that factors and
    errors for any rank are available without re-decomposing."""

    P: torch.Tensor            # (out, k) left singular vectors
    S: torch.Tensor            # (k,) singular values, descending
    Qt: torch.Tensor           # (k, in) right singular vectors (transposed)
    L_inv: Optional[torch.Tensor]  # (in, in) inverse Cholesky factor, None for plain SVD
    total_energy: float = field(init=False)

    def __post_init__(self):
        self.total_energy = float((self.S.double() ** 2).sum())

    @property
    def data_aware(self) -> bool:
        return self.L_inv is not None

    def factors(self, rank: int):
        r = max(1, min(rank, self.S.numel()))
        U = self.P[:, :r] * self.S[:r]
        V = self.Qt[:r]
        if self.L_inv is not None:
            V = V @ self.L_inv
        return U.float(), V.float()

    def rel_error(self, rank: int) -> float:
        """Relative error of the rank-r approximation: weight Frobenius error
        for plain SVD, output error on the calibration data for data-aware."""
        if self.total_energy <= 0:
            return 0.0
        tail = float((self.S[rank:].double() ** 2).sum())
        return math.sqrt(tail / self.total_energy)

    def error_curve(self) -> torch.Tensor:
        """rel_error for every rank 0..k (vectorised)."""
        sq = self.S.double() ** 2
        tails = torch.flip(torch.cumsum(torch.flip(sq, [0]), 0), [0])
        tails = torch.cat([tails, torch.zeros(1, dtype=tails.dtype)])
        return (tails / max(self.total_energy, 1e-300)).sqrt()


def factorize(W: torch.Tensor, cov: Optional[torch.Tensor] = None, damp: float = 1e-2) -> Factorization:
    """SVD of W, or of W L where L L^T = cov + damp * mean(diag(cov)) * I.

    Damping keeps the Cholesky factor well conditioned when some input
    directions are (near) unused by the calibration data."""
    W = W.double()
    if cov is None:
        P, S, Qt = torch.linalg.svd(W, full_matrices=False)
        return Factorization(P, S, Qt, None)
    C = cov.double()
    n = C.shape[0]
    C = C + damp * C.diagonal().mean().clamp_min(1e-12) * torch.eye(n, dtype=C.dtype)
    L = torch.linalg.cholesky(C)
    L_inv = torch.linalg.solve_triangular(L, torch.eye(n, dtype=C.dtype), upper=False)
    P, S, Qt = torch.linalg.svd(W @ L, full_matrices=False)
    return Factorization(P, S, Qt, L_inv)


# --------------------------------------------------------------------------
# Error metrics that are comparable across methods
# --------------------------------------------------------------------------

def weight_rel_error(W: torch.Tensor, W_hat: torch.Tensor) -> float:
    """||W - W_hat||_F / ||W||_F."""
    W = W.double()
    return float((W - W_hat.double()).norm() / W.norm().clamp_min(1e-300))


def output_rel_error(W: torch.Tensor, W_hat: torch.Tensor, cov: torch.Tensor) -> float:
    """Relative output error on activations with second moment `cov`:

        sqrt( tr(D C D^T) / tr(W C W^T) ),  D = W - W_hat,

    which equals ||(W - W_hat) X||_F / ||W X||_F for the X that produced
    C = X X^T / N. Undamped, so it is the true error for *any* factorisation
    (plain or data-aware) and comparable between them. Pass a covariance
    estimated on held-out activations to get an unbiased estimate."""
    C = cov.double()
    D = W.double() - W_hat.double()
    num = float(torch.einsum("ij,jk,ik->", D, C, D))
    den = float(torch.einsum("ij,jk,ik->", W.double(), C, W.double()))
    return (max(num, 0.0) / max(den, 1e-300)) ** 0.5


# --------------------------------------------------------------------------
# Rank selection
# --------------------------------------------------------------------------

@dataclass
class RankChoice:
    rank: Optional[int]   # None => keep layer dense
    rel_error: float      # error predicted by the factorisation at that rank (0 if dense)
    met_target: bool
    reason: str = ""


def max_useful_rank(in_features: int, out_features: int, min_saving: float, granularity: int) -> int:
    """Largest rank (multiple of `granularity`) whose factorised layer saves at
    least `min_saving` of the dense parameters: r (m + n) <= (1 - min_saving) m n.
    Ranks above it cost an extra GEMM for (almost) no size benefit."""
    cap = int((1.0 - min_saving) * in_features * out_features // (in_features + out_features))
    cap = min(cap, break_even_rank(in_features, out_features))
    return (cap // granularity) * granularity


def rank_for_target_error(fac: Factorization, in_features: int, out_features: int,
                          target_rel_error: float, granularity: int = 8, min_saving: float = 0.0) -> RankChoice:
    """Smallest rank (multiple of `granularity`) whose relative error meets
    the target, provided the factorised layer still saves at least
    `min_saving` of its parameters. Otherwise the layer is kept dense and the
    miss is reported, never hidden (cf. audit finding C2)."""
    cap = max_useful_rank(in_features, out_features, min_saving, granularity)
    curve = fac.error_curve()
    for r in range(granularity, cap + 1, granularity):
        e = float(curve[min(r, len(curve) - 1)])
        if e <= target_rel_error:
            return RankChoice(r, e, True)
    e_cap = float(curve[min(max(cap, 0), len(curve) - 1)])
    return RankChoice(None, 0.0, False,
                      f"target {target_rel_error} not reachable at rank <= {cap} "
                      f"(error there = {e_cap:.3f}); kept dense")


def _chunk_gains(fac: Factorization, n_chunks: int, granularity: int) -> torch.Tensor:
    sq = (fac.S.double() ** 2) / max(fac.total_energy, 1e-300)
    pad = n_chunks * granularity - sq.numel()
    if pad > 0:
        sq = torch.cat([sq, torch.zeros(pad, dtype=sq.dtype)])
    return sq[: n_chunks * granularity].reshape(n_chunks, granularity).sum(1)


def allocate_ranks(facs: Dict[str, Factorization], shapes: Dict[str, tuple], param_budget: int,
                   granularity: int = 8, min_saving: float = 0.0,
                   weights: Optional[Dict[str, float]] = None) -> Dict[str, int]:
    """Greedy allocation of `param_budget` (sum over targeted layers of
    r_i (m_i + n_i); biases excluded) minimising sum_i w_i relerr_i(r_i)^2.

    Every layer first receives one chunk (rank = granularity), since a
    rank-0 layer would delete the layer. Remaining chunks are taken in order
    of decreasing (weighted error reduction / parameter cost). Layers are
    capped at `max_useful_rank`. All layers stay factorised; use
    `allocate_ranks_dp` when a dense option should be considered.

    Raises ValueError if the budget cannot cover one chunk per layer."""
    weights = weights or {}
    ranks, used, items = {}, 0, []
    for name, fac in facs.items():
        out_f, in_f = shapes[name]
        cost = granularity * (in_f + out_f)
        ranks[name] = granularity
        used += cost
        n_chunks = max_useful_rank(in_f, out_f, min_saving, granularity) // granularity
        gains = _chunk_gains(fac, max(n_chunks, 1), granularity) * weights.get(name, 1.0)
        for c in range(1, n_chunks):
            items.append((float(gains[c]) / cost, name, c, cost))
    if used > param_budget:
        raise ValueError(f"budget {param_budget:,} cannot cover the minimum rank {granularity} "
                         f"for every layer ({used:,} needed)")
    # per-layer gains are non-increasing, so a stable global sort keeps chunks contiguous
    items.sort(key=lambda t: -t[0])
    for _, name, c, cost in items:
        if c * granularity != ranks[name] or used + cost > param_budget:
            continue
        ranks[name] += granularity
        used += cost
    return ranks


def allocate_ranks_dp(facs: Dict[str, Factorization], shapes: Dict[str, tuple], param_budget: int,
                      granularity: int = 8, min_saving: float = 0.0, allow_dense: bool = True,
                      unit: Optional[int] = None, weights: Optional[Dict[str, float]] = None) -> Dict[str, Optional[int]]:
    """Exact minimisation of sum_i w_i relerr_i(r_i)^2 subject to the budget,
    as a multiple-choice knapsack solved by dynamic programming over a budget
    discretised in steps of `unit` parameters (costs rounded *up*, so the
    returned allocation is always feasible). Options per layer: ranks
    granularity, 2*granularity, ..., max_useful_rank, and (if `allow_dense`)
    the dense layer at cost m*n and zero error. Returns {name: rank or None}."""
    import numpy as np

    weights = weights or {}
    names = list(facs)
    unit = unit or max(1, param_budget // 4000)
    B = param_budget // unit
    INF = float("inf")
    dp = np.full(B + 1, INF)
    dp[0] = 0.0
    choice = []  # per layer: array (B+1,) of option index
    options_per_layer = []
    for name in names:
        out_f, in_f = shapes[name]
        cap = max_useful_rank(in_f, out_f, min_saving, granularity)
        curve = facs[name].error_curve()
        w = weights.get(name, 1.0)
        opts = []  # (rank or None, cost_units, err2)
        for r in range(granularity, cap + 1, granularity):
            e = float(curve[min(r, len(curve) - 1)])
            opts.append((r, -(-r * (in_f + out_f) // unit), w * e * e))
        if allow_dense or not opts:
            opts.append((None, -(-in_f * out_f // unit), 0.0))
        new = np.full(B + 1, INF)
        arg = np.full(B + 1, -1, dtype=np.int64)
        for k, (_, cu, e2) in enumerate(opts):
            if cu > B:
                continue
            cand = np.full(B + 1, INF)
            cand[cu:] = dp[: B + 1 - cu] + e2
            better = cand < new
            new[better] = cand[better]
            arg[better] = k
        dp = new
        choice.append(arg)
        options_per_layer.append(opts)
    if not np.isfinite(dp).any():
        raise ValueError("budget infeasible for the minimum rank of every layer")
    b = int(np.argmin(dp))
    result: Dict[str, Optional[int]] = {}
    for i in range(len(names) - 1, -1, -1):
        k = int(choice[i][b])
        r, cu, _ = options_per_layer[i][k]
        result[names[i]] = r
        b -= cu
    return result


# --------------------------------------------------------------------------
# Model-level API
# --------------------------------------------------------------------------

def attention_predicate(name: str, module: nn.Module) -> bool:
    """Attention projections (qkv / q,k,v / out proj) by name convention."""
    n = name.lower()
    is_attn = any(t in n for t in ("attn", "attention", "qkv", "q_proj", "k_proj", "v_proj", "out_proj"))
    is_mlp = any(t in n for t in ("mlp", "fc1", "fc2", "intermediate", "output.dense", "ffn"))
    return is_attn and not is_mlp


def qkv_predicate(name: str, module: nn.Module) -> bool:
    n = name.lower()
    return attention_predicate(name, module) and any(t in n for t in ("qkv", "q_proj", "k_proj", "v_proj"))


def attn_out_predicate(name: str, module: nn.Module) -> bool:
    return attention_predicate(name, module) and not qkv_predicate(name, module)


def mlp_predicate(name: str, module: nn.Module) -> bool:
    n = name.lower()
    return any(t in n for t in ("mlp", "fc1", "fc2", "intermediate", "ffn")) or n.endswith("output.dense")


def attn_and_mlp_predicate(name: str, module: nn.Module) -> bool:
    return attention_predicate(name, module) or mlp_predicate(name, module)


def all_linear_predicate(name: str, module: nn.Module) -> bool:
    """Every nn.Linear, including the classifier head."""
    return True


PREDICATES: Dict[str, Callable[[str, nn.Module], bool]] = {
    "qkv": qkv_predicate, "attn_out": attn_out_predicate, "attn": attention_predicate,
    "mlp": mlp_predicate, "attn+mlp": attn_and_mlp_predicate, "all": all_linear_predicate,
}


def find_linears(model: nn.Module, predicate: Callable[[str, nn.Module], bool]) -> List[str]:
    return [n for n, m in model.named_modules() if isinstance(m, nn.Linear) and predicate(n, m)]


def _resolve_parent(model: nn.Module, dotted: str):
    parts = dotted.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


@dataclass
class CompressionReport:
    layers: List[dict]
    dense_params: int          # targeted layers' weight params before
    compressed_params: int     # targeted layers' weight params after
    method: str = "plain"
    allocator: str = ""

    @property
    def layer_ratio(self) -> float:
        return self.dense_params / max(1, self.compressed_params)

    def summary(self) -> str:
        n_lr = sum(1 for l in self.layers if l["rank"] is not None)
        missed = [l for l in self.layers if not l.get("met_target", True)]
        s = (f"[{self.method}/{self.allocator}] {n_lr}/{len(self.layers)} layers factorised; targeted weights "
             f"{self.dense_params:,} -> {self.compressed_params:,} ({self.layer_ratio:.2f}x)")
        we = [l["weight_rel_error"] for l in self.layers if l["rank"] is not None]
        oe = [l["output_rel_error"] for l in self.layers if l["rank"] is not None and l.get("output_rel_error") is not None]
        if we:
            s += f"; weight err mean={sum(we)/len(we):.3f}"
        if oe:
            s += f"; held-out output err mean={sum(oe)/len(oe):.3f} max={max(oe):.3f}"
        if missed:
            s += f"; {len(missed)} layer(s) missed the target and were kept dense"
        return s


def compress_lowrank(
    model: nn.Module,
    predicate: Callable[[str, nn.Module], bool] = attention_predicate,
    *,
    covs: Optional[Dict[str, torch.Tensor]] = None,
    eval_covs: Optional[Dict[str, torch.Tensor]] = None,
    damp: float = 1e-2,
    target_rel_error: Optional[float] = None,
    budget_fraction: Optional[float] = None,
    rank: Optional[int] = None,
    ranks: Optional[Dict[str, Optional[int]]] = None,
    allocator: str = "greedy",
    granularity: int = 8,
    min_saving: float = 0.1,
    weights: Optional[Dict[str, float]] = None,
    verbose: bool = False,
) -> CompressionReport:
    """Replace targeted nn.Linear layers in `model` (in place) with
    LowRankLinear.

    Rank policy: exactly one of `target_rel_error` (per layer), `budget_fraction`
    (fraction of the targeted layers' weight params to keep, distributed by
    `allocator` = "greedy" | "dp"), `rank` (uniform) or `ranks` (explicit).

    `covs` switches factorisation to data-aware SVD (damping `damp`).
    `eval_covs` (covariances from *held-out* calibration images) are used
    only for reporting `output_rel_error`, identically for both methods.
    """
    assert sum(x is not None for x in (target_rel_error, budget_fraction, rank, ranks)) == 1
    names = find_linears(model, predicate)
    if not names:
        raise ValueError("predicate matched no nn.Linear layers")
    linears = {n: getattr(*_resolve_parent(model, n)) for n in names}
    shapes = {n: tuple(l.weight.shape) for n, l in linears.items()}

    facs: Dict[str, Factorization] = {}
    for n, lin in linears.items():
        cov = None
        if covs is not None:
            if n not in covs:
                raise KeyError(f"no calibration covariance for layer {n}")
            cov = covs[n]
        facs[n] = factorize(lin.weight.detach(), cov, damp=damp)

    dense_params = sum(o * i for o, i in shapes.values())
    choices: Dict[str, RankChoice] = {}
    if target_rel_error is not None:
        for n in names:
            o, i = shapes[n]
            choices[n] = rank_for_target_error(facs[n], i, o, target_rel_error, granularity, min_saving)
    elif budget_fraction is not None:
        budget = int(budget_fraction * dense_params)
        if allocator == "greedy":
            alloc = allocate_ranks(facs, shapes, budget, granularity, min_saving, weights=weights)
        elif allocator == "dp":
            alloc = allocate_ranks_dp(facs, shapes, budget, granularity, min_saving, weights=weights)
        elif allocator == "dp_nodense":  # ablation: exact allocation without the keep-dense option
            alloc = allocate_ranks_dp(facs, shapes, budget, granularity, min_saving, allow_dense=False, weights=weights)
        else:
            raise ValueError(allocator)
        for n, r in alloc.items():
            choices[n] = RankChoice(r, facs[n].rel_error(r) if r else 0.0, True)
    else:
        for n in names:
            o, i = shapes[n]
            r = ranks[n] if ranks is not None else rank
            if r is None or r >= break_even_rank(i, o):
                choices[n] = RankChoice(None, 0.0, r is None, "" if r is None else "rank >= break-even; kept dense")
            else:
                choices[n] = RankChoice(r, facs[n].rel_error(r), True)

    layers, compressed = [], 0
    for n in names:
        o, i = shapes[n]
        ch = choices[n]
        lin = linears[n]
        W = lin.weight.detach()
        entry = {"name": n, "shape": [o, i], "rank": ch.rank, "met_target": ch.met_target, "note": ch.reason,
                 "predicted_rel_error": round(ch.rel_error, 5)}
        if ch.rank is None:
            compressed += o * i
            entry.update(weight_rel_error=0.0, output_rel_error=0.0 if eval_covs is not None else None)
        else:
            U, V = facs[n].factors(ch.rank)
            W_hat = U @ V
            entry["weight_rel_error"] = round(weight_rel_error(W, W_hat), 5)
            entry["output_rel_error"] = (round(output_rel_error(W, W_hat, eval_covs[n]), 5)
                                         if eval_covs is not None and n in eval_covs else None)
            new = LowRankLinear.from_factors(U, V, lin.bias.detach() if lin.bias is not None else None)
            parent, attr = _resolve_parent(model, n)
            setattr(parent, attr, new)
            compressed += ch.rank * (o + i)
        layers.append(entry)
        if verbose:
            tag = f"rank {ch.rank:4d}" if ch.rank else "dense    "
            print(f"  [{'data' if covs is not None else 'plain'}] {n:40s} {o}x{i} -> {tag} "
                  f"w_err={entry['weight_rel_error']:.3f} out_err={entry['output_rel_error']} {ch.reason}")
    alloc_name = (allocator + ("+weighted" if weights else "")) if budget_fraction is not None else (
        "target" if target_rel_error is not None else "uniform" if rank is not None else "explicit")
    return CompressionReport(layers, dense_params, compressed, "data" if covs is not None else "plain", alloc_name)
