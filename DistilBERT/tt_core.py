"""
tt_core.py
==========
Core mathematical machinery for Tensor-Train (TT) compression of dense
weight matrices, plus a drop-in PyTorch replacement for nn.Linear that
stores its weight as a chain of low-rank TT-cores instead of a dense
(out_features x in_features) matrix.

Math summary
------------
A dense weight matrix W of shape (out_features, in_features) is first
"tensorized": out_features is factorized into d integers m_1*m_2*...*m_d
and in_features into d integers n_1*n_2*...*n_d. W is reshaped/permuted
into a 2d-way tensor of shape (m_1, n_1, m_2, n_2, ..., m_d, n_d).

The TT-SVD algorithm (Oseledets, 2011) then unfolds this tensor
sequentially and applies truncated SVD at each step, producing a chain of
"cores" G_1, ..., G_d with

    G_k : shape (r_{k-1}, m_k, n_k, r_k),   r_0 = r_d = 1

such that

    W(i_1..i_d, j_1..j_d) ~= G_1[:, i_1, j_1, :] @ G_2[:, i_2, j_2, :] @ ... @ G_d[:, i_d, j_d, :]

The number of parameters in the TT representation is
    sum_k r_{k-1} * m_k * n_k * r_k
which is far smaller than m_1*n_1*...*m_d*n_d = out_features*in_features
whenever the TT-ranks r_k are kept small relative to the mode sizes.
HOSVD is used as the initialization step: the *per-mode* singular value
spectra (computed by unfolding the tensor along each mode independently)
are inspected to pick reasonable ranks/mode factorizations before running
TT-SVD, exactly mirroring how HOSVD is used to seed Tucker/TT
decompositions in practice.
"""

from __future__ import annotations

import math
import string
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# Mode factorization helpers
# --------------------------------------------------------------------------

def _prime_factors(n: int) -> List[int]:
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


def factorize_balanced(n: int, num_factors: int) -> Tuple[int, ...]:
    """Factorize integer n into `num_factors` roughly-balanced integers
    whose product equals n. Used to turn e.g. 3072 into (48, 64) or
    (16, 12, 16) for TT tensorization. Pads with 1s if n is prime-heavy.
    """
    if num_factors == 1:
        return (n,)

    primes = _prime_factors(n)
    # Greedily distribute prime factors into `num_factors` buckets,
    # always adding the next (largest-first) prime to the currently
    # smallest bucket -> balances the resulting products.
    buckets = [1] * num_factors
    for p in sorted(primes, reverse=True):
        idx = buckets.index(min(buckets))
        buckets[idx] *= p
    buckets.sort(reverse=True)
    assert math.prod(buckets) == n, "factorization must exactly reconstruct n"
    return tuple(buckets)


# --------------------------------------------------------------------------
# HOSVD utility: per-mode singular value spectra (used to pick ranks)
# --------------------------------------------------------------------------

def hosvd_mode_singular_values(tensor: np.ndarray) -> List[np.ndarray]:
    """Compute the singular values of every mode-k unfolding of `tensor`.
    This is the diagnostic step of Higher-Order SVD: it tells you, per
    mode, how much energy is captured by keeping the top-r singular
    values, which is exactly what you need to pick TT/Tucker ranks
    instead of guessing them blindly.
    """
    spectra = []
    ndim = tensor.ndim
    for mode in range(ndim):
        unfolded = np.moveaxis(tensor, mode, 0).reshape(tensor.shape[mode], -1)
        s = np.linalg.svd(unfolded, compute_uv=False)
        spectra.append(s)
    return spectra


def suggest_rank_for_energy(singular_values: np.ndarray, energy_keep: float = 0.99) -> int:
    """Smallest rank r such that the top-r singular values capture at
    least `energy_keep` fraction of the total squared energy."""
    sq = singular_values ** 2
    total = sq.sum()
    if total <= 0:
        return 1
    cum = np.cumsum(sq) / total
    r = int(np.searchsorted(cum, energy_keep) + 1)
    return max(1, min(r, len(singular_values)))


# --------------------------------------------------------------------------
# TT-SVD algorithm
# --------------------------------------------------------------------------

def tt_svd(tensor: np.ndarray, max_ranks: Sequence[int]) -> List[np.ndarray]:
    """Classic TT-SVD (Oseledets 2011). `tensor` has shape (n_1,...,n_d).
    `max_ranks` gives the ceiling on each internal bond rank r_1..r_{d-1}
    (length d-1). Returns a list of d cores, core_k with shape
    (r_{k-1}, n_k, r_k), r_0 = r_d = 1.
    """
    d = tensor.ndim
    shape = tensor.shape
    cores = []
    remainder = tensor.astype(np.float64)
    r_prev = 1
    for k in range(d - 1):
        n_k = shape[k]
        remainder = remainder.reshape(r_prev * n_k, -1)
        U, S, Vt = np.linalg.svd(remainder, full_matrices=False)
        r_k = min(max_ranks[k], len(S))
        U, S, Vt = U[:, :r_k], S[:r_k], Vt[:r_k, :]
        core = U.reshape(r_prev, n_k, r_k)
        cores.append(core)
        remainder = np.diag(S) @ Vt
        r_prev = r_k
    cores.append(remainder.reshape(r_prev, shape[-1], 1))
    return cores


def tt_reconstruct(cores: Sequence[np.ndarray]) -> np.ndarray:
    """Rebuild the full dense tensor from TT cores (for error-checking
    only -- never used inside the compressed forward pass)."""
    result = cores[0]
    for core in cores[1:]:
        result = np.tensordot(result, core, axes=([-1], [0]))
    return result.reshape([c.shape[1] for c in cores])


# --------------------------------------------------------------------------
# TTLinear: a nn.Linear drop-in whose weight lives in TT-format
# --------------------------------------------------------------------------

class TTLinear(nn.Module):
    """
    Tensor-Train compressed replacement for nn.Linear.

    The (out_features, in_features) weight is tensorized into d output
    modes (m_1..m_d) and d input modes (n_1..n_d), and stored as d cores
    of shape (r_{k-1}, m_k, n_k, r_k) instead of one dense matrix.

    Forward pass never reconstructs the full weight: it contracts the
    input against the cores one at a time via torch.einsum, so both the
    parameter count *and* the FLOPs of the matmul scale with the sum of
    core sizes rather than out_features * in_features.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        in_modes: Sequence[int],
        out_modes: Sequence[int],
        ranks: Sequence[int],
        bias: bool = True,
    ):
        super().__init__()
        assert math.prod(in_modes) == in_features
        assert math.prod(out_modes) == out_features
        assert len(in_modes) == len(out_modes)
        d = len(in_modes)
        assert len(ranks) == d - 1, "ranks holds only the d-1 *internal* bond ranks"

        self.in_features = in_features
        self.out_features = out_features
        self.in_modes = tuple(in_modes)
        self.out_modes = tuple(out_modes)
        full_ranks = [1, *ranks, 1]
        self.full_ranks = tuple(full_ranks)
        self.d = d

        self.cores = nn.ParameterList()
        for k in range(d):
            r_prev, r_next = full_ranks[k], full_ranks[k + 1]
            m_k, n_k = out_modes[k], in_modes[k]
            core = torch.empty(r_prev, m_k, n_k, r_next)
            nn.init.normal_(core, std=1.0 / math.sqrt(n_k * r_prev + 1e-8))
            self.cores.append(nn.Parameter(core))

        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        # Pre-build the einsum equation string once (depends only on d).
        self._einsum_eq = self._build_einsum_equation(d)

    @staticmethod
    def _build_einsum_equation(d: int) -> str:
        letters = string.ascii_lowercase
        in_letters = letters[0:d]          # a b c ...
        out_letters = letters[d:2 * d]     # (next d letters)
        rank_letters = letters[2 * d:2 * d + d + 1]
        batch = "z"

        x_term = batch + in_letters
        core_terms = []
        for k in range(d):
            core_terms.append(rank_letters[k] + out_letters[k] + in_letters[k] + rank_letters[k + 1])
        out_term = batch + out_letters  # trailing rank dims (size 1) dropped after squeeze
        eq = ",".join([x_term] + core_terms) + "->" + out_term + rank_letters[d]
        return eq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x = x.reshape(-1, self.in_features)
        batch = x.shape[0]
        x = x.reshape(batch, *self.in_modes)

        operands = [x] + list(self.cores)
        out = torch.einsum(self._einsum_eq, *operands)  # (batch, *out_modes, 1)
        out = out.reshape(batch, self.out_features)
        if self.bias is not None:
            out = out + self.bias
        out = out.reshape(*orig_shape[:-1], self.out_features)
        return out

    def num_parameters(self) -> int:
        n = sum(c.numel() for c in self.cores)
        if self.bias is not None:
            n += self.bias.numel()
        return n

    def compression_ratio(self) -> float:
        dense = self.in_features * self.out_features
        if self.bias is not None:
            dense += self.out_features
        return dense / self.num_parameters()

    # ---------------------------------------------------------------
    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        num_modes: int = 3,
        ranks: Sequence[int] | int = 8,
        in_modes: Sequence[int] | None = None,
        out_modes: Sequence[int] | None = None,
    ) -> "TTLinear":
        """Build a TTLinear that approximates an existing nn.Linear's
        weight via TT-SVD (HOSVD-informed rank selection under the hood).
        """
        in_features = linear.in_features
        out_features = linear.out_features

        in_modes = tuple(in_modes) if in_modes is not None else factorize_balanced(in_features, num_modes)
        out_modes = tuple(out_modes) if out_modes is not None else factorize_balanced(out_features, num_modes)
        d = len(in_modes)
        if isinstance(ranks, int):
            ranks = [ranks] * (d - 1)
        assert len(ranks) == d - 1

        W = linear.weight.detach().cpu().numpy().astype(np.float64)  # (out, in)

        # Reshape W -> (m_1,...,m_d, n_1,...,n_d) then interleave to
        # (m_1, n_1, m_2, n_2, ..., m_d, n_d) as required for TT-matrix format.
        W_t = W.reshape(*out_modes, *in_modes)
        perm = []
        for k in range(d):
            perm.append(k)          # m_k
            perm.append(d + k)      # n_k
        W_t = np.transpose(W_t, perm)  # (m_1, n_1, ..., m_d, n_d)

        # Collapse each (m_k, n_k) pair into one axis of size m_k*n_k so
        # ordinary TT-SVD (defined on a plain d-way tensor) applies directly.
        collapsed_shape = [out_modes[k] * in_modes[k] for k in range(d)]
        W_collapsed = W_t.reshape(collapsed_shape)

        cores_collapsed = tt_svd(W_collapsed, max_ranks=ranks)
        # TT-SVD may realize smaller ranks than requested (bounded by the
        # matrix dimensions at each unfolding step) -- build the layer with
        # the *actual* realized ranks so shapes always match.
        realized_ranks = [core.shape[-1] for core in cores_collapsed[:-1]]

        layer = cls(
            in_features=in_features,
            out_features=out_features,
            in_modes=in_modes,
            out_modes=out_modes,
            ranks=realized_ranks,
            bias=linear.bias is not None,
        )

        with torch.no_grad():
            for k, core in enumerate(cores_collapsed):
                r_prev, mn, r_next = core.shape
                m_k, n_k = out_modes[k], in_modes[k]
                core4 = core.reshape(r_prev, m_k, n_k, r_next)
                layer.cores[k].copy_(torch.from_numpy(core4).float())
            if linear.bias is not None:
                layer.bias.copy_(linear.bias.detach().clone())

        return layer

    def to_dense_weight(self) -> torch.Tensor:
        """Reconstruct the full (out_features, in_features) matrix -- for
        verification/debugging only; never used at inference time."""
        cores_np = [c.detach().cpu().numpy() for c in self.cores]
        collapsed = [c.reshape(c.shape[0], c.shape[1] * c.shape[2], c.shape[3]) for c in cores_np]
        W_collapsed = tt_reconstruct(collapsed)
        W_t = W_collapsed.reshape(*[dim for pair in zip(self.out_modes, self.in_modes) for dim in pair])
        # undo interleave -> (m_1..m_d, n_1..n_d)
        d = self.d
        perm = list(range(0, 2 * d, 2)) + list(range(1, 2 * d, 2))
        W = np.transpose(W_t, perm).reshape(self.out_features, self.in_features)
        return torch.from_numpy(W).float()
