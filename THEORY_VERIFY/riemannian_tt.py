"""
riemannian_tt.py
=================
Riemannian (manifold-constrained) optimization for low-rank / TT-rank
structures, as opposed to plain unconstrained ("Euclidean") gradient
descent directly on the core parameters.

Why this matters
-----------------
TT-format has a gauge redundancy: inserting any invertible r_k x r_k
matrix M and its inverse between cores k and k+1
    (G_k, G_{k+1}) -> (G_k M, M^{-1} G_{k+1})
leaves the represented tensor unchanged. Plain (Euclidean) SGD/Adam on
the raw core parameters is blind to this: it can spend steps moving
along the gauge orbit (which changes nothing about the represented
tensor) instead of making real progress, and the redundancy also makes
the loss landscape's conditioning worse than it needs to be. Riemannian
optimization works instead directly on the quotient manifold of
fixed-rank tensors, using:

  1. a tangent-space projection: turn an arbitrary ambient direction
     (e.g. a Euclidean gradient) into the "physically meaningful"
     direction of change of the *represented tensor*, discarding any
     component that only moves along the gauge orbit;
  2. a retraction: a map back onto the fixed-rank manifold after taking
     a step, since manifold + tangent-space step leaves the manifold.

Two levels are implemented here, in increasing generality:

  (A) The exact, classical fixed-rank-matrix case (Vandereycken, 2013).
      This is literally the d=2 special case of TT-format (a plain
      low-rank factorization), and has simple closed-form tangent-space
      projection and SVD-based retraction. Implemented and unit-tested
      exactly.

  (B) The general d-core TT case, using a *rounding retraction* scheme:
      take a full ambient gradient step on the reconstructed dense
      tensor, then re-project onto the fixed-TT-rank manifold via
      TT-SVD/TT-rounding. This is a standard, well-defined and provably
      rank-respecting retraction (Oseledets 2011's TT-rounding
      algorithm); it is the practical simplification used when the
      tensor is small enough to reconstruct densely each step (true for
      the individual weight matrices/kernels here). It differs from the
      most sample/FLOP-efficient version of Riemannian TT optimization
      (which restricts every operation to the tangent space and never
      forms the dense tensor -- see Steinlechner 2016 for that version,
      relevant at much larger scale than a single layer) but it is not
      an approximation of correctness: every iterate produced is an
      exact point on the fixed-TT-rank manifold.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from ttcompress.tt import tt_svd, tt_reconstruct


# ==========================================================================
# (A) Exact Riemannian optimization on the fixed-rank MATRIX manifold
# ==========================================================================

class FixedRankMatrix:
    """A point X = U @ diag(S) @ V.T on the manifold of rank-r matrices,
    stored in thin-SVD (orthonormal U, V) form, as required for the
    tangent-space projection formula below."""

    def __init__(self, U: np.ndarray, S: np.ndarray, V: np.ndarray):
        self.U, self.S, self.V = U, S, V  # U:(m,r), S:(r,), V:(n,r)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.U.shape[0], self.V.shape[0]

    def dense(self) -> np.ndarray:
        return (self.U * self.S) @ self.V.T

    @staticmethod
    def from_dense(X: np.ndarray, rank: int) -> "FixedRankMatrix":
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        r = min(rank, len(S))
        return FixedRankMatrix(U[:, :r], S[:r], Vt[:r, :].T)


def project_tangent_matrix(X: FixedRankMatrix, Z: np.ndarray) -> np.ndarray:
    """Orthogonal projection of an ambient direction Z (same shape as X)
    onto the tangent space of the fixed-rank-r matrix manifold at X.
    Standard formula (Vandereycken 2013, eq. 2.5):
        P_X(Z) = U U^T Z  +  Z V V^T  -  U U^T Z V V^T
    Returns a dense (m,n) matrix representing the tangent vector.
    """
    U, V = X.U, X.V
    UtZ = U.T @ Z          # (r, n)
    ZV = Z @ V              # (m, r)
    UtZV = UtZ @ V          # (r, r)
    return U @ UtZ + ZV @ V.T - U @ UtZV @ V.T


def riemannian_gradient_descent_matrix(
    target: np.ndarray, rank: int, steps: int = 100, lr: float = 0.5, init: FixedRankMatrix | None = None
) -> Tuple[FixedRankMatrix, List[float]]:
    """Minimize f(X) = 0.5*||X - target||_F^2 over the manifold of
    rank-`rank` matrices via exact Riemannian gradient descent with
    SVD retraction. Ambient gradient of f is simply (X - target); the
    Riemannian gradient is its tangent-space projection at X.
    """
    X = init or FixedRankMatrix.from_dense(np.random.randn(*target.shape) * target.std(), rank)
    losses = []
    for _ in range(steps):
        X_dense = X.dense()
        ambient_grad = X_dense - target
        loss = 0.5 * np.linalg.norm(ambient_grad) ** 2
        losses.append(loss)
        riemannian_grad = project_tangent_matrix(X, ambient_grad)
        step_dense = X_dense - lr * riemannian_grad
        X = FixedRankMatrix.from_dense(step_dense, rank)  # SVD retraction back onto the manifold
    return X, losses


# ==========================================================================
# (B) General TT case: rounding-retraction Riemannian gradient descent
# ==========================================================================

def tt_round(cores: Sequence[np.ndarray], max_ranks: Sequence[int]) -> List[np.ndarray]:
    """TT-rounding (Oseledets 2011, Algorithm 2): re-compress an existing
    TT representation to (at most) `max_ranks`, exactly, without ever
    forming the dense tensor. Two sweeps:
      1) right-to-left QR orthogonalization (makes cores 2..d right-orthonormal)
      2) left-to-right SVD truncation (enforces the target ranks)
    This is the retraction map used in (B) below when working truly
    tensor-train-natively (no dense reconstruction); `riemannian_tt_fit`
    uses the simpler dense-reconstruction retraction instead, since our
    layer-sized tensors comfortably fit in memory, but this function is
    provided and tested since it's the scalable, textbook retraction.
    """
    d = len(cores)
    cores = [c.copy() for c in cores]

    # Sweep 1: right-to-left QR orthogonalization.
    for k in range(d - 1, 0, -1):
        r_prev, n_k, r_k = cores[k].shape
        mat = cores[k].reshape(r_prev, n_k * r_k).T  # (n_k*r_k, r_prev)
        Q, R = np.linalg.qr(mat)
        new_r = Q.shape[1]
        cores[k] = Q.T.reshape(new_r, n_k, r_k)
        # absorb R into core k-1: core_{k-1} (r_{k-2}, n_{k-1}, r_prev) @ R.T (r_prev, new_r)
        r_pp, n_km1, _ = cores[k - 1].shape
        cores[k - 1] = (cores[k - 1].reshape(r_pp * n_km1, r_prev) @ R.T).reshape(r_pp, n_km1, new_r)

    # Sweep 2: left-to-right SVD truncation to the target ranks.
    for k in range(d - 1):
        r_prev, n_k, r_k = cores[k].shape
        mat = cores[k].reshape(r_prev * n_k, r_k)
        U, S, Vt = np.linalg.svd(mat, full_matrices=False)
        new_r = min(max_ranks[k], len(S))
        U, S, Vt = U[:, :new_r], S[:new_r], Vt[:new_r, :]
        cores[k] = U.reshape(r_prev, n_k, new_r)
        # absorb S @ Vt into next core
        carry = np.diag(S) @ Vt  # (new_r, r_k)
        r_next_prev, n_next, r_next = cores[k + 1].shape
        cores[k + 1] = (carry @ cores[k + 1].reshape(r_next_prev, n_next * r_next)).reshape(new_r, n_next, r_next)

    return cores


def riemannian_tt_fit(
    target: np.ndarray,
    max_ranks: Sequence[int],
    steps: int = 100,
    lr: float = 0.5,
    init_cores: List[np.ndarray] | None = None,
) -> Tuple[List[np.ndarray], List[float]]:
    """Minimize f(X) = 0.5*||X - target||_F^2 over the manifold of
    fixed-TT-rank tensors (rank <= max_ranks) via rounding-retraction
    Riemannian gradient descent. `target` is a d-way numpy tensor.
    Every iterate returned is an exact TT tensor of rank <= max_ranks
    (verified in tests/verify_riemannian.py).
    """
    if init_cores is None:
        init_cores = tt_svd(np.random.randn(*target.shape) * target.std(), max_ranks)
    cores = init_cores
    losses = []
    for _ in range(steps):
        X_dense = tt_reconstruct(cores)
        ambient_grad = X_dense - target
        loss = 0.5 * np.linalg.norm(ambient_grad) ** 2
        losses.append(loss)
        step_dense = X_dense - lr * ambient_grad
        cores = tt_svd(step_dense, max_ranks)  # retraction: re-project onto the fixed-TT-rank manifold
    return cores, losses
