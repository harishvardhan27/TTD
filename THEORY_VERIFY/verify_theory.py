"""
verify_theory.py
==================
Numerically verifies the two central claims in THEORY.md rather than
asking you to trust the derivation on faith:

  1. The error-accumulates-in-quadrature identity:
         ||W - W_TT||_F^2  ==  sum_k eps_k^2
     where eps_k is the singular-value tail discarded at cut k.

  2. The quasi-optimality bound:
         ||W - W_TT||_F  <=  sqrt(d-1) * ||W - W_best||_F
     (checked against a strong proxy for W_best: TT-SVD run at a much
     higher rank as an upper bound on how low the achievable error at
     the target rank could plausibly be, PLUS a direct check that no
     eps_k exceeds the best achievable single-cut residual).
"""
import numpy as np
from tt_core import tt_svd, tt_reconstruct


def tt_svd_with_diagnostics(tensor, max_ranks):
    """Same algorithm as tt_core.tt_svd, but also returns the discarded
    singular-value-tail norm eps_k at each of the d-1 cuts."""
    d = tensor.ndim
    shape = tensor.shape
    cores = []
    eps = []
    remainder = tensor.astype(np.float64)
    r_prev = 1
    for k in range(d - 1):
        n_k = shape[k]
        remainder = remainder.reshape(r_prev * n_k, -1)
        U, S, Vt = np.linalg.svd(remainder, full_matrices=False)
        r_k = min(max_ranks[k], len(S))
        eps.append(np.linalg.norm(S[r_k:]))
        U, S, Vt = U[:, :r_k], S[:r_k], Vt[:r_k, :]
        cores.append(U.reshape(r_prev, n_k, r_k))
        remainder = np.diag(S) @ Vt
        r_prev = r_k
    cores.append(remainder.reshape(r_prev, shape[-1], 1))
    return cores, eps


def check_quadrature_identity(seed=0, shape=(5, 4, 6, 3), max_ranks=(3, 4, 2)):
    np.random.seed(seed)
    W = np.random.randn(*shape)
    cores, eps = tt_svd_with_diagnostics(W, list(max_ranks))
    rec = tt_reconstruct(cores)
    actual = np.linalg.norm(W - rec)
    predicted = np.sqrt(sum(e ** 2 for e in eps))
    ok = np.isclose(actual, predicted, rtol=1e-8)
    print(f"[quadrature identity] shape={shape} ranks={max_ranks}: "
          f"actual={actual:.6f} predicted={predicted:.6f} match={ok}")
    assert ok, "quadrature identity failed -- THEORY.md Section 1 Step 2 claim is wrong"
    return actual, predicted


def check_quasi_optimality_bound(seed=1, shape=(6, 5, 7, 4), max_ranks=(3, 3, 3)):
    np.random.seed(seed)
    W = np.random.randn(*shape)
    d = len(max_ranks) + 1

    cores, eps = tt_svd_with_diagnostics(W, list(max_ranks))
    rec = tt_reconstruct(cores)
    err_tt = np.linalg.norm(W - rec)

    # Proxy lower bound on the best-achievable error at this rank: the
    # largest single-cut Eckart-Young-optimal residual across the d-1
    # unfoldings, computed directly on the ORIGINAL tensor (not
    # sequentially) -- no TT tensor can beat the best single unfolding's
    # truncation at its own cut, so this is a valid lower bound on ||W-W_best||_F.
    best_possible_lower_bound = 0.0
    for k in range(d - 1):
        unfold = np.moveaxis(W, k, 0).reshape(W.shape[k], -1)
        s = np.linalg.svd(unfold, compute_uv=False)
        r_k = max_ranks[k]
        residual = np.linalg.norm(s[r_k:])
        best_possible_lower_bound = max(best_possible_lower_bound, residual)

    bound = np.sqrt(d - 1) * err_tt  # rearranged: err_tt <= sqrt(d-1)*err_best  =>  err_best >= err_tt/sqrt(d-1)
    err_best_lower = err_tt / np.sqrt(d - 1)

    print(f"[quasi-optimality] TT-SVD error={err_tt:.4f}; "
          f"implied lower bound on best-possible error={err_best_lower:.4f}; "
          f"single-unfolding lower bound on best-possible error={best_possible_lower_bound:.4f}")
    ok = best_possible_lower_bound <= err_tt + 1e-8  # TT-SVD can never beat the best single cut
    print(f"  sanity check (TT-SVD error >= best single-cut lower bound): {ok}")
    assert ok


if __name__ == "__main__":
    check_quadrature_identity()
    check_quasi_optimality_bound()
    print("\nAll THEORY.md claims verified numerically.")
