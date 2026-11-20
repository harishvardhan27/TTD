# Mathematical Derivations

Every statement here maps to code; the mapping is given as
`module.function` and, where a test checks the statement numerically, as
`tests/…::test_name`. Nothing here is new mathematics; §2 follows the
whitening argument of SVD-LLM (Wang et al., 2024) and §5 is standard
separable resource allocation. See `LITERATURE_REVIEW.md` for attribution.

Notation. A linear layer computes `y = W x + b`, `W ∈ R^{m×n}` (out × in).
`X ∈ R^{n×N}` stacks N input vectors (every token of every calibration image
is one column). `‖·‖_F` Frobenius, `‖·‖_2` spectral norm, `σ_i(A)` singular
values in descending order.

---

## 1. Truncated SVD (weight-only, "plain")

**Theorem (Eckart–Young–Mirsky).** Let `W = U Σ Vᵀ`. For any `r < rank(W)`
the matrix `W_r = U_r Σ_r V_rᵀ` minimises `‖W − Ŵ‖` over all `Ŵ` with
`rank(Ŵ) ≤ r`, simultaneously for the Frobenius and spectral norms, with

    ‖W − W_r‖_F² = Σ_{i>r} σ_i²,        ‖W − W_r‖_2 = σ_{r+1}.

*Implementation.* `lowrank.factorize(W, cov=None)` computes the thin SVD in
float64; `Factorization.factors(r)` returns `U = U_r Σ_r` (`m×r`) and
`V = V_rᵀ` (`r×n`); `Factorization.rel_error(r) = sqrt(Σ_{i>r} σ_i² / Σ_i σ_i²)`.
Checked by `test_core.py::test_plain_lowrank_is_eckart_young_optimal`.

**Parameters.** Dense `mn` vs factorised `r(m+n)` (biases unchanged). The
factorised layer is smaller iff `r < mn/(m+n)`; the largest such integer is
the *break-even rank* `lowrank.break_even_rank`. `max_useful_rank` further
requires a saving of at least `s` (`min_saving`):
`r ≤ (1−s)·mn/(m+n)`.

**Limitation.** The objective weighs all input directions equally. A ViT's
layer inputs are highly anisotropic (LayerNorm outputs, post-GELU
activations), so the directions that weight-SVD keeps are not necessarily
the ones that carry signal. This motivates §2.

---

## 2. Activation-aware SVD ("data")

**Objective.** Minimise the layer's *output* error on real inputs:

    min_{rank(Ŵ) ≤ r}  ‖(W − Ŵ) X‖_F² .                                  (2.1)

**Reduction to an SVD.** Let `C = X Xᵀ / N` (uncentred second moment; the
mean direction matters because the error `(W−Ŵ)x` is applied to raw
inputs, not centred ones). For `D = W − Ŵ`,

    ‖D X‖_F² = tr(D X Xᵀ Dᵀ) = N · tr(D C Dᵀ).                              (2.2)

Assume `C ≻ 0` and let `C = L Lᵀ` (Cholesky, `L` lower-triangular,
invertible). Then `tr(D C Dᵀ) = ‖D L‖_F²`, so (2.1) is

    min_{rank(Ŵ) ≤ r}  ‖W L − Ŵ L‖_F² .

The map `Ŵ ↦ Z = Ŵ L` is a bijection on `{rank ≤ r}` because `L` is
invertible. Hence the problem is `min_{rank(Z) ≤ r} ‖W L − Z‖_F`, solved by
Eckart–Young: `Z* = P_r S_r Q_rᵀ` where `W L = P S Qᵀ`. Mapping back,

    Ŵ* = P_r S_r Q_rᵀ L⁻¹,   i.e.  U = P_r S_r,  V = Q_rᵀ L⁻¹,           (2.3)

with optimal value

    ‖(W − Ŵ*) X‖_F² / ‖W X‖_F² = Σ_{i>r} s_i² / Σ_i s_i² .                 (2.4)

Any square root of C works (e.g. `C^{1/2}`); Cholesky is used because it is
cheap and triangular solves are stable.

*Implementation.* `lowrank.factorize(W, cov)`: Cholesky of the damped
covariance (below), `L⁻¹` by triangular solve, SVD of `W L`, factors per
(2.3). `Factorization.rel_error(r)` is (2.4). Checked against a direct
measurement of `‖(W−Ŵ)X‖/‖WX‖` in
`test_core.py::test_data_aware_error_identity_and_beats_plain_on_output`
and `test_research.py::test_predicted_error_exact_only_without_damping…`.

### 2.1 Singular covariance and damping

`C` is singular if fewer than `n` linearly independent inputs are observed or
if some input direction is identically zero; then `L` is not invertible and
(2.1) has a non-unique minimiser (any `Ŵ` that differs on the null space of
`C` gives the same error). The implementation uses

    C_λ = C + λ μ I,   μ = tr(C)/n,                                         (2.5)

with `λ` = `damp` (default 1e-2, selected on dev, §5 of the protocol).
Then `tr(D C_λ Dᵀ) = tr(D C Dᵀ) + λ μ ‖D‖_F²`, so the damped problem is

    min_{rank(Ŵ) ≤ r}  ‖(W − Ŵ) X‖_F² / N  +  λ μ ‖W − Ŵ‖_F² ,            (2.6)

an explicit interpolation between activation-aware SVD (`λ → 0`) and plain
SVD (`λ → ∞`, since the second term dominates). This gives damping a
meaning beyond "numerical fix": it regularises toward the weight-space
solution in directions the calibration set did not explore.

**Equivalence with covariance shrinkage.** Write

    C + λμI = (1 + λ) · [ (1 − ρ) C + ρ μI ],    ρ = λ / (1 + λ).

Multiplying the covariance by a positive constant `κ` multiplies `W L` by
`√κ`, which leaves its singular vectors, the truncated factors (2.3) and all
*relative* errors unchanged. Hence damping by `λ` is **exactly** whitening
with the linear shrinkage estimator `(1−ρ)C + ρμI` (the form of the
Ledoit–Wolf estimator, with a fixed rather than data-estimated intensity):
`λ = 0.01 ↔ ρ ≈ 0.01`, `λ = 1 ↔ ρ = 0.5`, `λ = 3 ↔ ρ = 0.75`. Shrinkage
trades bias for variance in the covariance estimate, so the optimal `λ`
should *decrease* with more, and more diverse, calibration data. This is a
testable prediction (dev suites `dev2`/`dev3`, calibration-size and -source
ablations); it is an interpretation, not a theorem about accuracy.

**Consequence for reporting.** With `λ > 0`, (2.4) evaluated on the SVD of
`W L_λ` is the relative value of (2.6), *not* the output error. The code
therefore reports three distinct quantities per layer:

* `predicted_rel_error` — (2.4) for the factorisation actually used;
* `weight_rel_error` — `‖W − Ŵ‖_F / ‖W‖_F`;
* `output_rel_error` — `sqrt(tr(D C_ho Dᵀ) / tr(W C_ho Wᵀ))` with `C_ho`
  from **held-out** calibration images and **no damping**
  (`lowrank.output_rel_error`). This is the only number comparable between
  plain and data-aware factorisations (audit H2).

### 2.2 Conditioning

`κ(C_λ) ≤ (λ_max(C) + λμ) / (λμ) ≤ 1 + n/λ` (since `λ_max ≤ tr C = nμ`).
For `n = 1536` (DeiT-S fc2 input) and `λ = 1e-2`, `κ ≤ 1.5·10⁵`, comfortably
inside float64 (unit roundoff 1.1e−16): the whitening loses at most ~5
significant digits. All factorisation arithmetic is float64; factors are
cast to float32 at the end. With `λ = 0` conditioning depends on the data
and can be arbitrarily bad; this is ablated (A9).

### 2.3 Estimating C

* Accumulated by forward hooks on the *inputs* of the targeted layers
  (`calib.collect_input_covariances`), `C = Σ x xᵀ / N`, float64 sums.
* Inputs of later layers are those of the **uncompressed** model (all
  covariances are collected once from the pretrained model). Errors made in
  earlier layers therefore change later layers' true input distribution;
  sequential (re-calibrated) compression would address this and is not done
  here (limitation; documented in THEORY.md).
* Sample size: one ViT image contributes 197 tokens, so `n = 256` images
  give `N = 50,432` vectors for `n ≤ 1536` (DeiT-S) or `≤ 3072` (ViT-B)
  dimensions, i.e. `N/n ≥ 16`. Tokens of one image are correlated, so the
  effective sample size is smaller; this is measured empirically by the
  calibration-size ablation (A8) and by the gap between `predicted_rel_error`
  (fit set) and `output_rel_error` (held-out set).

---

## 3. Error propagation (what per-layer errors do and do not tell us)

For one layer and any input `x`, `‖(W − Ŵ) x‖ ≤ ‖W − Ŵ‖_2 ‖x‖`. Composing
layers, the output perturbation of a deep network is bounded by sums of such
terms multiplied by downstream Lipschitz constants. For pretrained ViTs this
bound is *vacuous*: per-layer spectral norms are ≫ 1, and LayerNorm,
softmax-attention and residual connections make the global Lipschitz constant
astronomically larger than observed sensitivities (the old `THEORY.md` §3
presented this bound as a rank-selection criterion; audit H5). Consequently
**no theorem in this project links the per-layer objective to accuracy**; the
link is established empirically (correlation of summed layer errors with
accuracy loss across the grid, reported in the analysis).

---

## 4. Tensor-Train baseline (for completeness)

TT-SVD (Oseledets, 2011) of a d-way tensor with sequential truncations
`ε_k` (Frobenius norm of the singular values discarded at step k, computed on
the *remainder* passed to step k) satisfies

    ‖W − W_TT‖_F² = Σ_k ε_k²         (exact, because the successive
                                       corrections are mutually orthogonal),

and, defining instead `δ_k` as the best rank-`r_k` error of the k-th
unfolding of the *original* tensor, `ε_k ≤ δ_k` and
`‖W − W_TT‖_F ≤ sqrt(Σ_k δ_k²) ≤ sqrt(d−1) · ‖W − W_best‖_F`
(Oseledets 2011, Thm 2.2). The old document mixed the two definitions; the
`√(d−1)` bound needs the `δ_k` version. For `d = 2` TT is exactly the plain
low-rank factorisation of §1.

For a weight matrix reshaped into `d ≥ 3` modes there is no guarantee that
TT at a given parameter count beats §1; that is an empirical question (B6).

---

## 5. Global rank allocation

**Problem.** Layers `i = 1..L` with shapes `m_i × n_i`, factorisation
spectra `s_{i,1} ≥ s_{i,2} ≥ …`, normalised energies
`e_{i,j} = s_{i,j}² / Σ_j s_{i,j}²`. Choose ranks `r_i` (multiples of a
granularity `g`, `g ≤ r_i ≤ R_i = max_useful_rank`) to

    minimise  F(r) = Σ_i w_i E_i(r_i),   E_i(r) = Σ_{j>r} e_{i,j}
    subject to  Σ_i r_i (m_i + n_i) ≤ B.                                   (5.1)

`E_i` is the relative squared error (2.4) of layer `i`; `w_i` are optional
sensitivity weights (default 1; non-uniform weights are not used in the
reported experiments because no validated sensitivity estimate exists here).

**Structure.** Group `e_{i,·}` into chunks of `g` singular values:
`G_{i,c} = Σ_{j ∈ chunk c} e_{i,j}`, each costing `g(m_i+n_i)` parameters.
Because the singular values are sorted, `G_{i,c}` is non-increasing in `c`,
so `E_i` is a convex, decreasing, piecewise-linear function of the
parameters spent on layer `i`.

**Greedy** (`lowrank.allocate_ranks`): give every layer one chunk, then add
chunks in decreasing order of `w_i G_{i,c} / (g(m_i+n_i))`, skipping any
chunk that does not fit the remaining budget.

*Guarantee.* Let `F_frac` be the optimum of the continuous relaxation
(fractional chunks allowed) and `F_int` the integer optimum. For a separable
problem whose per-layer items have non-increasing value/cost ratios, the
relaxation is a fractional knapsack: its optimum takes items in global
ratio order and splits at most one item, `k*`. The greedy solution contains
every item preceding `k*` in that order, hence

    F_greedy − F_int  ≤  F_greedy − F_frac  ≤  w_{k*} G_{k*}  ≤  max_{i,c} w_i G_{i,c}.   (5.2)

The greedy solution is therefore optimal up to one chunk's worth of
normalised error. It is **not** claimed to be globally optimal for (5.1).

**Exact DP** (`lowrank.allocate_ranks_dp`): (5.1) is a multiple-choice
knapsack. With costs rounded up to a unit `u`, a DP over `B/u` budget cells
solves it exactly in `O(L · K · B/u)` time, `K` = options per layer
(optionally including "keep dense" at cost `m_i n_i` and zero error). Rounding
costs up keeps every solution feasible; the result is exact for the
discretised problem and exact for (5.1) when `u = 1`
(`test_research.py::test_dp_is_exact_and_greedy_is_feasible` checks both
against brute force).

**Limitations.**
1. The objective sums *relative* layer errors with equal weight. This assumes
   every layer's relative error costs accuracy equally, which §3 shows we
   cannot prove; the A2 ablation (uniform vs adaptive ranks) tests whether
   the assumption helps in practice.
2. Errors are measured against the uncompressed model's activations
   (§2.3); interactions between layers are ignored.
3. Allocation stability across calibration sets is measured (A12: seeds;
   A8: source), not assumed.

---

## 6. Quantization and its interaction with factorisation

Dynamic INT8 (torch `quantize_dynamic`, ORT `quantize_dynamic`): weights are
quantized once with symmetric max-abs scaling (`q = round(w/Δ)`,
`Δ = max|w|/127`, per tensor or per output channel); activations are
quantized at run time per tensor. The weight rounding error per entry is at
most `Δ/2`.

*Order of operations.* Let `Q(·)` denote weight fake-quantization
(`quant.fake_quant_weight`). The two pipelines compared are

* **FQ**: `Ŵ = Q(U) Q(V)` with `(U,V)` from SVD of `W` (factorise, then the
  INT8 backend quantizes each factor);
* **QF**: `(U,V)` from SVD of `Q(W)`, then `Ŵ = Q(U) Q(V)`.

They do not commute in general: `Q` is non-linear, and the SVD of `Q(W)`
differs from that of `W` by a perturbation of norm `≤ ‖W − Q(W)‖_F`
(Weyl / Wedin bounds the change of singular values/subspaces). Since INT8
rounding error is typically much smaller than the truncation error at the
ranks used here, the difference is expected to be small; it is measured, not
assumed (core grid, `*_qf` variants).

*Error composition (FQ).* With `U' = U + E_U`, `V' = V + E_V`,
`W − U'V' = (W − UV) − E_U V − U E_V − E_U E_V`, so to first order the
quantization error of a low-rank layer adds `‖E_U V‖ + ‖U E_V‖` to the
truncation error. Note that `U = P_r S_r` carries the singular values, so its
per-channel scales are dominated by the largest singular values — a reason
per-channel quantization matters more for factorised layers than for dense
ones (tested: ORT `per_channel=True` is the default; A5/A6 compare dense and
low-rank INT8).
