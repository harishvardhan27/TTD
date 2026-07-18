# Theoretical Error Bounds for TT-Compressed Networks

This document proves the error guarantees the rest of the project relies
on empirically: (1) how far TT-SVD's answer can be from the best
possible fixed-TT-rank approximation, (2) the analogous bound for
TT-rounding, and (3) how a per-layer weight error propagates to the
network's output — the bound that turns "reconstruction error" into an
actionable, per-layer rank-selection criterion.

Notation: for a matrix/tensor `A`, `||A||_F` is the Frobenius norm.
For `x ∈ R^n`, `||x||` is the Euclidean norm. `||A||_2` is the spectral
(operator) norm.

---

## 1. TT-SVD quasi-optimality (Oseledets, 2011, Thm 2.2)

**Setup.** Let `W` be a d-way tensor with mode sizes `n_1,...,n_d`. Fix
target TT-ranks `(r_1,...,r_{d-1})`. Let `W_best` be the *best possible*
tensor of TT-rank `≤ (r_1,...,r_{d-1})` approximating `W` in Frobenius
norm (this optimum exists since the set of bounded-TT-rank tensors is
closed), and let `ε_k` be the truncation error the TT-SVD algorithm's
k-th SVD step actually discards — i.e. `ε_k = ||Σ_{i>r_k} σ_i^{(k)}||`
where `σ_i^{(k)}` are the singular values of the k-th unfolding beyond
the kept rank `r_k`, so `ε_k` is exactly the norm of the truncated tail
at step k of `tt_svd()` in `tt_core.py`.

**Theorem (quasi-optimality).** The tensor `W_TT` produced by TT-SVD
satisfies

```
||W - W_TT||_F  ≤  sqrt( Σ_{k=1}^{d-1} ε_k^2 )
```

and moreover this is within a controlled factor of the best possible
approximation error at the same ranks:

```
||W - W_TT||_F  ≤  sqrt(d-1) · ||W - W_best||_F
```

**Proof sketch.**

*Step 1 (single-cut optimality of SVD).* For any single matrix unfolding
`M` and target rank `r`, the Eckart–Young theorem says truncated SVD
gives the *exactly* optimal rank-`r` approximation in Frobenius norm:
no other rank-`r` matrix is closer to `M`. So at each of the `d-1` cuts
TT-SVD makes, the local truncation is optimal *for that cut, given the
tensor handed to it after the previous cuts' truncations*.

*Step 2 (error accumulates additively across sequential cuts, in
quadrature).* TT-SVD does not truncate the original tensor `d-1` times
independently; it truncates sequentially, each time operating on the
*already-truncated* remainder from the previous step. A standard
induction (Oseledets 2011, using that each truncation step is an
orthogonal projection in the appropriate unfolding, and orthogonal
projections satisfy a Pythagorean error identity) shows the errors
introduced at each of the `d-1` steps are mutually orthogonal as
corrections to the original tensor, so their Frobenius-norm
contributions add in quadrature rather than linearly:

```
||W - W_TT||_F^2  =  Σ_{k=1}^{d-1} ε_k^2     (exactly, not just an upper bound)
```

*Step 3 (comparison to the best approximation).* Because each `ε_k` is
by definition the *smallest possible* residual achievable at that cut
(Eckart–Young again) for the tensor arriving at step `k`, and the best
overall approximation `W_best` in particular induces *some* residual at
every cut (not necessarily the smallest one achievable locally, since
`W_best`'s cores are jointly optimized rather than sequentially), each
`ε_k ≤ ||W - W_best||_F`. Summing the `d-1` such bounds in quadrature:

```
||W - W_TT||_F = sqrt(Σ ε_k^2) ≤ sqrt(d-1) · ||W - W_best||_F
```

**∎**

**What this means in practice:** TT-SVD is never worse than
`sqrt(d-1)` times the best-possible fixed-rank error — for the `d=3`
factorizations used throughout this project, that's a worst-case factor
of `sqrt(2) ≈ 1.41`. It is exact and optimal at `d=2` (ordinary SVD, no
gap at all). This is why refining a TT-SVD initialization further (via
the Riemannian retraction GD in `riemannian_tt.py`, or plain fine-tuning)
can help close that `sqrt(d-1)` gap in principle, though our own
measurement (`riemannian_demo.py`) found TT-SVD and Riemannian GD landing
at the *same* answer for our synthetic weight — i.e. the gap was already
negligible for that particular decaying-spectrum matrix.

---

## 2. TT-rounding error bound

**Theorem (Oseledets 2011, Thm 2.3 analogue).** If `X` is a TT tensor of
possibly-large rank and `tt_round(X, target_ranks)` truncates it down,
the same additive-in-quadrature bound holds relative to `X` itself:

```
||X - round(X)||_F  ≤  sqrt( Σ_k δ_k^2 )
```

where `δ_k` is the discarded singular-value tail at cut `k` during the
rounding sweep. Proof is identical in structure to Section 1's, since
TT-rounding is literally "orthogonalize, then apply the same sequential
truncated-SVD procedure" — the only difference from TT-SVD is that it
starts from an existing TT tensor rather than a dense one, which is why
`tt_round()` in `riemannian_tt.py` produced *exactly* the same
reconstruction error as direct `tt_svd()` in our unit test (both are
realizing the same optimal sequential-SVD procedure on the same
underlying tensor).

**Corollary used by `riemannian_tt_fit`:** since every iterate of the
rounding-retraction Riemannian gradient scheme is produced by exactly
this rounding procedure, every iterate satisfies this bound relative to
the (dense) gradient-step tensor it was retracting — i.e. the retraction
never introduces more error than the quasi-optimality theorem already
allows for, so the iterative refinement can only move *within* the same
quality envelope TT-SVD itself operates under, never outside it.

---

## 3. From weight-matrix error to network-output error

Reconstruction error on a single weight matrix is only useful insofar as
it tells you something about the *model's* output. This section derives
that connection.

**Setup.** Consider a single layer `y = σ(Wx + b)` where `σ` is an
elementwise nonlinearity with Lipschitz constant `L_σ` (e.g. `L_σ = 1`
for ReLU, `L_σ ≈ 1.13` for GELU — both effectively 1 for this bound's
purposes). Suppose the TT-compressed weight `W_TT` satisfies
`||W - W_TT||_2 ≤ ε` (spectral-norm bound; note `||A||_2 ≤ ||A||_F`, so
the Frobenius bound from Sections 1–2 automatically implies this with
the same `ε`).

**Lemma (single-layer output perturbation).** For any input `x`,

```
||σ(Wx+b) - σ(W_TT x+b)||  ≤  L_σ · ||W - W_TT||_2 · ||x||  ≤  L_σ · ε · ||x||
```

*Proof.* `||σ(Wx+b) - σ(W_TT x+b)|| ≤ L_σ ||Wx - W_TT x|| = L_σ ||(W-W_TT)x|| ≤ L_σ ||W-W_TT||_2 ||x||`,
using the Lipschitz property then the definition of spectral norm. **∎**

**Theorem (end-to-end propagation through a stack of `N` layers).** Let
`f = f_N ∘ ... ∘ f_1` be a network of `N` layers, each `f_i(z) =
σ_i(W_i z + b_i)` with Lipschitz constant `L_i = L_{σ_i} ||W_i||_2` for
the whole layer map (this is the standard per-layer Lipschitz bound used
in robustness analysis), and suppose layer `i`'s weight is TT-compressed
with spectral-norm error `ε_i`. Then for any input `x` with
`||f_{i}(...f_1(x))|| ≤ B` for all intermediate activations (a bounded-
activation assumption, reasonable after LayerNorm/BatchNorm):

```
||f(x) - f_TT(x)||  ≤  B · Σ_{i=1}^{N} ε_i · L_{σ_i} · Π_{j>i} L_j
```

*Proof idea.* Standard telescoping/composition argument: write
`f(x) - f_TT(x)` as a telescoping sum over which layers have been
"switched" from exact to compressed, one at a time; each term is bounded
by the single-layer lemma above times the Lipschitz constants of all the
*downstream* (compressed) layers the perturbation still has to pass
through, since perturbations occurring earlier get amplified/damped by
every subsequent layer's Lipschitz constant. This is the same style of
argument used to bound adversarial-robustness certificates layer-by-
layer (e.g. via the product of per-layer operator norms). **∎**

**What this means in practice, and how to use it:**

- If every layer used the *same* rank/error budget `ε` uniformly, the
  bound above says errors compound **multiplicatively** through depth
  (via the `Π_{j>i} L_j` term) — deeper networks need *tighter*
  per-layer rank budgets to hold a fixed end-to-end error target, all
  else equal. This is exactly why `demo_pipeline.py`'s 4-6 stacked
  blocks show worse post-compression relative error than a single
  isolated layer would (Section 1's `sqrt(d-1)` bound is a per-layer
  statement; this Section 3 bound is what tells you it doesn't stay
  small automatically once you stack layers).
- Layers appearing **later** in the network (small `Π_{j>i} L_j`, since
  fewer downstream layers remain to amplify their error) can usually
  tolerate **more aggressive compression** than early layers, all else
  equal — a concrete, provable justification for the "adaptive
  per-layer TT-rank selection" extension listed in the project brief,
  implementable directly using `hosvd_mode_singular_values` +
  `suggest_rank_for_energy` per layer, weighted by this bound's implied
  sensitivity.
- Because `L_σ ≈ 1` for the activations used here, the practically
  important term to control is `||W_i||_2` (equivalently, `Π_j ||W_j||_2`
  — the network's overall Lipschitz constant) together with the
  *absolute* per-layer error `ε_i`, not just the *relative* error
  reported elsewhere in this project's results tables. A layer with
  large operator norm deserves comparatively smaller **absolute**
  TT-SVD truncation error even if its *relative* error target is the
  same as other layers.

---

## References

- I. Oseledets, "Tensor-Train Decomposition," SIAM J. Sci. Comput., 2011
  (TT-SVD algorithm, Thm 2.2 quasi-optimality, Thm 2.3 rounding bound).
- A. Novikov et al., "Tensorizing Neural Networks," NeurIPS 2015 (TT
  compression of fully-connected layers, the direct basis for
  `TTLinear`/`from_linear` in this project).
- M. Steinlechner, "Riemannian Optimization for High-Dimensional Tensor
  Completion," SIAM J. Sci. Comput., 2016 (tangent-space-only, no-dense-
  reconstruction Riemannian TT optimization — the more scalable
  alternative to the rounding-retraction scheme implemented here).
- B. Vandereycken, "Low-Rank Matrix Completion by Riemannian
  Optimization," SIAM J. Optim., 2013 (the exact fixed-rank-matrix
  tangent-space projection and SVD retraction implemented in
  `riemannian_tt.py`'s Part A).
