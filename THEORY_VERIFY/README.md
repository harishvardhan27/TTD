# THEORY_VERIFY/ — TT theory checks, Riemannian demo, custom kernels (historical)

| File | Status |
|---|---|
| `verify_theory.py` | numerical checks of TT-SVD error identities — runs |
| `riemannian_tt.py`, `riemannian_demo.py` | **Correction (audit H6):** `riemannian_tt_fit` is projected gradient descent (Euclidean step + TT-SVD), not tangent-space Riemannian GD; on the reconstruction loss TT-SVD is already the quasi-optimal closed form, so agreement with it is expected rather than evidence. Part A (fixed-rank matrix manifold) is a correct Riemannian method. |
| `THEORY.md` | superseded by `docs/THEORY.md` (corrects audit H5) |
| `tt_linear_cpu.cpp`, `benchmark_kernel.py` | CPU C++ extension (MSVC); re-benchmarked under the new protocol in Phase 5 |
| `tt_linear_cuda.cu`, `build_cuda_kernel.py`, `verify_cuda_kernel_logic.py` | **Never compiled or run** (no NVIDIA GPU); shared-memory limit too small for the project's shapes (audit H9). Excluded from all results. |
| `KERNELS_README.md` | original kernel notes (sandbox numbers, single runs) |
