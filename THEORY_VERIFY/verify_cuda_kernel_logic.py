"""
verify_cuda_kernel_logic.py
=============================
tt_linear_cuda.cu cannot be compiled or run in this sandbox (no GPU, no
nvcc -- verified: no `nvidia-smi`, no `nvcc`, torch.cuda.is_available()
is False). Rather than ship untested CUDA source and hope, this script
simulates the kernel's exact index arithmetic in pure Python/NumPy
(mirroring every loop and index expression in tt_linear_forward_kernel
line-for-line) and checks it against TTLinear's already-extensively-
verified forward pass. This confirms the ALGORITHM the CUDA kernel
implements is correct; compiling the .cu file on real GPU hardware is
still required to confirm the CUDA syntax itself builds cleanly and to
get real performance numbers (see kernels/build_cuda_kernel.py).

(Bug caught by this process during development: an earlier check
appeared to fail with ~25% relative error, which turned out to be an
unaccounted-for bias term in the comparison, not an actual indexing bug
-- fixed by disabling bias in the reference nn.Linear used for the test,
matching what the CUDA kernel itself does not handle (bias is added
separately in the Python wrapper, same as the CPU kernel).)
"""

import numpy as np
import torch
import torch.nn as nn

from tt_core import TTLinear


def simulate_cuda_kernel(x_b, core0, core1, core2, n1, n2, n3, m1, m2, m3, r1, r2):
    """Line-for-line simulation of tt_linear_forward_kernel's three
    contraction steps and index arithmetic (see kernels/tt_linear_cuda.cu)."""
    state_a = np.zeros((m1, n2, n3, r1), dtype=np.float32)
    for m1_ in range(m1):
        for n2_ in range(n2):
            for n3_ in range(n3):
                for r1_ in range(r1):
                    acc = 0.0
                    for n1_ in range(n1):
                        x_val = x_b[(n1_ * n2 + n2_) * n3 + n3_]
                        core_val = core0[m1_, n1_, r1_]
                        acc += x_val * core_val
                    state_a[m1_, n2_, n3_, r1_] = acc

    state_b = np.zeros((m1, m2, n3, r2), dtype=np.float32)
    for m1_ in range(m1):
        for m2_ in range(m2):
            for n3_ in range(n3):
                for r2_ in range(r2):
                    acc = 0.0
                    for r1_ in range(r1):
                        for n2_ in range(n2):
                            sa = state_a[m1_, n2_, n3_, r1_]
                            c1 = core1[r1_, m2_, n2_, r2_]
                            acc += sa * c1
                    state_b[m1_, m2_, n3_, r2_] = acc

    out_b = np.zeros((m1, m2, m3), dtype=np.float32)
    for m1_ in range(m1):
        for m2_ in range(m2):
            for m3_ in range(m3):
                acc = 0.0
                for r2_ in range(r2):
                    for n3_ in range(n3):
                        sb = state_b[m1_, m2_, n3_, r2_]
                        c2 = core2[r2_, m3_, n3_]
                        acc += sb * c2
                out_b[m1_, m2_, m3_] = acc
    return out_b.reshape(-1)


def main():
    torch.manual_seed(0)
    n1, n2, n3 = 2, 3, 2
    m1, m2, m3 = 4, 2, 3
    lin = nn.Linear(n1 * n2 * n3, m1 * m2 * m3, bias=False)  # kernel handles bias separately
    ttl = TTLinear.from_linear(lin, in_modes=(n1, n2, n3), out_modes=(m1, m2, m3), ranks=[3, 5])
    r1, r2 = ttl.full_ranks[1], ttl.full_ranks[2]

    core0 = ttl.cores[0].detach().numpy().squeeze(0)   # (m1, n1, r1)
    core1 = ttl.cores[1].detach().numpy()               # (r1, m2, n2, r2)
    core2 = ttl.cores[2].detach().numpy().squeeze(-1)   # (r2, m3, n3)

    x = np.random.randn(5, n1 * n2 * n3).astype(np.float32)
    out_sim = np.stack([
        simulate_cuda_kernel(x[i], core0, core1, core2, n1, n2, n3, m1, m2, m3, r1, r2)
        for i in range(x.shape[0])
    ])
    out_ref = ttl(torch.from_numpy(x)).detach().numpy()

    rel_err = np.linalg.norm(out_sim - out_ref) / np.linalg.norm(out_ref)
    print(f"CUDA kernel index-logic simulation vs verified TTLinear: relative error = {rel_err:.2e}")
    assert rel_err < 1e-4, "CUDA kernel algorithm does not match the verified reference!"
    print("PASSED: the CUDA kernel's contraction algorithm is correct.")
    print("(Compiling tt_linear_cuda.cu on real GPU hardware is still needed to confirm")
    print(" it builds and to measure real performance -- see kernels/build_cuda_kernel.py.)")


if __name__ == "__main__":
    main()
