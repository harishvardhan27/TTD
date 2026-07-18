"""
benchmark_kernel.py
=====================
JIT-compiles the custom fused CPU kernel (tt_linear_cpu.cpp) via
torch.utils.cpp_extension.load, verifies it produces IDENTICAL results
to the einsum-based TTLinear.forward (already extensively verified
elsewhere in this project), then benchmarks:

    dense nn.Linear   vs   TTLinear (torch.einsum)   vs   fused C++ kernel

This is the honest follow-up to the finding in README.md/tt_compression
that a generic einsum-based TT forward pass can be SLOWER than dense
matmul despite having far fewer parameters. Does hand-fusing the
contraction chain into explicit back-to-back BLAS calls close that gap?
Run this script to get a real, measured answer rather than an assumption.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

from tt_core import TTLinear

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

print("Compiling custom fused CPU kernel (this can take ~30-60s the first time)...")
tt_cpu = load(
    name="tt_linear_cpu",
    sources=[os.path.join(THIS_DIR, "kernels", "tt_linear_cpu.cpp")],
    verbose=False,
)
print("Compiled.")


_flatten_cache = {}


def fused_forward(ttlinear: TTLinear, x: torch.Tensor) -> torch.Tensor:
    key = id(ttlinear)
    if key not in _flatten_cache:
        # One-time cost: pre-flatten each core into its matmul-ready matrix.
        # This mirrors what a real deployment would do once at model-load
        # time, not on every forward call.
        _flatten_cache[key] = [tt_cpu.flatten_core(c.detach()) for c in ttlinear.cores]
    core_mats = _flatten_cache[key]
    ranks = list(ttlinear.full_ranks[1:])  # r_1..r_d (r_d always 1)

    out = tt_cpu.tt_linear_forward_fused(
        x.reshape(-1, ttlinear.in_features).contiguous(),
        core_mats,
        list(ttlinear.in_modes),
        list(ttlinear.out_modes),
        ranks,
    )
    if ttlinear.bias is not None:
        out = out + ttlinear.bias
    return out


def correctness_check():
    torch.manual_seed(0)
    lin = nn.Linear(768, 3072)
    ttl = TTLinear.from_linear(lin, num_modes=3, ranks=24)
    x = torch.randn(8, 768)

    y_einsum = ttl(x)
    y_fused = fused_forward(ttl, x)
    rel_err = (y_einsum - y_fused).norm().item() / y_einsum.norm().item()
    print(f"Correctness: fused kernel vs einsum relative difference = {rel_err:.2e} (should be ~1e-6, float32 precision)")
    assert rel_err < 1e-4, "custom kernel does not match the verified einsum reference!"
    return ttl, x


def benchmark(ttl, x, n_warmup=5, n_iters=50):
    lin_equiv = nn.Linear(ttl.in_features, ttl.out_features)

    def timeit(fn, *args):
        for _ in range(n_warmup):
            fn(*args)
        t0 = time.perf_counter()
        for _ in range(n_iters):
            fn(*args)
        return (time.perf_counter() - t0) / n_iters * 1000  # ms/iter

    with torch.no_grad():
        t_dense = timeit(lin_equiv, x)
        t_einsum = timeit(ttl, x)
        t_fused = timeit(lambda x_: fused_forward(ttl, x_), x)

    print()
    print(f"{'Method':<28}{'ms/iter (batch=8)':>20}")
    print("-" * 48)
    print(f"{'Dense nn.Linear':<28}{t_dense:>20.4f}")
    print(f"{'TTLinear (einsum)':<28}{t_einsum:>20.4f}")
    print(f"{'TTLinear (fused C++ kernel)':<28}{t_fused:>20.4f}")
    print()
    speedup_vs_einsum = t_einsum / t_fused
    ratio_vs_dense = t_fused / t_dense
    print(f"Fused kernel is {speedup_vs_einsum:.2f}x faster than generic einsum.")
    print(f"Fused kernel is {ratio_vs_dense:.2f}x the dense nn.Linear latency "
          f"({'faster' if ratio_vs_dense < 1 else 'still slower'} than dense on CPU).")


if __name__ == "__main__":
    ttl, x = correctness_check()
    benchmark(ttl, x)
