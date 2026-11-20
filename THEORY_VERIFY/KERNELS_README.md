# Custom Kernels for TT Contraction

## Files

| File | What it is | Status |
|---|---|---|
| `tt_linear_cpu.cpp` | Fused CPU kernel: explicit BLAS matmul chain instead of generic `torch.einsum` | **Compiled and benchmarked here** (see `../benchmark_kernel.py`) |
| `tt_linear_cuda.cu` | Fused single-launch CUDA kernel (d=3 case) | **Not compiled here** (no GPU/nvcc in this sandbox) — algorithm verified via Python simulation (`../verify_cuda_kernel_logic.py`), needs a real GPU to build and benchmark |
| `build_cuda_kernel.py` | `setuptools`/`CUDAExtension` build script for the CUDA kernel | For use on a CUDA machine |

## Honest result: does hand-fusing beat `torch.einsum` on CPU?

**No, not here.** Run `python benchmark_kernel.py` from the project root:

```
Method                         ms/iter (batch=8)
------------------------------------------------
Dense nn.Linear                           0.997
TTLinear (einsum)                         1.684
TTLinear (fused C++ kernel)               2.831
```

Even after removing the obvious inefficiency (recomputing each core's
permute+reshape on every forward call — fixed by precomputing
`flatten_core()` once), the hand-written chain of `transpose` +
`.contiguous()` + `matmul` calls is still **slower** than PyTorch's
built-in `einsum` for this shape pattern, at every batch size tested
(8/32/128/512). The likely reason: `torch.einsum`'s internal contraction
planner already picks efficient BLAS calls for this exact "chain of
small matrices" pattern and avoids some of the intermediate
`.contiguous()` copies our hand-written version forces by transposing
between steps. This is a genuinely useful negative result: **naively
"writing custom C++ instead of using the framework's einsum" is not
automatically a win** — the framework is already good at this on CPU.

## Why the CUDA kernel is a different story (in principle)

Kernel-launch overhead is much larger, relatively, on GPU than the cost
of an extra CPU-side `.contiguous()` copy. `torch.einsum` on GPU still
has to dispatch multiple separate CUDA kernel launches for a
multi-operand contraction; a hand-fused kernel that does the entire
3-core sequential contraction in **one** launch (using shared memory for
the intermediate states, as `tt_linear_cuda.cu` does) is exactly the
situation where custom kernels are known to pay off for small-tensor
workloads like this. We cannot confirm this quantitatively without GPU
hardware, but the qualitative reasoning is standard (it's the same
argument behind why frameworks like TensorRT/ONNX Runtime fuse small ops
specifically for GPU inference, and matters far less for CPU BLAS-bound
workloads like the one measured above).

## What's verified vs. what isn't

- **Verified by direct execution**: the CPU kernel's numerical
  correctness (matches `TTLinear`'s einsum forward to ~1e-7 relative
  error) and its real, measured latency (which is not a good result —
  reported above regardless).
- **Verified by algorithm simulation, not compilation**: the CUDA
  kernel's index arithmetic, checked by literally re-implementing the
  same loops/index expressions in NumPy and confirming they reproduce
  `TTLinear`'s output (see `../verify_cuda_kernel_logic.py`). This
  confirms the *algorithm* is correct; it does **not** confirm the CUDA
  syntax compiles cleanly with `nvcc`, or what the real GPU latency is.
  Both require hardware this sandbox doesn't have.

## Next steps on real GPU hardware

```bash
python kernels/build_cuda_kernel.py build_ext --inplace
python -c "
import torch, tt_linear_cuda
# build core0 (m1,n1,r1), core1 (r1,m2,n2,r2), core2 (r2,m3,n3) as float32 CUDA tensors
# and call tt_linear_cuda.tt_linear_forward_cuda(...) -- compare against
# TTLinear.forward on the same cores/input, exactly as verify_cuda_kernel_logic.py
# does on CPU, before trusting any speed numbers.
"
```
