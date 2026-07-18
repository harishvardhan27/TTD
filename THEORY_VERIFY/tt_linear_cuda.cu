// tt_linear_cuda.cu
// ===================
// Custom CUDA kernel for TT-core contraction, specialized to the d=3
// case (in_modes = (n1,n2,n3), out_modes = (m1,m2,m3), internal ranks
// (r1,r2)) which is this project's default factorization
// (num_modes=3 everywhere else in the codebase).
//
// IMPORTANT / HONESTY NOTE: this file requires an NVIDIA GPU + the CUDA
// toolkit (nvcc) to compile and run. Neither is available in the sandbox
// that built the rest of this project (checked: no `nvidia-smi`, no
// `nvcc`, torch.cuda.is_available() == False) -- see benchmark_kernel.py
// and its results for what WAS actually compiled and measured here (a
// CPU C++ extension, which turned out to be slower than plain
// torch.einsum once you count the transpose/copy overhead -- a genuinely
// useful negative result, not hidden). This CUDA file is provided as a
// correct, ready-to-build reference for real GPU deployment; it has NOT
// been compiled or benchmarked in this environment. Build/test it on a
// CUDA machine with:
//     python kernels/build_cuda_kernel.py build_ext --inplace
// (see that file for the setuptools/CUDAExtension build script) before
// relying on its numbers.
//
// Design: one CUDA thread block per batch sample. Within a block, the
// three sequential core contractions are done back-to-back using shared
// memory for the intermediate "state" tensor, so the ENTIRE per-sample
// TT contraction happens in a single kernel launch (unlike the CPU
// version's d separate ATen ops, or unlike calling 3 separate kernels).
// This matters much more on GPU than CPU: kernel-launch overhead
// (typically several microseconds) is a much larger fraction of the
// total cost for these small per-core matmuls on GPU than the
// already-fast BLAS dispatch is on CPU, so fusing into one launch is
// where a real speedup is actually plausible for this workload.
//
// Assumes ranks r1, r2 and mode sizes are small enough that the
// intermediate states fit in shared memory (true for the ranks used
// throughout this project, e.g. r <= 64, mode sizes <= ~64).

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define MAX_SHARED_FLOATS 4096  // generous for demo-scale ranks/mode-sizes; tune per-deployment

// core0: (1, m1, n1, r1)          -- flattened as (m1*n1, r1) since r0=1
// core1: (r1, m2, n2, r2)
// core2: (r2, m3, n3, 1)          -- flattened as (r2, m3*n3) since r3=1
// x:     (batch, n1*n2*n3)
// out:   (batch, m1*m2*m3)
__global__ void tt_linear_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ core0,
    const float* __restrict__ core1,
    const float* __restrict__ core2,
    float* __restrict__ out,
    int n1, int n2, int n3,
    int m1, int m2, int m3,
    int r1, int r2) {

  const int b = blockIdx.x;              // one block per batch sample
  const int tid = threadIdx.x;
  const int n_threads = blockDim.x;

  extern __shared__ float shared_mem[];
  // state_a: after contracting core0 -> shape (m1, n2, n3, r1)
  float* state_a = shared_mem;
  const int state_a_size = m1 * n2 * n3 * r1;
  // state_b: after contracting core1 -> shape (m1, m2, n3, r2)
  float* state_b = shared_mem + state_a_size;
  const int state_b_size = m1 * m2 * n3 * r2;

  const float* x_b = x + b * (n1 * n2 * n3);

  // ---- Step 1: contract core0 over n1 ----
  // state_a[m1_, n2_, n3_, r1_] = sum_{n1_} x_b[n1_, n2_, n3_] * core0[m1_, n1_, r1_]
  // core0 laid out as (m1, n1, r1) row-major (r0=1 squeezed out by caller).
  for (int idx = tid; idx < state_a_size; idx += n_threads) {
    int r1_ = idx % r1;
    int rest = idx / r1;
    int n3_ = rest % n3;
    rest /= n3;
    int n2_ = rest % n2;
    int m1_ = rest / n2;

    float acc = 0.0f;
    for (int n1_ = 0; n1_ < n1; ++n1_) {
      float x_val = x_b[(n1_ * n2 + n2_) * n3 + n3_];
      float core_val = core0[(m1_ * n1 + n1_) * r1 + r1_];
      acc += x_val * core_val;
    }
    state_a[idx] = acc;
  }
  __syncthreads();

  // ---- Step 2: contract core1 over (r1, n2) ----
  // state_b[m1_, m2_, n3_, r2_] = sum_{r1_,n2_} state_a[m1_,n2_,n3_,r1_] * core1[r1_, m2_, n2_, r2_]
  for (int idx = tid; idx < state_b_size; idx += n_threads) {
    int r2_ = idx % r2;
    int rest = idx / r2;
    int n3_ = rest % n3;
    rest /= n3;
    int m2_ = rest % m2;
    int m1_ = rest / m2;

    float acc = 0.0f;
    for (int r1_ = 0; r1_ < r1; ++r1_) {
      for (int n2_ = 0; n2_ < n2; ++n2_) {
        float sa = state_a[((m1_ * n2 + n2_) * n3 + n3_) * r1 + r1_];
        float c1 = core1[((r1_ * m2 + m2_) * n2 + n2_) * r2 + r2_];
        acc += sa * c1;
      }
    }
    state_b[idx] = acc;
  }
  __syncthreads();

  // ---- Step 3: contract core2 over (r2, n3) ----
  // out[m1_, m2_, m3_] = sum_{r2_, n3_} state_b[m1_,m2_,n3_,r2_] * core2[r2_, m3_, n3_]
  const int out_size = m1 * m2 * m3;
  float* out_b = out + b * out_size;
  for (int idx = tid; idx < out_size; idx += n_threads) {
    int m3_ = idx % m3;
    int rest = idx / m3;
    int m2_ = rest % m2;
    int m1_ = rest / m2;

    float acc = 0.0f;
    for (int r2_ = 0; r2_ < r2; ++r2_) {
      for (int n3_ = 0; n3_ < n3; ++n3_) {
        float sb = state_b[((m1_ * m2 + m2_) * n3 + n3_) * r2 + r2_];
        float c2 = core2[(r2_ * m3 + m3_) * n3 + n3_];
        acc += sb * c2;
      }
    }
    out_b[idx] = acc;
  }
}

torch::Tensor tt_linear_forward_cuda(
    torch::Tensor x,       // (batch, n1*n2*n3)
    torch::Tensor core0,   // (m1, n1, r1)  [r0=1 already squeezed by caller]
    torch::Tensor core1,   // (r1, m2, n2, r2)
    torch::Tensor core2,   // (r2, m3, n3)  [r3=1 already squeezed by caller]
    int64_t n1, int64_t n2, int64_t n3,
    int64_t m1, int64_t m2, int64_t m3,
    int64_t r1, int64_t r2) {

  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(x.dtype() == torch::kFloat32, "this reference kernel only supports float32");

  const int64_t batch = x.size(0);
  auto out = torch::empty({batch, m1 * m2 * m3}, x.options());

  const int threads = 256;
  const int blocks = batch;
  const size_t shared_bytes = (m1 * n2 * n3 * r1 + m1 * m2 * n3 * r2) * sizeof(float);
  TORCH_CHECK(shared_bytes / sizeof(float) <= MAX_SHARED_FLOATS,
              "intermediate states too large for this reference kernel's shared-memory budget; "
              "reduce ranks/mode sizes or restructure into multiple kernel launches");

  tt_linear_forward_kernel<<<blocks, threads, shared_bytes>>>(
      x.data_ptr<float>(),
      core0.contiguous().data_ptr<float>(),
      core1.contiguous().data_ptr<float>(),
      core2.contiguous().data_ptr<float>(),
      out.data_ptr<float>(),
      (int)n1, (int)n2, (int)n3,
      (int)m1, (int)m2, (int)m3,
      (int)r1, (int)r2);

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tt_linear_forward_cuda", &tt_linear_forward_cuda,
        "Fused single-launch TT-core contraction for d=3 (CUDA, requires GPU)");
}
