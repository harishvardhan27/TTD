// tt_linear_cpu.cpp
// ===================
// Custom fused kernel for TT-core contraction, exposed as a PyTorch C++
// extension (JIT-compiled via torch.utils.cpp_extension.load, no manual
// setup.py/build step needed).
//
// Why this exists: TTLinear.forward() in tt_core.py uses a single
// torch.einsum call with a dynamically-built equation string. That is
// correct (verified extensively elsewhere in this project) but leaves
// contraction-order and kernel-fusion decisions to PyTorch's generic
// einsum path, which is not necessarily using the fastest possible
// sequence of BLAS calls for this specific "chain of small matrices"
// structure. This file instead explicitly walks the TT-core chain left
// to right, at each step doing exactly one reshape+matmul (so cuBLAS/
// MKL's batched GEMM is used directly, back-to-back, with no
// intermediate generic contraction-planning overhead) -- i.e. it is
// hand-fused for this specific access pattern rather than relying on a
// general-purpose contraction engine.
//
// This CPU version is compiled and benchmarked for real inside the
// project's sandbox (see benchmark_kernel.py) since it needs no GPU.
// See kernels/tt_linear_cuda.cu for the GPU/CUDA counterpart, which
// additionally fuses the whole chain into a single kernel launch (this
// CPU version still does d separate matmul calls -- fusing further on
// CPU brings little benefit since each matmul already dispatches to a
// fast BLAS routine, but on GPU avoiding d separate kernel launches
// matters much more, which is exactly what the CUDA kernel does).

#include <torch/extension.h>
#include <vector>

// x: (batch, in_features), already reshaped by caller into modes
// cores: vector of d tensors, core_k of shape (r_prev, out_k, in_k, r_k)
// Returns: (batch, out_features)
// x: (batch, in_features), already reshaped by caller into modes
// core_mats: vector of d PRE-COMPUTED flattened core matrices, core_mats[k]
//   of shape (r_prev*in_k, out_k*r_k) -- i.e. cores[k].permute(0,2,1,3).reshape(...)
//   done ONCE when the layer is built/loaded, not on every forward call.
// Returns: (batch, out_features)
torch::Tensor tt_linear_forward_fused(
    torch::Tensor x,
    std::vector<torch::Tensor> core_mats,
    std::vector<int64_t> in_modes,
    std::vector<int64_t> out_modes,
    std::vector<int64_t> ranks) {  // ranks[k] = r_k for k=0..d-1 (r_d always 1, not included)

  TORCH_CHECK(core_mats.size() == in_modes.size(), "core_mats/in_modes length mismatch");
  const int64_t d = static_cast<int64_t>(core_mats.size());
  const int64_t B = x.size(0);

  std::vector<int64_t> state_shape = {B, 1};
  for (auto n : in_modes) state_shape.push_back(n);
  torch::Tensor state = x.reshape(state_shape).contiguous();

  int64_t out_prod_so_far = 1;
  int64_t r_prev = 1;

  for (int64_t k = 0; k < d; ++k) {
    int64_t out_k = out_modes[k];
    int64_t in_k = in_modes[k];
    int64_t r_k = ranks[k];

    int64_t rest_prod = 1;
    for (int64_t j = k + 1; j < d; ++j) rest_prod *= in_modes[j];

    torch::Tensor state_r = state.reshape({B * out_prod_so_far, r_prev * in_k, rest_prod});
    torch::Tensor state_t = state_r.transpose(1, 2);  // view, no copy needed for matmul's benefit

    // core_mats[k] is already (r_prev*in_k, out_k*r_k), precomputed once outside the hot path
    torch::Tensor result = torch::matmul(state_t, core_mats[k]);  // (batch', rest_prod, out_k*r_k)
    result = result.transpose(1, 2).contiguous();                 // (batch', out_k*r_k, rest_prod)

    out_prod_so_far *= out_k;
    r_prev = r_k;
    state = result.reshape({B, out_prod_so_far, r_prev, rest_prod});
  }

  torch::Tensor out = state.reshape({B, out_prod_so_far});
  return out;
}

// Helper (called once at layer-build time, not in the forward hot path):
// pre-flatten a core (r_prev, out_k, in_k, r_k) into (r_prev*in_k, out_k*r_k).
torch::Tensor flatten_core(torch::Tensor core) {
  int64_t r_prev = core.size(0), out_k = core.size(1), in_k = core.size(2), r_k = core.size(3);
  return core.permute({0, 2, 1, 3}).contiguous().reshape({r_prev * in_k, out_k * r_k});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tt_linear_forward_fused", &tt_linear_forward_fused,
        "Fused TT-core chain contraction (CPU), takes pre-flattened core matrices");
  m.def("flatten_core", &flatten_core,
        "One-time core flattening, call at layer-build time and cache the result");
}
