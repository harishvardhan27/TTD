"""
build_cuda_kernel.py
======================
Build script for tt_linear_cuda.cu. Requires an NVIDIA GPU + the CUDA
toolkit (nvcc) installed -- neither is available in the sandbox that
built the rest of this project, and none on the AMD Ryzen 7730U laptop
used for the measured results (no NVIDIA GPU) -- it has NEVER been compiled
or run. Known limitation (RESEARCH_AUDIT.md H9): MAX_SHARED_FLOATS=4096 is too
small for the 768->3072 FFN shape even at rank 8. On a
CUDA-capable machine:

    pip install torch  # with CUDA support matching your driver
    cd THEORY_VERIFY && python build_cuda_kernel.py build_ext --inplace

Then in Python:

    import tt_linear_cuda
    out = tt_linear_cuda.tt_linear_forward_cuda(x, core0, core1, core2,
                                                 n1, n2, n3, m1, m2, m3, r1, r2)

Or use torch.utils.cpp_extension.load for JIT compilation instead of this
setuptools build, exactly like benchmark_kernel.py does for the CPU
kernel:

    from torch.utils.cpp_extension import load
    tt_cuda = load(name="tt_linear_cuda", sources=["THEORY_VERIFY/tt_linear_cuda.cu"])
"""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="tt_linear_cuda",
    ext_modules=[
        CUDAExtension(
            name="tt_linear_cuda",
            sources=["tt_linear_cuda.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
