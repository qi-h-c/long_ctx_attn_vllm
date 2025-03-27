from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="my_ops",
    ext_modules=[
        CUDAExtension(
            name="my_ops",
            sources=["add_cuda.cpp", "add_cuda_kernel.cu"],
            extra_compile_args={"cxx": ["-O2"], "nvcc": ["-O2"]}
        )
    ],
    cmdclass={"build_ext": BuildExtension}
)