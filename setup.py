#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="diff_gaussian_rasterization",
    packages=['diff_gaussian_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            # add debug flag that allows cuda-gdb debugger for inspection
            # -g: generate debug information for host code
            # -G: generate debug information for device code
            # -O: specify optimization level for host code, 0 means no optimizations, direct translation from source code, easy for debugging
            # -shared: generate a shared library during linking
            # -Xcompiler: specify options directly to the lower-level compiler/preprocessor w/o the need for nvcc to know such options
            # extra_compile_args={"nvcc": ["-O0", "-Xcompiler", "-fPIC", "-G", "-g", 
            #                              "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")],
            #                     'cxx': ["-g"]},
            # extra_link_args=["-shared"]
            )
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
