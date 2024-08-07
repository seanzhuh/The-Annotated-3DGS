# Annotated Differential Gaussian Rasterization

Recently I've been playing with 3DGS for a while, and realize that it's an opportunity to dive into CUDA. This repo contains detailed annotations, or simulations that help to get the main idea, also what each step is doing. The annotation will touch on aspects like how to bind C++/CUDA PyTorch extentions with Python, i.e., provide an interface to Python, as well as the actual implementations of CUDA files.

## Code Structure

Let's start with how PyTorch define a neural module, which we are the most familiar with I guess.

```python
# ./diff_gaussian_rasterization/__init__.py
import torch.nn as nn
from . import _C

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings
    
    def markVisible(self, positions):
        return _C.mark_visible(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return rasterize_gaussians(*args, **kwarags)
```

This code snippet defines in the way we usually extend PyTorch to implement your own customized module in which a forward function must be implemented.

```python
# ./diff_gaussian_rasterization/__init__.py
def rasterize_gaussians(*args, **kwargs):
    return _RasterizeGaussians.apply(*args, **kwargs)

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
        return color, radii

    @staticmethod
    def backward(ctx, grad_out_color, _):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color, 
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.debug)

        grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads
```

The `rasterize_gaussians` function calls `_RasterizeGaussians` that is of type `torch.autograd.Function`, The `torch.autograd.Function` is another alternative to `torch.nn.Module` that allows you to define and implement your own neural network modules but with additional `backward` function. We have to derive gradient calculation via chain rule and implement `backward` function using operations provided by PyTorch. However if operations of PyTorch is slow and you want to optimize the speed using CUDA, this is the entry point that links PyTorch with your customized CUDA kernels. [PyTorch - Extending Autograd](https://pytorch.org/docs/stable/notes/extending.html#extending-autograd) gives a tutorial w.r.t. how to use this class.

```c++
/* ext.cpp
*/
#include <torch/extension.h>
#include "rasterize_points.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("mark_visible", &markVisible);
}
```

This file binds functions in Python to functions in C++, in other words, executing `_C.rasterize_gaussians()` in python is equal to directly calling `RasterizeGaussiansCUDA` in C++. The `rasterize_points.h` provides definitions for all three customized functions. It defines the arguments, return values of these functions.

In `rasterize_points.cu`, implementations of these functions can be found. These functions further calls functions defined under `CudaRasterizer` namespace that is defined in `./cuda_rasterizer/rasterizer.h` file.

```c++
/* rasterize_points.cu
*/

// arguments are omitted for brevity
RasterizeGaussiansCUDA()
{
	  rendered = CudaRasterizer::Rasterizer::forward();
      return rendered
}

RasterizeGaussiansBackwardCUDA()
{
    CudaRasterizer::Rasterizer::backward()
}

markVisible()
{
    CudaRasterizer::Rasterizer::markVisible()
}
```

The actual implementations and header files lie in cuda_rasterizer folder, now let's dive into this folder.

The `rasterizer.h` file defines `CudaRasterizer::Rasterizer::forward/backward/markVisible` and `rasterizer_impl.cu` file gives their implementations.

```c++
/* cuda_rasterizer/rasterizer_impl.cu
*/

CudaRasterizer::Rasterizer::forward() {
    FORWARD::preprocess()
    FORWARD::render()
}

CudaRasterizer::Rasterizer::backward() {
    BACKWARD::render()
    BACKWARD::preprocess()
}
```

What is immediately clear is that the lowest implementations are two functions `preprocess` and `render` in both `FORWARD` and `BACKWARD` namespaces defined in `forward.h` and `backward.h`, respectively. So as you can guess, the implementations are correspondingly given in `forward.cu` and `backward.cu` files.

Detailed annotations are mainly given in `forward.cu` and `backward.cu` files since the majority of other files are merely header files and hierarchically calls functions that eventually reaches these two functions. So there is a route that starts from `GaussianRasterizer` that provides the highest abstract api all the way down to these two `render` and `preprocess` functions in `forward.cu` and `backward.cu` files. Don't be itimidated, this file structure is relatively clear as long as you temporarily forget figuring out what all parameters mean in these files.