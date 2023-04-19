import os

import torch
from setuptools import setup, Extension
from torch.utils import cpp_extension


torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "include")
torch_lib_path_c10 = os.path.join(torch_lib_path, "include", "c10")
torch_lib_path_torch = os.path.join(torch_lib_path, "include", "torch")

custom_kwargs = {
    'extra_compile_args': ['-g'],
    'libraries': ['-Wl,-rpath,/path/to/custom/libs']
}

setup(
    name='quant_cuda',
    ext_modules=[cpp_extension.CUDAExtension(
        'quant_cuda', ['quant_cuda.cpp', 'quant_cuda_kernel.cu'],
        library_dirs=[torch_lib_path],
    )],
    version='0.0.2',
    include_package_data=True,
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
