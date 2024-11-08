from setuptools import setup
from torch.utils import cpp_extension
from Cython.Build import cythonize

setup(
    name ='NMSCUDA',   #编译后的链接名称
    ext_modules=[cpp_extension.CUDAExtension('NMSCUDA', 
    ['nms_cuda/nms.cpp', 
    'nms_cuda/nms_kernel.cu', 
    ],)
    ],   #待编译文件，以及编译函数
    cmdclass={'build_ext': cpp_extension.BuildExtension}, #执行编译命令设置
    packages=['FastNMS']
)