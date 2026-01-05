import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# 强制指定CUDA 11.6路径（conda环境）
os.environ['CUDA_HOME'] = os.path.expanduser("~/anaconda3/envs/r3dg")
os.environ['PATH'] = os.path.join(os.environ['CUDA_HOME'], 'bin') + os.pathsep + os.environ['PATH']
os.environ['LD_LIBRARY_PATH'] = os.path.join(os.environ['CUDA_HOME'], 'lib64') + os.pathsep + os.environ.get('LD_LIBRARY_PATH', '')

_src_path = os.path.dirname(os.path.abspath(__file__))

# 核心修复：升级到C++17 + 关闭冲突宏 + 适配sm_80
nvcc_args = [
    "-O3",
    "--expt-extended-lambda",
    "-arch=sm_80",          # A800算力
    "-std=c++17",           # 升级到C++17（解决GCC 11兼容问题）
    "-D_GLIBCXX_USE_CXX11_ABI=0",  # 匹配PyTorch 1.12.1 ABI
    # 关闭导致冲突的半精度宏（关键！）
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
]

cxx_args = [
    "-O3",
    "-std=c++17",           # C++17标准
    "-D_GLIBCXX_USE_CXX11_ABI=0",
    "-fPIC",                # 位置无关代码
    "-Wno-sign-compare",    # 屏蔽GCC警告
]

setup(
    name='bvh_tracing._C',
    description='CUDA RayTracer with BVH acceleration for 3DGS',
    ext_modules=[
        CUDAExtension(
            name='bvh_tracing._C',
            sources=[os.path.join(_src_path, 'src', f) for f in [
                'bvh.cu',
                'trace.cu',
                'construct.cu',
                'bindings.cpp',
            ]],
            include_dirs=[
                os.path.join(_src_path, 'include'),
                os.path.join(os.environ['CUDA_HOME'], 'include'),
            ],
            extra_compile_args={
                "nvcc": nvcc_args,
                "cxx": cxx_args
            },
            library_dirs=[os.path.join(os.environ['CUDA_HOME'], 'lib64')],
            libraries=['cudart'],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(no_python_abi_suffix=True),
    },
)