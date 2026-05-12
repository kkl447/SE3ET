import os
from typing import Any, Dict, List

from setuptools import setup
from torch.utils import cpp_extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


PACKAGE_NAME = 'vgtk'
EXT_MODULES = ['gathering', 'grouping', 'zpconv']
PACKAGES = [
    'app',
    'cuda',
    'functional',
    'point3d',
    'pc',
    'mesh',
    'voxel',
    'spconv',
    'so3conv',
    'transform',
    'data.anchors',
]
INSTALL_REQUIREMENTS = [
    'numpy',
    'scikit-image==0.18.3',
    'scikit-learn==0.20.1',
    'open3d',
    'trimesh==3.2.0',
    'tqdm',
    'imageio',
    'plyfile',
    'parse',
    'colour',
]


def _disable_torch_cuda_version_check() -> None:
    def _skip_check(*args: Any, **kwargs: Any) -> None:
        return None

    cpp_extension._check_cuda_version = _skip_check


class PatchedBuildExtension(BuildExtension):
    def build_extensions(self) -> None:
        _disable_torch_cuda_version_check()
        super().build_extensions()


def _build_compile_args() -> Dict[str, List[str]]:
    cxx_flags = ['-O3', '-std=c++17']
    nvcc_flags = ['-O3', '--expt-relaxed-constexpr', '-std=c++17']

    if os.name != 'nt':
        cxx_flags.append('-fPIC')
        nvcc_flags.extend(['--compiler-options', '-fPIC'])

    return {'cxx': cxx_flags, 'nvcc': nvcc_flags}


def cuda_extension(package_name: str, ext: str) -> CUDAExtension:
    ext_name = f'{package_name}.cuda.{ext}'
    ext_cpp = f'{package_name}/cuda/{ext}_cuda.cpp'
    ext_cu = f'{package_name}/cuda/{ext}_cuda_kernel.cu'
    return CUDAExtension(
        ext_name,
        [ext_cpp, ext_cu],
        extra_compile_args=_build_compile_args(),
    )


pkg_name = PACKAGE_NAME
ext_modules = [cuda_extension(pkg_name, ext) for ext in EXT_MODULES]
pkgs = [pkg_name] + [f'{pkg_name}.{pkg}' for pkg in PACKAGES]
install_reqs = [req for req in INSTALL_REQUIREMENTS]


setup(
    description='Vision-Graphics deep learning ToolKit',
    author='VGL (Shichen Liu*, Haiwei Chen*)',
    author_email='liushichen95@gmail.com',
    license='MIT License',
    version='0.0.1',
    name=pkg_name,
    packages=pkgs,
    package_data={'': ['*.ply']},
    include_package_data=True,
    install_requires=install_reqs,
    ext_modules=ext_modules,
    cmdclass={'build_ext': PatchedBuildExtension},
)
