from setuptools import Extension, setup

try:
    import pybind11
except ImportError as exc:
    raise RuntimeError(
        "pybind11 is required. Install with: python3 -m pip install --user pybind11"
    ) from exc

ext_modules = [
    Extension(
        "rockchip_rga",
        ["rockchip_rga.cpp"],
        include_dirs=[pybind11.get_include(), "/usr/include/rga"],
        libraries=["rga"],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17"],
    )
]

setup(
    name="rockchip-rga-backend",
    version="0.1.0",
    ext_modules=ext_modules,
)
