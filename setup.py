"""Setup script for chess engine C++ extensions."""

import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import find_packages, setup

ext_modules = [
    Pybind11Extension(
        "engine._core.chess_engine_core",
        [
            "src/engine/_core/bindings.cpp",
            "src/engine/_core/board.cpp",
            "src/engine/_core/movegen.cpp",
        ],
        cxx_std=20,  # C++20 standard as requested
        include_dirs=[pybind11.get_cmake_dir() + "/../include", "src/engine/_core"],
        define_macros=[("VERSION_INFO", '"dev"')],
    ),
]

setup(
    name="chessengine",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
