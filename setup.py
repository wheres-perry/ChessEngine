"""Setup script for chess engine C++ extensions."""

import os

import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import (  # pyright: ignore[reportMissingModuleSource]
    find_packages,
    setup,
)

# Performance-optimized compiler flags
extra_compile_args = [
    "-O3",  # Maximum optimization
    "-march=native",  # Optimize for current CPU architecture
    "-mtune=native",  # Tune for current CPU
    "-ffast-math",  # Fast math operations (be careful with this)
    "-funroll-loops",  # Unroll loops for speed
    "-finline-functions",  # Inline functions aggressively
    "-flto",  # Link-time optimization
    "-DNDEBUG",  # Remove debug assertions
    "-Wall",  # Enable warnings
    "-Wextra",  # Extra warnings
]

# Link-time optimization flags
extra_link_args = [
    "-flto",  # Link-time optimization
    "-O3",  # Optimization at link time
]

# Add these flags for PGO (requires two-step build process)
pgo_generate = ["-fprofile-generate"]  # First build
pgo_use = ["-fprofile-use"]  # Second build after profiling

ext_modules = [
    Pybind11Extension(
        "engine._core.chess_engine_core",
        [
            "src/engine/_cpp/main_bindings.cpp",
            "src/engine/_cpp/board/board.cpp",
            "src/engine/_cpp/board/bindings.cpp",
            "src/engine/_cpp/move_generation/move_generation.cpp",
            "src/engine/_cpp/move_generation/bindings.cpp",
            "src/engine/_cpp/search/bindings.cpp",
        ],
        cxx_std=20,
        include_dirs=[pybind11.get_cmake_dir() + "/../include", "src/engine/_core"],
        define_macros=[
            ("VERSION_INFO", '"dev"'),
            ("NDEBUG", None),  # Remove debug code
        ],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
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
