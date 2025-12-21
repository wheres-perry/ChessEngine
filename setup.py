"""Setup script for chess engine C++ extensions."""

import os
import sys
from pathlib import Path

import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import (  # pyright: ignore[reportMissingModuleSource]
    find_packages,
    setup,
)

# Get the directory where setup.py is located (project root)
HERE = Path(__file__).parent.resolve()

# Platform-specific compiler flags
if sys.platform == "win32":
    # MSVC compiler flags for Windows
    extra_compile_args = [
        "/O2",  # Maximum optimization
        "/W3",  # Warning level 3
        "/GL",  # Whole program optimization
        "/DNDEBUG",  # Remove debug assertions
    ]
    extra_link_args = [
        "/LTCG",  # Link-time code generation
    ]
else:
    # GCC/Clang flags for Linux/Mac
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
extra_link_args = [
    "-flto",  # Link-time optimization
    "-O3",  # Optimization at link time
]

# PGO flags (commented out as they require two-step build)
# pgo_generate = ["-fprofile-generate"]  # First build
# pgo_use = ["-fprofile-use"]  # Second build after profiling

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
            "src/engine/_cpp/halfkp/halfkp.cpp",
            "src/engine/_cpp/halfkp/bindings.cpp",
        ],
        cxx_std=20,
        include_dirs=[
            pybind11.get_cmake_dir() + "/../include",
            str(HERE / "src/engine/_core"),
            str(HERE / "src/engine/_cpp"),
            str(HERE / "src/engine/_cpp/board"),
            str(HERE / "src/engine/_cpp/move_generation"),
            str(HERE / "src/engine/_cpp/search"),
            str(HERE / "src/engine/_cpp/halfkp"),
        ],
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
