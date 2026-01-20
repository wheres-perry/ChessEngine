"""Setup script for chess engine C++ extensions."""

import os
from pathlib import Path

import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import (  # pyright: ignore[reportMissingModuleSource]
    find_namespace_packages,
    setup,
)

# Get the directory where setup.py is located (project root)
HERE = Path(__file__).parent.resolve()

# Platform-specific compiler flags
MSVC_COMPILE_ARGS = [
    "/O2",  # Maximum optimization
    "/W3",  # Warning level 3
    "/GL",  # Whole program optimization
    "/DNDEBUG",  # Remove debug assertions
]
MSVC_LINK_ARGS = [
    "/LTCG",  # Link-time code generation
]

UNIX_COMPILE_ARGS = [
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
UNIX_LINK_ARGS = [
    "-flto",  # Link-time optimization
    "-O3",  # Optimization at link time
]

# Pick a safe default based on the platform, but finalize per-compiler below.
extra_compile_args = MSVC_COMPILE_ARGS if os.name == "nt" else UNIX_COMPILE_ARGS
extra_link_args = MSVC_LINK_ARGS if os.name == "nt" else UNIX_LINK_ARGS


class BuildExt(build_ext):
    """Ensure compiler-specific flags are applied reliably."""

    def build_extensions(self) -> None:
        compiler_type = self.compiler.compiler_type

        # Check for sanitizer or custom flags in environment variables
        cflags = os.environ.get("CFLAGS", "")
        cxxflags = os.environ.get("CXXFLAGS", "")
        ldflags = os.environ.get("LDFLAGS", "")

        # If sanitizer flags are present, use them instead of optimization flags
        # This allows for address/undefined sanitizer builds in CI
        all_flags = (
            cflags.split() if cflags else []
        ) + (
            cxxflags.split() if cxxflags else []
        ) + (
            ldflags.split() if ldflags else []
        )
        has_sanitizer = any(
            flag.startswith("-fsanitize=") or flag.startswith("/fsanitize:")
            for flag in all_flags
        )

        if has_sanitizer:
            # Use environment variable flags for sanitizer builds
            compile_args = cxxflags.split() if cxxflags else (cflags.split() if cflags else [])
            link_args = ldflags.split() if ldflags else []
        elif compiler_type == "msvc":
            compile_args = MSVC_COMPILE_ARGS
            link_args = MSVC_LINK_ARGS
        else:
            compile_args = UNIX_COMPILE_ARGS
            link_args = UNIX_LINK_ARGS

        for ext in self.extensions:
            ext.extra_compile_args = compile_args
            ext.extra_link_args = link_args

        super().build_extensions()


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
        depends=[
            "src/engine/_cpp/board/board.hpp",
            "src/engine/_cpp/move_generation/move_generation.hpp",
            "src/engine/_cpp/halfkp/halfkp.hpp",
            "src/engine/_cpp/search/zobrist.hpp",
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
    packages=find_namespace_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    package_data={
        "engine": ["_cpp/**/*.hpp", "_cpp/**/*.cpp"],
    },
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
)
