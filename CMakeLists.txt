cmake_minimum_required(VERSION 3.21)
project(chessengine_cpp LANGUAGES CXX)

# 1. Define the C++ Standard (Moved from pyproject.toml)
# Ensure C++17 is used (adjust if needed, e.g., 20).
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Prefer standard C++ over compiler-specific extensions
set(CMAKE_CXX_EXTENSIONS OFF)

# 2. Enable ccache for faster rebuilds during development
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    message(STATUS "Found ccache, enabling C/C++ compiler launcher")
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_FOUND})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_FOUND})
endif()

# 3. Performance Optimization Flags (Crucial for Chess Engine)
# CMAKE_BUILD_TYPE is set by scikit-build-core (to "Release" based on pyproject.toml).
# CMake handles standard Release flags (like -O3) automatically. We add specialized flags here.
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Release build detected: Applying high-performance optimizations.")
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        # -march=native: Optimize for the specific CPU architecture of the build machine
        add_compile_options(-march=native)
    elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        # MSVC optimization flags (e.g., /O2) are typically handled by the Release configuration.
    endif()
endif()

# 4. Find Python and pybind11
# Use the Python interpreter that invoked the build process (managed by scikit-build-core)
find_package(Python REQUIRED COMPONENTS Interpreter Development)
# Find pybind11 (installed via uv/pip)
find_package(pybind11 CONFIG REQUIRED)

# 5. Define the extension module
# CUSTOMIZE: List your C++ source files here.
# Assuming your Python packages are in 'src/engine' and 'src/io_utils'.
set(SOURCES
    # Example: src/engine/cpp/main.cpp
    # Example: src/engine/cpp/search.cpp
)

# Create the Python extension module (e.g., chessengine_core.so)
# Check if SOURCES list is populated before defining the module.
if (SOURCES)
    pybind11_add_module(chessengine_core MODULE ${SOURCES})

    # Link against pybind11 and Python libraries
    target_link_libraries(chessengine_core PRIVATE pybind11::headers Python::Python)

    # 6. Installation (Handles the 'src' layout)
    # CUSTOMIZE: If your main package is 'engine' located at 'src/engine', the destination is 'src/engine'.
    install(TARGETS chessengine_core
        DESTINATION src/engine/
    )
else()
    message(WARNING "No C++ source files defined in the SOURCES variable in CMakeLists.txt. The C++ extension will not be built.")
endif()