cmake_minimum_required(VERSION 3.18)
project(chess_engine_core)

# Set C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)

add_library(chess_engine_core MODULE bindings.cpp board.cpp movegen.cpp)

target_link_libraries(chess_engine_core PRIVATE pybind11::module)
target_include_directories(chess_engine_core PRIVATE ${pybind11_INCLUDE_DIRS})
