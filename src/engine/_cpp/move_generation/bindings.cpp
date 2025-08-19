#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../board/board.hpp"
#include "move_generation.hpp"

namespace py = pybind11;

void init_movegen_bindings(py::module_& m) {
  m.def("get_attacked_squares", &get_attacked_squares, py::arg("board"),
        py::arg("by_color"),
        "Returns a bitboard of all squares attacked by the given color");

  m.def("is_in_check", &is_in_check, py::arg("board"), py::arg("us"),
        "Returns whether the given color is in check");
}