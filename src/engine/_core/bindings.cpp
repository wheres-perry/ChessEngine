#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "board.h"
// Remove Zobrist for this example if not needed yet
// #include "zobrist.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Core C++ components for the chess engine";

  py::class_<Board>(m, "Board")
      .def(py::init<>())
      // Expose static methods to act like class methods in Python
      .def_static("from_fen", &Board::from_fen, py::arg("fen"),
                  "Creates a board from a FEN string.")
      .def_static("from_pgn", &Board::from_pgn, py::arg("pgn_content"),
                  "Creates a board from the content of a PGN file.")
      .def("pretty", &Board::pretty,
           "Returns a string representation of the board.");
}