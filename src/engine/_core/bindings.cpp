#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "board.hpp"

namespace py = pybind11;

PYBIND11_MODULE(chess_engine_core, m) {
  m.doc() = "Core C++ components for the chess engine";

  py::class_<Board>(m, "Board")
      .def(py::init<>())
      .def_static("from_fen", &Board::from_fen, py::arg("fen"))
      .def("make_move", &Board::make_move, py::arg("move"))
      .def("generate_legal_moves", &Board::generate_legal_moves)
      .def("to_fen", &Board::to_fen)
      .def("pretty", &Board::pretty)
      .def("get_castling_rights",
           &Board::get_castling_rights)  // Essential for tests
      .def("get_side_to_move", &Board::get_side_to_move)
      .def("get_en_passant_square", &Board::get_en_passant_square);

  py::class_<Move>(m, "Move")
      .def(py::init<uint8_t, uint8_t, uint8_t>(), py::arg("from"),
           py::arg("to"), py::arg("promotion") = 0)
      .def_readwrite("from_square", &Move::from)
      .def_readwrite("to", &Move::to)
      .def_readwrite("promotion", &Move::promotion);

  m.def("move_to_string", &move_to_string, py::arg("move"), py::arg("board"));
  m.def("move_debug_string", &move_debug_string, py::arg("move"),
        py::arg("board"));
  m.def("moves_to_string", &moves_to_string, py::arg("moves"),
        py::arg("board"));
}
