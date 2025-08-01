#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "board.hpp"
#include "movegen.hpp"

namespace py = pybind11;

PYBIND11_MODULE(chess_engine_core, m) {
  // Module documentation
  m.doc() = "High-performance C++ chess engine core components";

  // Enums
  py::enum_<Color>(m, "Color")
      .value("WHITE", Color::WHITE)
      .value("BLACK", Color::BLACK)
      .export_values();

  py::enum_<PieceType>(m, "PieceType")
      .value("PAWN", PieceType::PAWN)
      .value("KNIGHT", PieceType::KNIGHT)
      .value("BISHOP", PieceType::BISHOP)
      .value("ROOK", PieceType::ROOK)
      .value("QUEEN", PieceType::QUEEN)
      .value("KING", PieceType::KING)
      .export_values();

  // Move class
  py::class_<Move>(m, "Move")
      .def(py::init<uint8_t, uint8_t, uint8_t>(), py::arg("from"),
           py::arg("to"), py::arg("promotion") = 0)
      .def_readwrite("from_square", &Move::from)
      .def_readwrite("to", &Move::to)
      .def_readwrite("promotion", &Move::promotion);

  // Board class
  py::class_<Board>(m, "Board")
      .def(py::init<>())
      .def_static("from_fen", &Board::from_fen, py::arg("fen"))
      // Core board operations
      .def("make_move", &Board::make_move, py::arg("move"))
      .def("generate_legal_moves", &Board::generate_legal_moves)
      // Display functions
      .def("to_fen", &Board::to_fen)
      .def("pretty", &Board::pretty)
      // State accessors
      .def("get_castling_rights", &Board::get_castling_rights)
      .def("get_side_to_move", &Board::get_side_to_move)
      .def("get_en_passant_square", &Board::get_en_passant_square)
      .def("get_halfmove_clock", &Board::get_halfmove_clock)
      .def("get_fullmove_number", &Board::get_fullmove_number)
      // Bitboard access
      .def(
          "get_piece_bb",
          py::overload_cast<PieceType, Color>(&Board::get_piece_bb, py::const_),
          py::arg("piece_type"), py::arg("color"))
      .def("get_color_bb", &Board::get_color_bb, py::arg("color"))
      .def("get_all_pieces_bb", &Board::get_all_pieces_bb);

  // Helper functions
  m.def("move_to_string", &move_to_string, py::arg("move"), py::arg("board"));
  m.def("move_debug_string", &move_debug_string, py::arg("move"),
        py::arg("board"));
  m.def("moves_to_string", &moves_to_string, py::arg("moves"),
        py::arg("board"));
}