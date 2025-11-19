#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "board.hpp"

namespace py = pybind11;

void init_board_bindings(py::module_& m) {
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

  py::enum_<GameState>(m, "GameState")
      .value("ONGOING", GameState::ONGOING)
      .value("CHECKMATE", GameState::CHECKMATE)
      .value("STALEMATE", GameState::STALEMATE)
      .value("DRAW_BY_FIFTY_MOVE", GameState::DRAW_BY_FIFTY_MOVE)
      .value("DRAW_BY_INSUFFICIENT_MATERIAL",
             GameState::DRAW_BY_INSUFFICIENT_MATERIAL)
      .value("DRAW_BY_REPETITION", GameState::DRAW_BY_REPETITION)
      .export_values();

  py::class_<Piece>(m, "Piece")
      .def_property_readonly("piece_type",
                             [](const Piece& piece) { return piece.type; })
      .def_property_readonly("color",
                             [](const Piece& piece) { return piece.color; })
      .def("symbol", &Piece::symbol)
      .def("__repr__", [](const Piece& piece) {
        return piece.valid ? std::string("Piece('") + piece.symbol() + "')"
                           : std::string("Piece(None)");
      })
      .def("__bool__", [](const Piece& piece) { return piece.valid; });

  // Move class
  py::class_<Move>(m, "Move")
      .def(py::init<uint8_t, uint8_t, uint8_t>(), py::arg("from"),
           py::arg("to"), py::arg("promotion") = 0)
      .def_readwrite("from_square", &Move::from)
      .def_readwrite("to_square", &Move::to)
      .def_readwrite("promotion", &Move::promotion)
      .def_static("from_uci", &move_from_uci, py::arg("uci"))
      .def("uci", &move_to_uci)
      .def("__repr__", [](const Move& move) {
        return "<Move " + move_to_uci(move) + ">";
      })
      .def(
          "__hash__",
          [](const Move& move) {
            return (static_cast<int>(move.from) << 16) ^
                   (static_cast<int>(move.to) << 8) ^ move.promotion;
          })
      .def("__eq__",
           [](const Move& a, const Move& b) {
             return a.from == b.from && a.to == b.to &&
                    a.promotion == b.promotion;
           })
      .def("__ne__",
           [](const Move& a, const Move& b) {
             return !(a.from == b.from && a.to == b.to &&
                      a.promotion == b.promotion);
           });

  // Board class
  py::class_<Board>(m, "Board")
      .def(py::init<>())
      .def_static("from_fen", &Board::from_fen, py::arg("fen"))
      .def("make_move", &Board::make_move, py::arg("move"))
      .def("push", &Board::push, py::arg("move"))
      .def("pop", &Board::pop)
      .def("push_san", &Board::push_san, py::arg("san"))
      .def("to_fen", &Board::to_fen)
      .def("fen", &Board::fen)
      .def("set_fen", &Board::set_fen)
      .def("pretty", &Board::pretty)
      .def("get_castling_rights", &Board::get_castling_rights)
      .def("get_side_to_move", &Board::get_side_to_move)
      .def("get_en_passant_square", &Board::get_en_passant_square)
      .def("get_halfmove_clock", &Board::get_halfmove_clock)
      .def("get_fullmove_number", &Board::get_fullmove_number)
      .def(
          "get_piece_bb",
          py::overload_cast<PieceType, Color>(&Board::get_piece_bb, py::const_),
          py::arg("piece_type"), py::arg("color"))
      .def("get_color_bb", &Board::get_color_bb, py::arg("color"))
      .def("get_all_pieces_bb", &Board::get_all_pieces_bb)
      .def("piece_at", &Board::piece_at, py::arg("square"))
      .def("pieces", &Board::pieces, py::arg("piece_type"), py::arg("color"))
      .def("king", &Board::king, py::arg("color"))
      .def("has_kingside_castling_rights",
           &Board::has_kingside_castling_rights, py::arg("color"))
      .def("has_queenside_castling_rights",
           &Board::has_queenside_castling_rights, py::arg("color"))
      .def("is_capture", &Board::is_capture, py::arg("move"))
      .def("is_castling", &Board::is_castling, py::arg("move"))
      .def("is_kingside_castling", &Board::is_kingside_castling,
           py::arg("move"))
      .def("is_queenside_castling", &Board::is_queenside_castling,
           py::arg("move"))
      .def("is_en_passant", &Board::is_en_passant, py::arg("move"))
      .def("is_check", &Board::is_check)
      .def("copy", &Board::copy)
      .def("is_game_over", &Board::is_game_over)
      .def("generate_legal_moves", &Board::generate_legal_moves)
      .def_property_readonly("turn", &Board::turn)
      .def_property_readonly("ep_square", &Board::ep_square)
      .def_property_readonly(
          "legal_moves",
          [](const Board& board) { return board.generate_legal_moves(); });

  // Helper functions
  m.def("move_to_string", &move_to_string, py::arg("move"), py::arg("board"));
  m.def("move_debug_string", &move_debug_string, py::arg("move"),
        py::arg("board"));
  m.def("moves_to_string", &moves_to_string, py::arg("moves"),
        py::arg("board"));
  m.def("square_file",
        [](uint8_t square) { return ::square_file(square); },
        py::arg("square"));
  m.def("square_rank",
        [](uint8_t square) { return ::square_rank(square); },
        py::arg("square"));
  m.def("move_to_uci", &move_to_uci, py::arg("move"));

  m.attr("SQUARES") = py::cast(SQUARES);
  m.attr("PIECE_TYPES") = py::cast(PIECE_TYPES_ARRAY);
  m.attr("BB_A1") = py::int_(BB_A1);
  m.attr("BB_H1") = py::int_(BB_H1);
  m.attr("BB_A8") = py::int_(BB_A8);
  m.attr("BB_H8") = py::int_(BB_H8);
}