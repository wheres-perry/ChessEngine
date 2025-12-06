#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "halfkp.hpp"

namespace py = pybind11;

void bind_halfkp(py::module_ &m) {
  auto halfkp_module = m.def_submodule("halfkp", "HalfKP feature extraction");

  // Constants
  halfkp_module.attr("NUM_SQUARES") = halfkp::NUM_SQUARES;
  halfkp_module.attr("NUM_PIECE_TYPES_NO_KING") =
      halfkp::NUM_PIECE_TYPES_NO_KING;
  halfkp_module.attr("NUM_COLORS") = halfkp::NUM_COLORS;
  halfkp_module.attr("NUM_PLANES") = halfkp::NUM_PLANES;
  halfkp_module.attr("HALFKP_FEATURES_PER_SIDE") =
      halfkp::HALFKP_FEATURES_PER_SIDE;
  halfkp_module.attr("TOTAL_FEATURES") = halfkp::TOTAL_FEATURES;

  // Functions
  halfkp_module.def("orient_square", &halfkp::orient_square,
                    py::arg("is_white_pov"), py::arg("square"),
                    "Orient square from perspective of given color");

  halfkp_module.def(
      "get_piece_index",
      [](PieceType pt, Color piece_color, bool is_white_pov) {
        return halfkp::get_piece_index(pt, piece_color, is_white_pov);
      },
      py::arg("piece_type"), py::arg("piece_color"), py::arg("is_white_pov"),
      "Get piece index (0-9) for HalfKP encoding");

  halfkp_module.def(
      "halfkp_index",
      [](bool is_white_pov, uint8_t king_square, uint8_t piece_square,
         PieceType pt, Color piece_color) {
        return halfkp::halfkp_index(is_white_pov, king_square, piece_square, pt,
                                    piece_color);
      },
      py::arg("is_white_pov"), py::arg("king_square"), py::arg("piece_square"),
      py::arg("piece_type"), py::arg("piece_color"),
      "Compute HalfKP feature index for a single piece");

  halfkp_module.def(
      "board_to_halfkp_indices", &halfkp::board_to_halfkp_indices,
      py::arg("board"), py::arg("is_white_pov"),
      "Extract all active HalfKP feature indices for one perspective");

  halfkp_module.def(
      "board_to_input_tensor", &halfkp::board_to_input_tensor, py::arg("board"),
      "Convert board to dense float32 tensor (both perspectives concatenated)");

  // AccumulatorUpdate struct
  py::class_<halfkp::AccumulatorUpdate>(halfkp_module, "AccumulatorUpdate")
      .def(py::init<>())
      .def_readwrite("added_indices", &halfkp::AccumulatorUpdate::added_indices)
      .def_readwrite("removed_indices",
                     &halfkp::AccumulatorUpdate::removed_indices);

  halfkp_module.def(
      "create_accumulator_updates", &halfkp::create_accumulator_updates,
      py::arg("board"), py::arg("move"),
      "Compute incremental updates for a move (both perspectives)");
}
