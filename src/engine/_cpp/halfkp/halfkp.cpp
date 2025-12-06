#include "halfkp.hpp"

#include <algorithm>
#include <cstring>

namespace halfkp {

std::vector<uint32_t> board_to_halfkp_indices(const Board &board,
                                              bool is_white_pov) noexcept {
  std::vector<uint32_t> indices;
  indices.reserve(32); // Typical piece count

  // Get king square for this perspective
  Color pov_color = is_white_pov ? Color::WHITE : Color::BLACK;
  auto king_sq_opt = board.king(pov_color);
  if (!king_sq_opt)
    return indices; // No king = invalid position
  uint8_t king_square = *king_sq_opt;

  // Iterate over all piece types (excluding king)
  for (uint8_t pt_idx = 0; pt_idx < NUM_PIECE_TYPES_NO_KING; ++pt_idx) {
    PieceType pt = static_cast<PieceType>(pt_idx);

    // Process both colors
    for (uint8_t color_idx = 0; color_idx < NUM_COLORS; ++color_idx) {
      Color color = static_cast<Color>(color_idx);
      std::vector<uint8_t> squares = board.pieces(pt, color);

      for (uint8_t sq : squares) {
        uint32_t idx = halfkp_index(is_white_pov, king_square, sq, pt, color);
        indices.push_back(idx);
      }
    }
  }

  return indices;
}

std::vector<float> board_to_input_tensor(const Board &board) noexcept {
  // Allocate dense tensor: white POV + black POV + bias planes
  std::vector<float> tensor(TOTAL_FEATURES, 0.0f);

  // White perspective
  {
    std::vector<uint32_t> white_indices = board_to_halfkp_indices(board, true);
    for (uint32_t idx : white_indices) {
      tensor[idx] = 1.0f;
    }
    // Bias plane for white (last feature of white side)
    tensor[HALFKP_FEATURES_PER_SIDE - 1] = 1.0f;
  }

  // Black perspective
  {
    std::vector<uint32_t> black_indices = board_to_halfkp_indices(board, false);
    uint32_t offset = HALFKP_FEATURES_PER_SIDE;
    for (uint32_t idx : black_indices) {
      tensor[offset + idx] = 1.0f;
    }
    // Bias plane for black (last feature of black side)
    tensor[TOTAL_FEATURES - 1] = 1.0f;
  }

  return tensor;
}

std::pair<AccumulatorUpdate, AccumulatorUpdate>
create_accumulator_updates(const Board &board, const Move &move) noexcept {
  AccumulatorUpdate white_update;
  AccumulatorUpdate black_update;

  // Get piece being moved
  auto piece_opt = board.piece_at(move.from);
  if (!piece_opt || !piece_opt->valid) {
    return {white_update, black_update};
  }

  PieceType moving_piece = piece_opt->type;
  Color moving_color = piece_opt->color;

  // Get king squares for both perspectives
  auto white_king_opt = board.king(Color::WHITE);
  auto black_king_opt = board.king(Color::BLACK);
  if (!white_king_opt || !black_king_opt) {
    return {white_update, black_update};
  }
  uint8_t white_king = *white_king_opt;
  uint8_t black_king = *black_king_opt;

  // Handle king moves specially (all features change)
  if (moving_piece == PieceType::KING) {
    // For king moves, we need full recomputation (return empty updates)
    // The caller should detect this and do full feature extraction
    return {white_update, black_update};
  }

  // Standard piece move: remove from old square, add to new square
  // White perspective
  {
    uint32_t removed_idx =
        halfkp_index(true, white_king, move.from, moving_piece, moving_color);
    white_update.removed_indices.push_back(removed_idx);

    PieceType final_piece = (move.promotion != 0)
                                ? static_cast<PieceType>(move.promotion)
                                : moving_piece;
    uint32_t added_idx =
        halfkp_index(true, white_king, move.to, final_piece, moving_color);
    white_update.added_indices.push_back(added_idx);
  }

  // Black perspective
  {
    uint32_t removed_idx =
        halfkp_index(false, black_king, move.from, moving_piece, moving_color);
    black_update.removed_indices.push_back(removed_idx);

    PieceType final_piece = (move.promotion != 0)
                                ? static_cast<PieceType>(move.promotion)
                                : moving_piece;
    uint32_t added_idx =
        halfkp_index(false, black_king, move.to, final_piece, moving_color);
    black_update.added_indices.push_back(added_idx);
  }

  // Handle captures (remove captured piece)
  auto captured_opt = board.piece_at(move.to);
  if (captured_opt && captured_opt->valid) {
    PieceType captured_piece = captured_opt->type;
    Color captured_color = captured_opt->color;

    // White perspective
    uint32_t removed_capture_white =
        halfkp_index(true, white_king, move.to, captured_piece, captured_color);
    white_update.removed_indices.push_back(removed_capture_white);

    // Black perspective
    uint32_t removed_capture_black = halfkp_index(
        false, black_king, move.to, captured_piece, captured_color);
    black_update.removed_indices.push_back(removed_capture_black);
  }

  // Handle en passant captures
  if (board.is_en_passant(move)) {
    uint8_t ep_square = board.get_en_passant_square();
    if (ep_square < 64) {
      // Captured pawn is on same file as destination, but different rank
      uint8_t captured_pawn_sq =
          (moving_color == Color::WHITE) ? (move.to - 8) : (move.to + 8);
      Color captured_color =
          (moving_color == Color::WHITE) ? Color::BLACK : Color::WHITE;

      // White perspective
      uint32_t removed_ep_white = halfkp_index(
          true, white_king, captured_pawn_sq, PieceType::PAWN, captured_color);
      white_update.removed_indices.push_back(removed_ep_white);

      // Black perspective
      uint32_t removed_ep_black = halfkp_index(
          false, black_king, captured_pawn_sq, PieceType::PAWN, captured_color);
      black_update.removed_indices.push_back(removed_ep_black);
    }
  }

  // Handle castling (rook moves)
  if (board.is_castling(move)) {
    uint8_t rook_from, rook_to;
    bool is_kingside = (move.to > move.from);

    if (moving_color == Color::WHITE) {
      if (is_kingside) {
        rook_from = 7; // h1
        rook_to = 5;   // f1
      } else {
        rook_from = 0; // a1
        rook_to = 3;   // d1
      }
    } else {
      if (is_kingside) {
        rook_from = 63; // h8
        rook_to = 61;   // f8
      } else {
        rook_from = 56; // a8
        rook_to = 59;   // d8
      }
    }

    // White perspective
    {
      uint32_t rook_removed = halfkp_index(true, white_king, rook_from,
                                           PieceType::ROOK, moving_color);
      uint32_t rook_added = halfkp_index(true, white_king, rook_to,
                                         PieceType::ROOK, moving_color);
      white_update.removed_indices.push_back(rook_removed);
      white_update.added_indices.push_back(rook_added);
    }

    // Black perspective
    {
      uint32_t rook_removed = halfkp_index(false, black_king, rook_from,
                                           PieceType::ROOK, moving_color);
      uint32_t rook_added = halfkp_index(false, black_king, rook_to,
                                         PieceType::ROOK, moving_color);
      black_update.removed_indices.push_back(rook_removed);
      black_update.added_indices.push_back(rook_added);
    }
  }

  return {white_update, black_update};
}

} // namespace halfkp
