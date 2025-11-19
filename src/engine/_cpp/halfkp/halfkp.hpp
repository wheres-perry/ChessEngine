#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "../board/board.hpp"

// HalfKP feature extraction constants
namespace halfkp {

static constexpr uint8_t NUM_SQUARES = 64;
static constexpr uint8_t NUM_PIECE_TYPES_NO_KING = 5;  // Exclude king
static constexpr uint8_t NUM_COLORS = 2;

// Feature dimensions
static constexpr uint32_t NUM_PLANES =
    NUM_SQUARES * NUM_PIECE_TYPES_NO_KING * NUM_COLORS + 1;
static constexpr uint32_t HALFKP_FEATURES_PER_SIDE = NUM_SQUARES * NUM_PLANES;
static constexpr uint32_t TOTAL_FEATURES = 2 * HALFKP_FEATURES_PER_SIDE;

// Orient square from perspective of given color
[[nodiscard]] constexpr inline uint8_t orient_square(
    bool is_white_pov, uint8_t square) noexcept {
  // White POV: no change; Black POV: vertical flip
  return is_white_pov ? square : (square ^ 56);
}

// Get piece index (0-9) for HalfKP encoding
// Friendly pieces: 0-4 (P,N,B,R,Q), opponent pieces: 5-9
[[nodiscard]] constexpr inline uint8_t get_piece_index(
    PieceType pt, Color piece_color, bool is_white_pov) noexcept {
  uint8_t base_idx = static_cast<uint8_t>(pt);
  bool is_friendly = (is_white_pov && piece_color == Color::WHITE) ||
                     (!is_white_pov && piece_color == Color::BLACK);
  return is_friendly ? base_idx : (base_idx + NUM_PIECE_TYPES_NO_KING);
}

// Compute HalfKP feature index for a single piece
[[nodiscard]] constexpr inline uint32_t halfkp_index(
    bool is_white_pov,
    uint8_t king_square,
    uint8_t piece_square,
    PieceType pt,
    Color piece_color) noexcept {
  uint8_t oriented_king = orient_square(is_white_pov, king_square);
  uint8_t oriented_piece = orient_square(is_white_pov, piece_square);
  uint8_t piece_idx = get_piece_index(pt, piece_color, is_white_pov);

  return static_cast<uint32_t>(oriented_king) * NUM_PLANES +
         static_cast<uint32_t>(piece_idx) * NUM_SQUARES +
         static_cast<uint32_t>(oriented_piece);
}

// Extract all active HalfKP feature indices for one perspective
// Returns vector of indices (sparse representation)
[[nodiscard]] std::vector<uint32_t> board_to_halfkp_indices(
    const Board& board, bool is_white_pov) noexcept;

// Convert board to dense float32 tensor (both perspectives concatenated)
// Output: array of size TOTAL_FEATURES
[[nodiscard]] std::vector<float> board_to_input_tensor(
    const Board& board) noexcept;

// Accumulator update structure for incremental feature updates
struct AccumulatorUpdate {
  std::vector<uint32_t> added_indices;
  std::vector<uint32_t> removed_indices;
};

// Compute incremental updates for a move (both perspectives)
[[nodiscard]] std::pair<AccumulatorUpdate, AccumulatorUpdate>
create_accumulator_updates(const Board& board, const Move& move) noexcept;

}  // namespace halfkp

