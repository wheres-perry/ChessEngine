#pragma once

#include <array>
#include <vector>

#include "../board/board.hpp"

// Fast bit manipulation helpers are now in board.hpp

// Forward declarations for move generation functions
[[nodiscard]] Bitboard get_ray_attacks(int sq, const int *directions,
                                       int num_dirs,
                                       Bitboard occupied) noexcept;

[[nodiscard]] Bitboard compute_attacked_squares(const Board &board,
                                                Color by_color) noexcept;

// Backwards-compatible alias; kept for older callers/bindings.
[[nodiscard]] inline Bitboard get_attacked_squares(const Board &board,
                                                   Color by_color) noexcept {
  return compute_attacked_squares(board, by_color);
}

[[nodiscard]] bool is_in_check(const Board &board, Color us) noexcept;

[[nodiscard]] bool is_castling_legal(const Board &board, Color us,
                                     bool kingside) noexcept;

[[nodiscard]] std::pair<Bitboard, std::array<Bitboard, 64>>
compute_pinned_pieces(const Board &board, Color us) noexcept;

// Add lookup table declarations
extern const std::array<Bitboard, 64> KNIGHT_ATTACKS;
extern const std::array<std::array<Bitboard, 64>, 2> PAWN_ATTACKS;
extern const std::array<Bitboard, 64> KING_ATTACKS;
extern const int ROOK_DIRECTIONS[4];
extern const int BISHOP_DIRECTIONS[4];
extern const int QUEEN_DIRECTIONS[8];
