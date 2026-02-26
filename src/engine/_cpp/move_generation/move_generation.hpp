#pragma once

#include <array>
#include <vector>

#include "../board/board.hpp"

// ---------------------------------------------------------------------------
// Precomputed attack tables — defined in move_generation.cpp, declared here
// so that board.cpp (and search) can include them.
// ---------------------------------------------------------------------------

extern const std::array<Bitboard, 64> KNIGHT_ATTACKS;
extern const std::array<std::array<Bitboard, 64>, 2> PAWN_ATTACKS;
extern const std::array<Bitboard, 64> KING_ATTACKS;

// RAY_ATTACKS[sq][dir] — squares reachable from sq in direction dir (exclusive
// of sq itself) assuming an empty board.  Direction indices:
//   0=N, 1=S, 2=E, 3=W, 4=NE, 5=SW, 6=NW, 7=SE
extern const std::array<std::array<Bitboard, 8>, 64> RAY_ATTACKS;

// BETWEEN_SQUARES[a][b] — bitmask of squares strictly between a and b along
// their shared rank, file, or diagonal.  Zero when not aligned.
extern const std::array<std::array<Bitboard, 64>, 64> BETWEEN_SQUARES;

// Slider direction index sets (indices into the second dimension of
// RAY_ATTACKS).
extern const int ROOK_DIRECTIONS[4];   // 0,1,2,3  (N/S/E/W)
extern const int BISHOP_DIRECTIONS[4]; // 4,5,6,7  (NE/SW/NW/SE)
extern const int QUEEN_DIRECTIONS[8];  // 0-7  (all)

// ---------------------------------------------------------------------------
// Sliding-piece ray computation
// ---------------------------------------------------------------------------

// Returns all squares attacked along the given directions from sq, stopping at
// the first blocker in each direction (the blocker square is included).
[[nodiscard]] Bitboard get_ray_attacks(int sq, const int *directions,
                                       int num_dirs,
                                       Bitboard occupied) noexcept;

// ---------------------------------------------------------------------------
// Global attacked-square and check functions
// ---------------------------------------------------------------------------

// Returns a bitboard of all squares attacked by by_color, using the supplied
// occupancy.  Separating occupancy from board state lets callers exclude the
// moving king when checking king-move legality.
[[nodiscard]] Bitboard attacked_by(const Board &board, Color by_color,
                                   Bitboard occ) noexcept;

// Convenience wrapper that uses the full board occupancy.
[[nodiscard]] inline Bitboard
compute_attacked_squares(const Board &board, Color by_color) noexcept {
  return attacked_by(board, by_color, board.get_all_pieces_bb());
}

// Backwards-compatible alias retained for older callers / Python bindings.
[[nodiscard]] inline Bitboard get_attacked_squares(const Board &board,
                                                   Color by_color) noexcept {
  return compute_attacked_squares(board, by_color);
}

// Returns true if the king of the given color is currently in check.
[[nodiscard]] bool is_in_check(const Board &board, Color us) noexcept;

// Returns a bitboard of all pieces belonging to the opponent that are
// currently giving check to the king of color us.  king_sq and occ are passed
// in to avoid redundant computation at call sites that already have them.
[[nodiscard]] Bitboard get_checkers(const Board &board, Color us,
                                    uint8_t king_sq, Bitboard occ) noexcept;

// ---------------------------------------------------------------------------
// Castling legality
// ---------------------------------------------------------------------------

[[nodiscard]] bool is_castling_legal(const Board &board, Color us,
                                     bool kingside) noexcept;

// ---------------------------------------------------------------------------
// Pin computation
// ---------------------------------------------------------------------------

// Returns the set of our pieces that are absolutely pinned to the king and,
// for each pinned piece, the bitboard of squares it is legally allowed to
// move to (the pin ray from king through the piece to the pinner, inclusive
// of the pinner for capture).
[[nodiscard]] std::pair<Bitboard, std::array<Bitboard, 64>>
compute_pinned_pieces(const Board &board, Color us) noexcept;
