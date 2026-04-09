#pragma once

#include <array>
#include <cstdint>

namespace eval {

// Constexpr helper: mirror 32 half-board centipawn values into a full
// 64-square table (ranks 1-4 followed by their reverse for ranks 5-8).
// This matches the Python ``make_table()`` layout.
constexpr std::array<int, 64> make_table(const int (&half)[32]) {
  std::array<int, 64> table{};
  for (int i = 0; i < 32; ++i)
    table[i] = half[i];
  for (int i = 0; i < 32; ++i)
    table[32 + i] = half[31 - i];
  return table;
}

// =====================================================================
// Middlegame Piece-Square Tables  (centipawns)
// =====================================================================

// clang-format off
inline constexpr int MG_PAWN_HALF[32] = {
     0,   0,   0,   0,   0,   0,   0,   0,
    50,  50,  50,  50,  50,  50,  50,  50,
    10,  10,  20,  30,  30,  20,  10,  10,
     5,   5,  10,  25,  25,  10,   5,   5,
};

inline constexpr int MG_KNIGHT_HALF[32] = {
   -50, -40, -30, -30, -30, -30, -40, -50,
   -40, -20,   0,   0,   0,   0, -20, -40,
   -30,   0,  10,  15,  15,  10,   0, -30,
   -30,   5,  15,  20,  20,  15,   5, -30,
};

inline constexpr int MG_BISHOP_HALF[32] = {
   -20, -10, -10, -10, -10, -10, -10, -20,
   -10,   0,   0,   0,   0,   0,   0, -10,
   -10,   0,   5,  10,  10,   5,   0, -10,
   -10,   5,   5,  10,  10,   5,   5, -10,
};

inline constexpr int MG_ROOK_HALF[32] = {
     0,   0,   0,   0,   0,   0,   0,   0,
     5,  10,  10,  10,  10,  10,  10,   5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
};

inline constexpr int MG_QUEEN_HALF[32] = {
   -20, -10, -10,  -5,  -5, -10, -10, -20,
   -10,   0,   0,   0,   0,   0,   0, -10,
   -10,   0,   5,   5,   5,   5,   0, -10,
    -5,   0,   5,   5,   5,   5,   0,  -5,
};

inline constexpr int MG_KING_HALF[32] = {
   -30, -40, -40, -50, -50, -40, -40, -30,
   -30, -40, -40, -50, -50, -40, -40, -30,
   -30, -40, -40, -50, -50, -40, -40, -30,
   -30, -40, -40, -50, -50, -40, -40, -30,
};

// Endgame Piece-Square Tables  (centipawns)

inline constexpr int EG_PAWN_HALF[32] = {
     0,   0,   0,   0,   0,   0,   0,   0,
    80,  80,  80,  80,  80,  80,  80,  80,
    50,  50,  50,  50,  50,  50,  50,  50,
    30,  30,  30,  30,  30,  30,  30,  30,
};

inline constexpr int EG_KNIGHT_HALF[32] = {
   -50, -40, -30, -30, -30, -30, -40, -50,
   -40, -20,   0,   0,   0,   0, -20, -40,
   -30,   0,  10,  15,  15,  10,   0, -30,
   -30,   5,  15,  20,  20,  15,   5, -30,
};

inline constexpr int EG_BISHOP_HALF[32] = {
   -20, -10, -10, -10, -10, -10, -10, -20,
   -10,   0,   0,   0,   0,   0,   0, -10,
   -10,   0,   5,  10,  10,   5,   0, -10,
   -10,   5,   5,  10,  10,   5,   5, -10,
};

inline constexpr int EG_ROOK_HALF[32] = {
     0,   0,   0,   0,   0,   0,   0,   0,
     5,  10,  10,  10,  10,  10,  10,   5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
};

inline constexpr int EG_QUEEN_HALF[32] = {
   -20, -10, -10,  -5,  -5, -10, -10, -20,
   -10,   0,   0,   0,   0,   0,   0, -10,
   -10,   0,   5,   5,   5,   5,   0, -10,
    -5,   0,   5,   5,   5,   5,   0,  -5,
};

inline constexpr int EG_KING_HALF[32] = {
   -50, -40, -30, -20, -20, -30, -40, -50,
   -30, -20, -10,   0,   0, -10, -20, -30,
   -30, -10,  20,  30,  30,  20, -10, -30,
   -30, -10,  30,  40,  40,  30, -10, -30,
};
// clang-format on

// Full 64-square tables (constexpr).
inline constexpr auto MG_PAWN = make_table(MG_PAWN_HALF);
inline constexpr auto MG_KNIGHT = make_table(MG_KNIGHT_HALF);
inline constexpr auto MG_BISHOP = make_table(MG_BISHOP_HALF);
inline constexpr auto MG_ROOK = make_table(MG_ROOK_HALF);
inline constexpr auto MG_QUEEN = make_table(MG_QUEEN_HALF);
inline constexpr auto MG_KING = make_table(MG_KING_HALF);

inline constexpr auto EG_PAWN = make_table(EG_PAWN_HALF);
inline constexpr auto EG_KNIGHT = make_table(EG_KNIGHT_HALF);
inline constexpr auto EG_BISHOP = make_table(EG_BISHOP_HALF);
inline constexpr auto EG_ROOK = make_table(EG_ROOK_HALF);
inline constexpr auto EG_QUEEN = make_table(EG_QUEEN_HALF);
inline constexpr auto EG_KING = make_table(EG_KING_HALF);

// Index by PieceType enum value (0=PAWN .. 5=KING).
inline constexpr std::array<const std::array<int, 64> *, 6> MG_TABLES = {
    &MG_PAWN, &MG_KNIGHT, &MG_BISHOP, &MG_ROOK, &MG_QUEEN, &MG_KING,
};

inline constexpr std::array<const std::array<int, 64> *, 6> EG_TABLES = {
    &EG_PAWN, &EG_KNIGHT, &EG_BISHOP, &EG_ROOK, &EG_QUEEN, &EG_KING,
};

} // namespace eval
