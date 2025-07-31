#pragma once

#include <array>
#include <vector>

#include "board.hpp"

// Forward declarations for move generation functions
[[nodiscard]] Bitboard get_ray_attacks(int sq, const int* directions,
                                       int num_dirs,
                                       Bitboard occupied) noexcept;
[[nodiscard]] Bitboard get_attacked_squares(const Board& board,
                                            Color by_color) noexcept;
[[nodiscard]] bool is_in_check(const Board& board, Color us) noexcept;
[[nodiscard]] bool is_castling_legal(const Board& board, Color us,
                                     bool kingside) noexcept;
[[nodiscard]] std::pair<Bitboard, std::array<Bitboard, 64>>
compute_pinned_pieces(const Board& board, Color us) noexcept;
