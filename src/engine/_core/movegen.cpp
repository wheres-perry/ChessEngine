#include <algorithm>
#include <array>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "board.hpp"

// Precomputed attack masks (compile-time generation for efficiency)
// Knight moves from each square (bitboards of possible targets)
static constexpr std::array<Bitboard, 64> KNIGHT_ATTACKS = []() constexpr {
  std::array<Bitboard, 64> attacks{};
  for (int sq = 0; sq < 64; ++sq) {
    int r = sq / 8, f = sq % 8;
    // All 8 possible knight moves, clipped to board
    const std::array<std::pair<int, int>, 8> deltas = {{{2, 1},
                                                        {2, -1},
                                                        {-2, 1},
                                                        {-2, -1},
                                                        {1, 2},
                                                        {1, -2},
                                                        {-1, 2},
                                                        {-1, -2}}};
    for (const auto& [dr, df] : deltas) {
      int nr = r + dr, nf = f + df;
      if (nr >= 0 && nr < 8 && nf >= 0 && nf < 8) {
        attacks[sq] |= (1ULL << (nr * 8 + nf));
      }
    }
  }
  return attacks;
}();

// Pawn attacks (separate for white/black; captures only, not pushes)
static constexpr std::array<std::array<Bitboard, 64>, 2> PAWN_ATTACKS =
    []() constexpr {
      std::array<std::array<Bitboard, 64>, 2> attacks{};
      for (int sq = 0; sq < 64; ++sq) {
        int r = sq / 8, f = sq % 8;
        // White pawn attacks (forward-left/right)
        if (r < 7) {
          if (f > 0) attacks[0][sq] |= (1ULL << ((r + 1) * 8 + (f - 1)));
          if (f < 7) attacks[0][sq] |= (1ULL << ((r + 1) * 8 + (f + 1)));
        }
        // Black pawn attacks (forward-left/right)
        if (r > 0) {
          if (f > 0) attacks[1][sq] |= (1ULL << ((r - 1) * 8 + (f - 1)));
          if (f < 7) attacks[1][sq] |= (1ULL << ((r - 1) * 8 + (f + 1)));
        }
      }
      return attacks;
    }();

// King attacks (precomputed for efficiency, similar to knights)
static constexpr std::array<Bitboard, 64> KING_ATTACKS = []() constexpr {
  std::array<Bitboard, 64> attacks{};
  for (int sq = 0; sq < 64; ++sq) {
    int r = sq / 8, f = sq % 8;
    const std::array<std::pair<int, int>, 8> deltas = {
        {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}}};
    for (const auto& [dr, df] : deltas) {
      int nr = r + dr, nf = f + df;
      if (nr >= 0 && nr < 8 && nf >= 0 && nf < 8) {
        attacks[sq] |= (1ULL << (nr * 8 + nf));
      }
    }
  }
  return attacks;
}();

// Directions for sliders
static constexpr int ROOK_DIRECTIONS[4] = {8, -8, 1,
                                           -1};  // North, South, East, West
static constexpr int BISHOP_DIRECTIONS[4] = {9, -9, 7, -7};  // NE, SW, NW, SE
static constexpr int QUEEN_DIRECTIONS[8] = {8, -8, 1, -1, 9, -9, 7, -7};

// Robust ray attack function with wrap detection
[[nodiscard]] Bitboard get_ray_attacks(int sq, const int* directions,
                                       int num_dirs,
                                       Bitboard occupied) noexcept {
  Bitboard attacks = 0;
  int sq_r = sq / 8;
  int sq_f = sq % 8;
  for (int d = 0; d < num_dirs; ++d) {
    int offset = directions[d];
    int target = sq;
    int current_r = sq_r;
    int current_f = sq_f;
    while (true) {
      target += offset;
      if (target < 0 || target >= 64) break;
      int target_r = target / 8;
      int target_f = target % 8;
      int delta_r = target_r - current_r;
      int delta_f = target_f - current_f;
      if (std::abs(delta_r) > 1 || std::abs(delta_f) > 1 ||
          (std::abs(delta_r) + std::abs(delta_f) != 1 &&
           std::abs(delta_r) + std::abs(delta_f) != 2) ||
          (std::abs(delta_r) == std::abs(delta_f) && std::abs(delta_r) != 1) ||
          (std::abs(delta_r) != std::abs(delta_f) && delta_r != 0 &&
           delta_f != 0)) {
        break;
      }
      attacks |= (1ULL << target);
      current_r = target_r;
      current_f = target_f;
      if (occupied & (1ULL << target)) break;
    }
  }
  return attacks;
}

// Get all squares attacked by opponent (for check detection and castling)
[[nodiscard]] Bitboard get_attacked_squares(const Board& board,
                                            Color by_color) noexcept {
  Bitboard attacked = 0;
  Bitboard occupied = board.get_all_pieces_bb();

  // Pawn attacks
  Bitboard pawns = board.get_piece_bb(PieceType::PAWN, by_color);
  while (pawns) {
    uint8_t sq = __builtin_ctzll(pawns);
    pawns &= pawns - 1;
    attacked |= PAWN_ATTACKS[static_cast<uint8_t>(by_color)][sq];
  }

  // Knight attacks
  Bitboard knights = board.get_piece_bb(PieceType::KNIGHT, by_color);
  while (knights) {
    uint8_t sq = __builtin_ctzll(knights);
    knights &= knights - 1;
    attacked |= KNIGHT_ATTACKS[sq];
  }

  // Bishop attacks
  Bitboard bishops = board.get_piece_bb(PieceType::BISHOP, by_color);
  while (bishops) {
    uint8_t sq = __builtin_ctzll(bishops);
    bishops &= bishops - 1;
    attacked |= get_ray_attacks(sq, BISHOP_DIRECTIONS, 4, occupied);
  }

  // Rook attacks
  Bitboard rooks = board.get_piece_bb(PieceType::ROOK, by_color);
  while (rooks) {
    uint8_t sq = __builtin_ctzll(rooks);
    rooks &= rooks - 1;
    attacked |= get_ray_attacks(sq, ROOK_DIRECTIONS, 4, occupied);
  }

  // Queen attacks (rook + bishop)
  Bitboard queens = board.get_piece_bb(PieceType::QUEEN, by_color);
  while (queens) {
    uint8_t sq = __builtin_ctzll(queens);
    queens &= queens - 1;
    attacked |= get_ray_attacks(sq, QUEEN_DIRECTIONS, 8, occupied);
  }

  // King attacks
  Bitboard king = board.get_piece_bb(PieceType::KING, by_color);
  if (king) {
    uint8_t sq = __builtin_ctzll(king);
    attacked |= KING_ATTACKS[sq];
  }

  return attacked;
}

// Complete check detection implementation
[[nodiscard]] bool is_in_check(const Board& board, Color us) noexcept {
  // Find king's square
  Bitboard king_bb = board.get_piece_bb(PieceType::KING, us);
  if (king_bb == 0) return false;  // No king (edge case)
  uint8_t king_sq = __builtin_ctzll(king_bb);

  Color them = (us == Color::WHITE) ? Color::BLACK : Color::WHITE;
  Bitboard occupied = board.get_all_pieces_bb();
  Bitboard king_square_bb = 1ULL << king_sq;

  // Check for pawn attacks
  Bitboard enemy_pawns = board.get_piece_bb(PieceType::PAWN, them);
  while (enemy_pawns) {
    uint8_t pawn_sq = __builtin_ctzll(enemy_pawns);
    enemy_pawns &= enemy_pawns - 1;
    if (PAWN_ATTACKS[static_cast<uint8_t>(them)][pawn_sq] & king_square_bb) {
      return true;
    }
  }

  // Check for knight attacks
  Bitboard enemy_knights = board.get_piece_bb(PieceType::KNIGHT, them);
  while (enemy_knights) {
    uint8_t knight_sq = __builtin_ctzll(enemy_knights);
    enemy_knights &= enemy_knights - 1;
    if (KNIGHT_ATTACKS[knight_sq] & king_square_bb) {
      return true;
    }
  }

  // Check for sliding piece attacks (bishops, rooks, queens)
  // Bishop/Queen diagonal attacks
  Bitboard enemy_bishops = board.get_piece_bb(PieceType::BISHOP, them) |
                           board.get_piece_bb(PieceType::QUEEN, them);
  while (enemy_bishops) {
    uint8_t bishop_sq = __builtin_ctzll(enemy_bishops);
    enemy_bishops &= enemy_bishops - 1;
    if (get_ray_attacks(bishop_sq, BISHOP_DIRECTIONS, 4, occupied) &
        king_square_bb) {
      return true;
    }
  }

  // Rook/Queen horizontal/vertical attacks
  Bitboard enemy_rooks = board.get_piece_bb(PieceType::ROOK, them) |
                         board.get_piece_bb(PieceType::QUEEN, them);
  while (enemy_rooks) {
    uint8_t rook_sq = __builtin_ctzll(enemy_rooks);
    enemy_rooks &= enemy_rooks - 1;
    if (get_ray_attacks(rook_sq, ROOK_DIRECTIONS, 4, occupied) &
        king_square_bb) {
      return true;
    }
  }

  // Check for king attacks (adjacent squares)
  Bitboard enemy_king = board.get_piece_bb(PieceType::KING, them);
  if (enemy_king) {
    uint8_t enemy_king_sq = __builtin_ctzll(enemy_king);
    if (KING_ATTACKS[enemy_king_sq] & king_square_bb) {
      return true;
    }
  }

  return false;
}

// Castling legality
[[nodiscard]] bool is_castling_legal(const Board& board, Color us,
                                     bool kingside) noexcept {
  Color them = (us == Color::WHITE) ? Color::BLACK : Color::WHITE;
  if (is_in_check(board, us)) return false;
  Bitboard attacked = get_attacked_squares(board, them);
  Bitboard path;
  if (kingside) {
    path = (us == Color::WHITE) ? ((1ULL << 5) | (1ULL << 6))
                                : ((1ULL << 61) | (1ULL << 62));
  } else {
    path = (us == Color::WHITE) ? ((1ULL << 3) | (1ULL << 2))
                                : ((1ULL << 59) | (1ULL << 58));
  }
  return (path & attacked) == 0 && (path & board.get_all_pieces_bb()) == 0;
}

// Helper function to check if two squares are on the same ray
[[nodiscard]] bool squares_on_same_ray(uint8_t sq1, uint8_t sq2,
                                       const int* directions,
                                       int num_dirs) noexcept {
  int r1 = sq1 / 8, f1 = sq1 % 8;
  int r2 = sq2 / 8, f2 = sq2 % 8;

  for (int d = 0; d < num_dirs; ++d) {
    int offset = directions[d];
    int dr = (offset / 8) - ((offset < 0 && offset % 8 != 0) ? 1 : 0);
    int df = offset % 8;
    if (df > 4) df -= 8;  // Handle negative wrap

    if (dr == 0 && df != 0) {  // Horizontal
      if (r1 == r2 && ((f1 < f2 && df > 0) || (f1 > f2 && df < 0))) return true;
    } else if (df == 0 && dr != 0) {  // Vertical
      if (f1 == f2 && ((r1 < r2 && dr > 0) || (r1 > r2 && dr < 0))) return true;
    } else if (std::abs(dr) == std::abs(df)) {  // Diagonal
      int rank_diff = r2 - r1;
      int file_diff = f2 - f1;
      if (std::abs(rank_diff) == std::abs(file_diff) &&
          ((rank_diff > 0) == (dr > 0)) && ((file_diff > 0) == (df > 0)))
        return true;
    }
  }
  return false;
}

// Fixed pinned pieces computation
[[nodiscard]] std::pair<Bitboard, std::array<Bitboard, 64>>
compute_pinned_pieces(const Board& board, Color us) noexcept {
  Bitboard pinned = 0;
  std::array<Bitboard, 64> pin_rays = {0};
  Color them = (us == Color::WHITE) ? Color::BLACK : Color::WHITE;
  Bitboard occupied = board.get_all_pieces_bb();
  Bitboard our_pieces = board.get_color_bb(us);
  Bitboard king_bb = board.get_piece_bb(PieceType::KING, us);
  if (king_bb == 0) return {pinned, pin_rays};
  uint8_t king_sq = __builtin_ctzll(king_bb);

  // Check rook/queen pins
  Bitboard potential_pinners = board.get_piece_bb(PieceType::ROOK, them) |
                               board.get_piece_bb(PieceType::QUEEN, them);
  while (potential_pinners) {
    uint8_t pinner_sq = __builtin_ctzll(potential_pinners);
    potential_pinners &= potential_pinners - 1;

    // Check if pinner and king are on the same rook ray
    if (!squares_on_same_ray(pinner_sq, king_sq, ROOK_DIRECTIONS, 4)) continue;

    // Get all squares between pinner and king
    Bitboard ray_from_pinner =
        get_ray_attacks(pinner_sq, ROOK_DIRECTIONS, 4, 1ULL << king_sq);
    if (!(ray_from_pinner & (1ULL << king_sq))) continue;

    // Create mask for squares between pinner and king
    uint8_t min_sq = std::min(pinner_sq, king_sq);
    uint8_t max_sq = std::max(pinner_sq, king_sq);
    Bitboard between_mask = 0;

    // Determine step direction
    int step;
    int pinner_rank = pinner_sq / 8, pinner_file = pinner_sq % 8;
    int king_rank = king_sq / 8, king_file = king_sq % 8;

    if (pinner_rank == king_rank) {  // Same rank
      step = (pinner_file < king_file) ? 1 : -1;
    } else {  // Same file
      step = (pinner_rank < king_rank) ? 8 : -8;
    }

    for (uint8_t sq = pinner_sq + step; sq != king_sq; sq += step) {
      between_mask |= (1ULL << sq);
    }

    Bitboard pieces_between = between_mask & occupied & our_pieces;
    if (__builtin_popcountll(pieces_between) == 1) {
      uint8_t pinned_sq = __builtin_ctzll(pieces_between);
      pinned |= (1ULL << pinned_sq);
      // Pin ray includes the path from pinner through pinned piece to king
      pin_rays[pinned_sq] = ray_from_pinner | (1ULL << pinner_sq);
    }
  }

  // Check bishop/queen pins
  potential_pinners = board.get_piece_bb(PieceType::BISHOP, them) |
                      board.get_piece_bb(PieceType::QUEEN, them);
  while (potential_pinners) {
    uint8_t pinner_sq = __builtin_ctzll(potential_pinners);
    potential_pinners &= potential_pinners - 1;

    // Check if pinner and king are on the same bishop ray
    if (!squares_on_same_ray(pinner_sq, king_sq, BISHOP_DIRECTIONS, 4))
      continue;

    Bitboard ray_from_pinner =
        get_ray_attacks(pinner_sq, BISHOP_DIRECTIONS, 4, 1ULL << king_sq);
    if (!(ray_from_pinner & (1ULL << king_sq))) continue;

    // Create mask for squares between pinner and king
    Bitboard between_mask = 0;
    int pinner_rank = pinner_sq / 8, pinner_file = pinner_sq % 8;
    int king_rank = king_sq / 8, king_file = king_sq % 8;

    // Determine diagonal step
    int rank_step = (pinner_rank < king_rank) ? 1 : -1;
    int file_step = (pinner_file < king_file) ? 1 : -1;
    int step = rank_step * 8 + file_step;

    for (uint8_t sq = pinner_sq + step; sq != king_sq; sq += step) {
      between_mask |= (1ULL << sq);
    }

    Bitboard pieces_between = between_mask & occupied & our_pieces;
    if (__builtin_popcountll(pieces_between) == 1) {
      uint8_t pinned_sq = __builtin_ctzll(pieces_between);
      pinned |= (1ULL << pinned_sq);
      pin_rays[pinned_sq] = ray_from_pinner | (1ULL << pinner_sq);
    }
  }

  return {pinned, pin_rays};
}

// Main generate_legal_moves implementation - now with fixed pin handling
std::vector<Move> Board::generate_legal_moves() const noexcept {
  std::vector<Move> legal_moves;
  legal_moves.reserve(64);  // Generous reserve to avoid reallocations

  Color us = side_to_move ? Color::WHITE : Color::BLACK;
  Color them = (us == Color::WHITE) ? Color::BLACK : Color::WHITE;
  Bitboard our_pieces = get_color_bb(us);
  Bitboard their_pieces = get_color_bb(them);
  Bitboard occupied = get_all_pieces_bb();
  Bitboard empty = ~occupied;

  // Compute pinned pieces and pin rays
  auto [pinned, pin_rays] = compute_pinned_pieces(*this, us);

  // Helper lambda to add moves from a source square, with legality checking
  auto add_legal_moves = [&](uint8_t from, Bitboard targets,
                             uint8_t promotion = 0) {
    while (targets) {
      uint8_t to = __builtin_ctzll(targets);
      targets &= targets - 1;  // Clear LSB

      // Test legality by making the move and checking if king is in check
      Board temp = *this;
      Move test_move = {from, to, promotion};
      temp.make_move(test_move);

      if (!is_in_check(temp, us)) {
        legal_moves.push_back(test_move);
      }
    }
  };

  // Pawn moves
  Bitboard pawns = get_piece_bb(PieceType::PAWN, us);
  int direction = (us == Color::WHITE) ? 8 : -8;
  int start_rank = (us == Color::WHITE) ? 1 : 6;
  int promo_rank = (us == Color::WHITE) ? 7 : 0;

  while (pawns) {
    uint8_t from = __builtin_ctzll(pawns);
    pawns &= pawns - 1;
    int from_rank = from / 8;

    Bitboard pawn_is_pinned = (1ULL << from) & pinned;
    Bitboard allowed_targets = ~0ULL;  // All by default
    if (pawn_is_pinned) {
      allowed_targets = pin_rays[from];  // Restrict to pin ray
    }

    // Single push
    uint8_t to = from + direction;
    Bitboard single_push = (to < 64 && (1ULL << to) & empty) ? (1ULL << to) : 0;
    single_push &= allowed_targets;
    if (single_push) {
      if (from_rank == promo_rank - (us == Color::WHITE ? 1 : -1)) {
        // Promotion - add all four promotion types
        add_legal_moves(from, single_push,
                        static_cast<uint8_t>(PieceType::QUEEN));
        add_legal_moves(from, single_push,
                        static_cast<uint8_t>(PieceType::ROOK));
        add_legal_moves(from, single_push,
                        static_cast<uint8_t>(PieceType::BISHOP));
        add_legal_moves(from, single_push,
                        static_cast<uint8_t>(PieceType::KNIGHT));
      } else {
        add_legal_moves(from, single_push);
      }

      // Double push from start rank
      if (from_rank == start_rank) {
        uint8_t double_to = to + direction;
        Bitboard double_push = (double_to < 64 && (1ULL << double_to) & empty)
                                   ? (1ULL << double_to)
                                   : 0;
        double_push &= allowed_targets;
        add_legal_moves(from, double_push);
      }
    }

    // Captures
    Bitboard captures =
        PAWN_ATTACKS[static_cast<uint8_t>(us)][from] & their_pieces;
    captures &= allowed_targets;
    if (from_rank == promo_rank - (us == Color::WHITE ? 1 : -1)) {
      // Promotion captures
      while (captures) {
        uint8_t cap_to = __builtin_ctzll(captures);
        captures &= captures - 1;
        add_legal_moves(from, 1ULL << cap_to,
                        static_cast<uint8_t>(PieceType::QUEEN));
        add_legal_moves(from, 1ULL << cap_to,
                        static_cast<uint8_t>(PieceType::ROOK));
        add_legal_moves(from, 1ULL << cap_to,
                        static_cast<uint8_t>(PieceType::BISHOP));
        add_legal_moves(from, 1ULL << cap_to,
                        static_cast<uint8_t>(PieceType::KNIGHT));
      }
    } else {
      add_legal_moves(from, captures);
    }

    // En passant
    if (en_passant_square != -1) {
      Bitboard ep_attacks = PAWN_ATTACKS[static_cast<uint8_t>(us)][from] &
                            (1ULL << en_passant_square);
      ep_attacks &= allowed_targets;
      add_legal_moves(from, ep_attacks);
    }
  }

  // Knight moves
  Bitboard knights = get_piece_bb(PieceType::KNIGHT, us);
  while (knights) {
    uint8_t from = __builtin_ctzll(knights);
    knights &= knights - 1;

    Bitboard knight_is_pinned = (1ULL << from) & pinned;
    Bitboard allowed_targets = ~0ULL;
    if (knight_is_pinned) {
      allowed_targets = pin_rays[from];
    }

    Bitboard targets = KNIGHT_ATTACKS[from] & ~our_pieces & allowed_targets;
    add_legal_moves(from, targets);
  }

  // King moves (non-castling, using precomputed table)
  Bitboard king = get_piece_bb(PieceType::KING, us);
  if (king) {
    uint8_t from = __builtin_ctzll(king);
    Bitboard targets = KING_ATTACKS[from] & ~our_pieces;
    add_legal_moves(from, targets);
  }

  // Castling - complete implementation with all legality checks
  uint8_t king_home = (us == Color::WHITE) ? 4 : 60;
  if (get_piece_bb(PieceType::KING, us) &
      (1ULL << king_home)) {  // King on home square
    // Kingside castling
    uint8_t rights_mask_ks = (us == Color::WHITE) ? 1 : 4;
    if (castling_rights & rights_mask_ks) {
      Bitboard ks_path = (us == Color::WHITE) ? ((1ULL << 5) | (1ULL << 6))
                                              : ((1ULL << 61) | (1ULL << 62));
      if ((ks_path & occupied) == 0 && is_castling_legal(*this, us, true)) {
        legal_moves.push_back(
            {king_home, static_cast<uint8_t>(king_home + 2), 0});
      }
    }

    // Queenside castling
    uint8_t rights_mask_qs = (us == Color::WHITE) ? 2 : 8;
    if (castling_rights & rights_mask_qs) {
      Bitboard qs_path = (us == Color::WHITE)
                             ? ((1ULL << 1) | (1ULL << 2) | (1ULL << 3))
                             : ((1ULL << 57) | (1ULL << 58) | (1ULL << 59));
      if ((qs_path & occupied) == 0 && is_castling_legal(*this, us, false)) {
        legal_moves.push_back(
            {king_home, static_cast<uint8_t>(king_home - 2), 0});
      }
    }
  }

  // Sliding pieces (rooks, bishops, queens) - with pin handling
  // Rooks
  Bitboard rooks = get_piece_bb(PieceType::ROOK, us);
  while (rooks) {
    uint8_t from = __builtin_ctzll(rooks);
    rooks &= rooks - 1;

    Bitboard rook_is_pinned = (1ULL << from) & pinned;
    Bitboard allowed_targets = ~0ULL;
    if (rook_is_pinned) {
      allowed_targets = pin_rays[from];
    }

    Bitboard attacks = get_ray_attacks(from, ROOK_DIRECTIONS, 4, occupied);
    Bitboard targets = attacks & ~our_pieces & allowed_targets;
    add_legal_moves(from, targets);
  }

  // Bishops
  Bitboard bishops = get_piece_bb(PieceType::BISHOP, us);
  while (bishops) {
    uint8_t from = __builtin_ctzll(bishops);
    bishops &= bishops - 1;

    Bitboard bishop_is_pinned = (1ULL << from) & pinned;
    Bitboard allowed_targets = ~0ULL;
    if (bishop_is_pinned) {
      allowed_targets = pin_rays[from];
    }

    Bitboard attacks = get_ray_attacks(from, BISHOP_DIRECTIONS, 4, occupied);
    Bitboard targets = attacks & ~our_pieces & allowed_targets;
    add_legal_moves(from, targets);
  }

  // Queens (combine rook + bishop rays)
  Bitboard queens = get_piece_bb(PieceType::QUEEN, us);
  while (queens) {
    uint8_t from = __builtin_ctzll(queens);
    queens &= queens - 1;

    Bitboard queen_is_pinned = (1ULL << from) & pinned;
    Bitboard allowed_targets = ~0ULL;
    if (queen_is_pinned) {
      allowed_targets = pin_rays[from];
    }

    Bitboard attacks = get_ray_attacks(from, QUEEN_DIRECTIONS, 8, occupied);
    Bitboard targets = attacks & ~our_pieces & allowed_targets;
    add_legal_moves(from, targets);
  }

  return legal_moves;
}
