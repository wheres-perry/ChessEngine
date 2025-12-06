#include "move_generation.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <sstream>
#include <vector>

#include "../board/board.hpp"

// Precomputed attack masks (compile-time generation for efficiency)
// Knight moves from each square (bitboards of possible targets)
const std::array<Bitboard, 64> KNIGHT_ATTACKS = []() {
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
const std::array<std::array<Bitboard, 64>, 2> PAWN_ATTACKS = []() {
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
const std::array<Bitboard, 64> KING_ATTACKS = []() {
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

// Directions for sliders (Indices into RAY_ATTACKS)
// 0:N, 1:S, 2:E, 3:W, 4:NE, 5:SW, 6:NW, 7:SE
const int ROOK_DIRECTIONS[4] = {0, 1, 2, 3};
const int BISHOP_DIRECTIONS[4] = {4, 5, 6, 7};
const int QUEEN_DIRECTIONS[8] = {0, 1, 2, 3, 4, 5, 6, 7};

// Precomputed ray attacks [square][direction_index]
const std::array<std::array<Bitboard, 8>, 64> RAY_ATTACKS = []() {
  std::array<std::array<Bitboard, 8>, 64> attacks{};
  // Offsets corresponding to indices 0-7
  const int offsets[8] = {8, -8, 1, -1, 9, -9, 7, -7};

  for (int sq = 0; sq < 64; ++sq) {
    int sq_r = sq / 8;
    int sq_f = sq % 8;

    for (int d = 0; d < 8; ++d) {
      int offset = offsets[d];
      int target = sq;
      int current_r = sq_r;
      int current_f = sq_f;

      while (true) {
        target += offset;
        if (target < 0 || target >= 64) break;

        int target_r = target / 8;
        int target_f = target % 8;

        // Wrap detection
        if ((offset == 1 && target_r != current_r) ||   // East wrap
            (offset == -1 && target_r != current_r) ||  // West wrap
            (std::abs(target_r - current_r) > 1) ||     // Vertical jump too far
            (std::abs(target_f - current_f) > 1)) {     // Horizontal jump too far
          break;
        }

        attacks[sq][d] |= (1ULL << target);
        current_r = target_r;
        current_f = target_f;
      }
    }
  }
  return attacks;
}();

inline int get_lsb_index(Bitboard bb) noexcept {
  return __builtin_ctzll(bb);
}

inline int get_msb_index(Bitboard bb) noexcept {
  return 63 - __builtin_clzll(bb);
}

// Robust ray attack function - optimized using precomputed tables
Bitboard get_ray_attacks(int sq, const int* directions, int num_dirs,
                         Bitboard occupied) noexcept {
  Bitboard attacks = 0;

  for (int i = 0; i < num_dirs; ++i) {
    int dir = directions[i];
    Bitboard ray = RAY_ATTACKS[sq][dir];
    Bitboard blockers = ray & occupied;

    if (blockers) {
      // Even indices are positive directions (lsb), odd are negative (msb)
      int blocker_sq = (dir % 2 == 0) ? get_lsb_index(blockers) : get_msb_index(blockers);
      
      // XOR out the ray starting from the blocker (inclusive) to the edge
      // ray ^ ray_from_blocker removes the "shadow"
      // RAY_ATTACKS[blocker][dir] is the ray FROM the blocker EXCLUDING the blocker.
      // Wait, RAY_ATTACKS[sq] excludes sq.
      // So RAY_ATTACKS[blocker] excludes blocker.
      // We want to keep the blocker in the attacks (capture).
      // So we want to remove everything AFTER the blocker.
      // ray (from sq) includes blocker and shadow.
      // RAY_ATTACKS[blocker] is the shadow.
      // So ray ^ (RAY_ATTACKS[blocker] & ray) ?
      // Since RAY_ATTACKS[blocker] is subset of ray (mostly), XOR works nicely.
      attacks |= (ray ^ RAY_ATTACKS[blocker_sq][dir]);
    } else {
      attacks |= ray;
    }
  }
  return attacks;
}

// Get all squares attacked by opponent (for check detection and castling)
Bitboard compute_attacked_squares(const Board& board, Color by_color) noexcept {
  Bitboard attacked = 0;
  Bitboard occupied = board.get_all_pieces_bb();
  uint8_t color_idx = static_cast<uint8_t>(by_color);

  // Pawn attacks - optimized to directly use the lookup table
  Bitboard pawns = board.get_piece_bb(PieceType::PAWN, by_color);
  while (pawns) {
    uint8_t sq = pop_lsb(pawns);
    attacked |= PAWN_ATTACKS[color_idx][sq];
  }

  // Knight attacks - directly use the lookup table
  Bitboard knights = board.get_piece_bb(PieceType::KNIGHT, by_color);
  while (knights) {
    uint8_t sq = pop_lsb(knights);
    attacked |= KNIGHT_ATTACKS[sq];
  }

  // Bishop attacks
  Bitboard bishops = board.get_piece_bb(PieceType::BISHOP, by_color);
  while (bishops) {
    uint8_t sq = pop_lsb(bishops);
    attacked |= get_ray_attacks(sq, BISHOP_DIRECTIONS, 4, occupied);
  }

  // Rook attacks
  Bitboard rooks = board.get_piece_bb(PieceType::ROOK, by_color);
  while (rooks) {
    uint8_t sq = pop_lsb(rooks);
    attacked |= get_ray_attacks(sq, ROOK_DIRECTIONS, 4, occupied);
  }

  // Queen attacks (rook + bishop)
  Bitboard queens = board.get_piece_bb(PieceType::QUEEN, by_color);
  while (queens) {
    uint8_t sq = pop_lsb(queens);
    attacked |= get_ray_attacks(sq, QUEEN_DIRECTIONS, 8, occupied);
  }

  // King attacks - directly use the lookup table
  Bitboard king = board.get_piece_bb(PieceType::KING, by_color);
  if (king) {
    uint8_t sq = __builtin_ctzll(king);
    attacked |= KING_ATTACKS[sq];
  }

  return attacked;
}

// Complete check detection implementation - optimized
bool is_in_check(const Board& board, Color us) noexcept {
  // Find king's square
  Bitboard king_bb = board.get_piece_bb(PieceType::KING, us);
  if (!king_bb) return false;  // No king (edge case)
  uint8_t king_sq = __builtin_ctzll(king_bb);

  Color them = (us == Color::WHITE) ? Color::BLACK : Color::WHITE;
  Bitboard occupied = board.get_all_pieces_bb();
  Bitboard king_square_bb = 1ULL << king_sq;

  // Check for pawn attacks - using lookup table
  Bitboard enemy_pawns = board.get_piece_bb(PieceType::PAWN, them);
  if (PAWN_ATTACKS[static_cast<uint8_t>(us)][king_sq] & enemy_pawns)
    return true;

  // Check for knight attacks - using lookup table
  Bitboard enemy_knights = board.get_piece_bb(PieceType::KNIGHT, them);
  if (KNIGHT_ATTACKS[king_sq] & enemy_knights) return true;

  // Check for sliding piece attacks (bishops, rooks, queens)
  // Bishop/Queen diagonal attacks
  Bitboard enemy_bishops = board.get_piece_bb(PieceType::BISHOP, them) |
                           board.get_piece_bb(PieceType::QUEEN, them);
  while (enemy_bishops) {
    uint8_t bishop_sq = pop_lsb(enemy_bishops);
    if (get_ray_attacks(bishop_sq, BISHOP_DIRECTIONS, 4, occupied) &
        king_square_bb) {
      return true;
    }
  }

  // Rook/Queen horizontal/vertical attacks
  Bitboard enemy_rooks = board.get_piece_bb(PieceType::ROOK, them) |
                         board.get_piece_bb(PieceType::QUEEN, them);
  while (enemy_rooks) {
    uint8_t rook_sq = pop_lsb(enemy_rooks);
    if (get_ray_attacks(rook_sq, ROOK_DIRECTIONS, 4, occupied) &
        king_square_bb) {
      return true;
    }
  }

  // Check for king attacks - using lookup table
  Bitboard enemy_king = board.get_piece_bb(PieceType::KING, them);
  if (enemy_king &&
      (KING_ATTACKS[__builtin_ctzll(enemy_king)] & king_square_bb)) {
    return true;
  }

  return false;
}

// Castling legality - optimized
bool is_castling_legal(const Board& board, Color us, bool kingside) noexcept {
  // Fast check for in-check first
  if (is_in_check(board, us)) return false;

  Color them = (us == Color::WHITE) ? Color::BLACK : Color::WHITE;
  Bitboard occupied = board.get_all_pieces_bb();

  if (kingside) {
    // Kingside: f1/f8 and g1/g8 must be empty and not attacked
    const Bitboard ks_path = (us == Color::WHITE) ? 0x0000000000000060ULL   // f1, g1 (bits 5,6)
                                                  : 0x6000000000000000ULL;  // f8, g8 (bits 61,62)
    return (ks_path & occupied) == 0 &&
           (ks_path & board.get_attacked_squares(them)) == 0;
  } else {
    // Queenside: b1/b8, c1/c8, d1/d8 must be empty; c1/c8, d1/d8 must not be attacked
    const Bitboard qs_path_empty = (us == Color::WHITE) ? 0x000000000000000EULL   // b1, c1, d1 (bits 1,2,3)
                                                        : 0x0E00000000000000ULL;  // b8, c8, d8 (bits 57,58,59)
    const Bitboard qs_path_safe = (us == Color::WHITE) ? 0x000000000000000CULL    // c1, d1 (bits 2,3)
                                                       : 0x0C00000000000000ULL;   // c8, d8 (bits 58,59)
    return (qs_path_empty & occupied) == 0 &&
           (qs_path_safe & board.get_attacked_squares(them)) == 0;
  }
}

// Helper function to check if two squares are on the same ray - optimized
bool squares_on_same_ray(uint8_t sq1, uint8_t sq2) noexcept {
  int r1 = sq1 / 8, f1 = sq1 % 8;
  int r2 = sq2 / 8, f2 = sq2 % 8;

  // Same rank
  if (r1 == r2) return true;

  // Same file
  if (f1 == f2) return true;

  // Same diagonal
  int rank_diff = r2 - r1;
  int file_diff = f2 - f1;
  return std::abs(rank_diff) == std::abs(file_diff);
}

// Fixed pinned pieces computation - optimized
// Fast move legality checking without board copies
bool is_move_legal_fast(const Board& board, const Move& move, Color us,
                       uint8_t king_sq, Bitboard pinned,
                       const std::array<Bitboard, 64>& pin_rays) noexcept {
  uint8_t from = move.from;
  uint8_t to = move.to;

  // King moves: check if destination square is attacked
  if (from == king_sq) {
    Bitboard attacked = get_attacked_squares(board, (us == Color::WHITE) ? Color::BLACK : Color::WHITE);
    return (attacked & (1ULL << to)) == 0;
  }

  // If piece is pinned, move must stay on pin ray
  if ((1ULL << from) & pinned) {
    return (pin_rays[from] & (1ULL << to)) != 0;
  }

  // For en passant captures, need special handling
  if (board.en_passant_square != -1 && to == board.en_passant_square) {
    // Simplified check: if the move captures en passant, verify it doesn't leave king in check
    // This is a rare case so we can afford a bit more computation
    Board temp = board;
    temp.make_move(move);
    return !is_in_check(temp, us);
  }

  // For all other moves: check if they don't leave king in check
  // Use incremental check detection
  return !does_move_leave_king_in_check(board, move, us, king_sq);
}

// Helper: Check if a move leaves king in check without full board copy
bool does_move_leave_king_in_check(const Board& board, const Move& move,
                                  Color us, uint8_t king_sq) noexcept {
  // For now, use optimized approach: pre-compute attacked squares and check incrementally
  // This is still a performance optimization over full board copy
  Color them = (us == Color::WHITE) ? Color::BLACK : Color::WHITE;

  // Get attacked squares by enemy before move
  Bitboard attacked_before = board.get_attacked_squares(them);

  // If king is attacked before move, move is illegal unless it's a king move that escapes
  if (attacked_before & (1ULL << king_sq)) {
    // King is in check - only king moves can possibly escape
    if (move.from != king_sq) return true;

    // Check if king move escapes to safe square
    return (attacked_before & (1ULL << move.to)) != 0;
  }

  // King not in check - check if move exposes king to attack
  // This requires simulating the move's effect on attacked squares
  // For simplicity and correctness, fall back to board copy for now
  // TODO: Implement full incremental attacked square updates
  Board temp = board;
  temp.make_move(move);
  return is_in_check(temp, us);
}

std::pair<Bitboard, std::array<Bitboard, 64>> compute_pinned_pieces(
    const Board& board, Color us) noexcept {
  Bitboard pinned = 0;
  std::array<Bitboard, 64> pin_rays = {0};

  Color them = (us == Color::WHITE) ? Color::BLACK : Color::WHITE;
  Bitboard occupied = board.get_all_pieces_bb();
  Bitboard our_pieces = board.get_color_bb(us);
  Bitboard king_bb = board.get_piece_bb(PieceType::KING, us);
  if (!king_bb) return {0, pin_rays};

  uint8_t king_sq = __builtin_ctzll(king_bb);

  // Check rook/queen pins
  Bitboard potential_pinners = board.get_piece_bb(PieceType::ROOK, them) |
                               board.get_piece_bb(PieceType::QUEEN, them);
  while (potential_pinners) {
    uint8_t pinner_sq = pop_lsb(potential_pinners);

    // Check if pinner and king are aligned horizontally or vertically
    int pinner_r = pinner_sq / 8, pinner_f = pinner_sq % 8;
    int king_r = king_sq / 8, king_f = king_sq % 8;

    // If not on same rank or file, continue
    if (pinner_r != king_r && pinner_f != king_f) continue;

    // Determine step direction
    int step =
        (pinner_r == king_r) ? ((pinner_f < king_f) ? 1 : -1)
                             :               // Same rank: move horizontally
            ((pinner_r < king_r) ? 8 : -8);  // Same file: move vertically

    // Create mask for squares between pinner and king
    Bitboard between_mask = 0;
    for (uint8_t sq = pinner_sq + step; sq != king_sq; sq += step) {
      between_mask |= (1ULL << sq);
    }

    // Check ALL pieces between pinner and king
    Bitboard all_pieces_between = between_mask & occupied;
    Bitboard our_pieces_between = all_pieces_between & our_pieces;
    
    // A piece is pinned only if:
    // 1. There's exactly one of OUR pieces between pinner and king
    // 2. That piece is the ONLY piece between them (no enemy pieces blocking)
    if (popcount(our_pieces_between) == 1 && popcount(all_pieces_between) == 1) {
      uint8_t pinned_sq = __builtin_ctzll(our_pieces_between);
      pinned |= (1ULL << pinned_sq);

      // Pin ray: from king through pinned piece to edge (more inclusive)
      // IMPORTANT: Add the pinner's square to the pin ray to allow capturing
      // the pinner
      pin_rays[pinned_sq] =
          (get_ray_attacks(king_sq, ROOK_DIRECTIONS, 4, 0) &
           get_ray_attacks(pinner_sq, ROOK_DIRECTIONS, 4, 0)) |
          (1ULL << pinner_sq);
    }
  }

  // Check bishop/queen pins - optimized
  potential_pinners = board.get_piece_bb(PieceType::BISHOP, them) |
                      board.get_piece_bb(PieceType::QUEEN, them);
  while (potential_pinners) {
    uint8_t pinner_sq = pop_lsb(potential_pinners);

    // Check if on same diagonal
    int pinner_r = pinner_sq / 8, pinner_f = pinner_sq % 8;
    int king_r = king_sq / 8, king_f = king_sq % 8;

    int r_diff = king_r - pinner_r;
    int f_diff = king_f - pinner_f;

    // If not on same diagonal, continue
    if (std::abs(r_diff) != std::abs(f_diff)) continue;

    // Determine diagonal step
    int r_step = (r_diff > 0) ? 1 : -1;
    int f_step = (f_diff > 0) ? 1 : -1;
    int step = r_step * 8 + f_step;

    // Create mask for squares between pinner and king
    Bitboard between_mask = 0;
    for (uint8_t sq = pinner_sq + step; sq != king_sq; sq += step) {
      between_mask |= (1ULL << sq);
    }

    // Check ALL pieces between pinner and king
    Bitboard all_pieces_between = between_mask & occupied;
    Bitboard our_pieces_between = all_pieces_between & our_pieces;
    
    // A piece is pinned only if:
    // 1. There's exactly one of OUR pieces between pinner and king
    // 2. That piece is the ONLY piece between them (no enemy pieces blocking)
    if (popcount(our_pieces_between) == 1 && popcount(all_pieces_between) == 1) {
      uint8_t pinned_sq = __builtin_ctzll(our_pieces_between);
      pinned |= (1ULL << pinned_sq);

      // Pin ray: from king through pinned piece to edge
      // IMPORTANT: Add the pinner's square to the pin ray to allow capturing
      // the pinner
      pin_rays[pinned_sq] =
          (get_ray_attacks(king_sq, BISHOP_DIRECTIONS, 4, 0) &
           get_ray_attacks(pinner_sq, BISHOP_DIRECTIONS, 4, 0)) |
          (1ULL << pinner_sq);
    }
  }

  return {pinned, pin_rays};
}

// Main generate_legal_moves implementation - now with fixed pin handling and
// optimized
std::vector<Move> Board::generate_legal_moves() const noexcept {
  // Pre-allocate with exact size for common case (avg ~38 moves)
  std::vector<Move> legal_moves;
  legal_moves.reserve(48);

  Color us = side_to_move ? Color::WHITE : Color::BLACK;
  Color them = (us == Color::WHITE) ? Color::BLACK : Color::WHITE;
  Bitboard our_pieces = get_color_bb(us);
  Bitboard their_pieces = get_color_bb(them);
  Bitboard occupied = get_all_pieces_bb();
  Bitboard empty = ~occupied;

  // Compute pinned pieces and pin rays
  auto [pinned, pin_rays] = compute_pinned_pieces(*this, us);

  // King's square for fast lookup
  Bitboard king_bb = get_piece_bb(PieceType::KING, us);
  uint8_t king_sq = king_bb ? __builtin_ctzll(king_bb) : 0xFF;

  // Helper lambda to add moves from a source square, with legality checking
  auto add_legal_moves = [&](uint8_t from, Bitboard targets,
                             uint8_t promotion = 0) {
    while (targets) {
      uint8_t to = pop_lsb(targets);
      Move test_move = {from, to, promotion};

      // Fast legality check: only test if move doesn't leave king in check
      if (is_move_legal_fast(test_move, us, king_sq, pinned, pin_rays)) {
        legal_moves.push_back(test_move);
      }
    }
  };

  // Pawn moves
  Bitboard pawns = get_piece_bb(PieceType::PAWN, us);
  int direction = (us == Color::WHITE) ? 8 : -8;
  int start_rank = (us == Color::WHITE) ? 1 : 6;
  uint8_t us_idx = static_cast<uint8_t>(us);

  while (pawns) {
    uint8_t from = pop_lsb(pawns);
    int from_rank = from / 8;

    Bitboard allowed_targets = (1ULL << from) & pinned ? pin_rays[from] : ~0ULL;

    // Single push
    uint8_t to = from + direction;
    if (to < 64 && ((1ULL << to) & empty)) {
      Bitboard single_push = (1ULL << to) & allowed_targets;

      if (single_push) {
        if (from_rank ==
            (us == Color::WHITE ? 6 : 1)) {  // 7th/2nd rank (promotion next)
          // Promotion - add all four promotion types
          for (uint8_t pt = 1; pt <= 4; pt++) {  // KNIGHT through QUEEN
            add_legal_moves(from, single_push, pt);
          }
        } else {
          add_legal_moves(from, single_push);

          // Double push from start rank (more efficient check)
          if (from_rank == start_rank) {
            uint8_t double_to = to + direction;
            if ((1ULL << double_to) & empty & allowed_targets) {
              add_legal_moves(from, 1ULL << double_to);
            }
          }
        }
      }
    }

    // Captures - using lookup tables
    Bitboard captures =
        PAWN_ATTACKS[us_idx][from] & their_pieces & allowed_targets;

    if (from_rank ==
        (us == Color::WHITE ? 6 : 1)) {  // 7th/2nd rank (promotion next)
      // Promotion captures - optimized to avoid unnecessary loops
      while (captures) {
        uint8_t cap_to = pop_lsb(captures);
        Bitboard cap_bb = 1ULL << cap_to;

        // Add all promotion types in one loop
        for (uint8_t pt = 1; pt <= 4; pt++) {  // KNIGHT through QUEEN
          add_legal_moves(from, cap_bb, pt);
        }
      }
    } else if (captures) {
      add_legal_moves(from, captures);
    }

    // En passant - fast check
    if (en_passant_square != -1) {
      Bitboard ep_attacks = PAWN_ATTACKS[us_idx][from] &
                            (1ULL << en_passant_square) & allowed_targets;
      if (ep_attacks) add_legal_moves(from, ep_attacks);
    }
  }

  // Knight moves - using lookup table
  Bitboard knights = get_piece_bb(PieceType::KNIGHT, us);
  while (knights) {
    uint8_t from = pop_lsb(knights);

    // Knights can't move along pin ray, so just check if pinned
    if ((1ULL << from) & pinned) continue;

    Bitboard targets = KNIGHT_ATTACKS[from] & ~our_pieces;
    add_legal_moves(from, targets);
  }

  // King moves (non-castling, using precomputed table)
  if (king_bb) {
    uint8_t from = __builtin_ctzll(king_bb);
    Bitboard targets = KING_ATTACKS[from] & ~our_pieces;

    // Add all king moves in one go - they're never pinned
    add_legal_moves(from, targets);
  }

  // Castling - complete implementation with all legality checks
  if (king_bb &&
      !is_in_check(*this, us)) {  // Quick check if castling is even possible
    uint8_t king_home = (us == Color::WHITE) ? 4 : 60;
    if (king_sq == king_home) {  // King on home square
      // Kingside castling
      uint8_t rights_mask_ks = (us == Color::WHITE) ? 1 : 4;
      if (castling_rights & rights_mask_ks) {
        Bitboard ks_path = (us == Color::WHITE) ? 0x0000000000000060ULL
                                                : 0x6000000000000000ULL;
        if ((ks_path & occupied) == 0 &&
            (ks_path & get_attacked_squares(*this, them)) == 0) {
          legal_moves.push_back(
              {king_home, static_cast<uint8_t>(king_home + 2), 0});
        }
      }

      // Queenside castling
      uint8_t rights_mask_qs = (us == Color::WHITE) ? 2 : 8;
      if (castling_rights & rights_mask_qs) {
        // King passes through d1/d8 and lands on c1/c8 (must not be attacked)
        Bitboard qs_path_king = (us == Color::WHITE) ? 0x000000000000000CULL   // c1, d1 (bits 2,3)
                                                     : 0x0C00000000000000ULL;  // c8, d8 (bits 58,59)
        // Squares between king and rook must be empty: b1/b8, c1/c8, d1/d8
        Bitboard qs_path_all = (us == Color::WHITE) ? 0x000000000000000EULL    // b1, c1, d1 (bits 1,2,3)
                                                    : 0x0E00000000000000ULL;   // b8, c8, d8 (bits 57,58,59)

        // Path for king must be empty and not attacked, rook path must be empty
        if ((qs_path_all & occupied) == 0 &&
            (qs_path_king & get_attacked_squares(*this, them)) == 0) {
          legal_moves.push_back(
              {king_home, static_cast<uint8_t>(king_home - 2), 0});
        }
      }
    }
  }

  // Sliding pieces (rooks, bishops, queens)
  // Rooks
  Bitboard rooks = get_piece_bb(PieceType::ROOK, us);
  while (rooks) {
    uint8_t from = pop_lsb(rooks);
    Bitboard allowed_targets = (1ULL << from) & pinned ? pin_rays[from] : ~0ULL;
    Bitboard attacks = get_ray_attacks(from, ROOK_DIRECTIONS, 4, occupied);
    Bitboard targets = attacks & ~our_pieces & allowed_targets;
    add_legal_moves(from, targets);
  }

  // Bishops
  Bitboard bishops = get_piece_bb(PieceType::BISHOP, us);
  while (bishops) {
    uint8_t from = pop_lsb(bishops);
    Bitboard allowed_targets = (1ULL << from) & pinned ? pin_rays[from] : ~0ULL;
    Bitboard attacks = get_ray_attacks(from, BISHOP_DIRECTIONS, 4, occupied);
    Bitboard targets = attacks & ~our_pieces & allowed_targets;
    add_legal_moves(from, targets);
  }

  // Queens (combine rook + bishop rays)
  Bitboard queens = get_piece_bb(PieceType::QUEEN, us);
  while (queens) {
    uint8_t from = pop_lsb(queens);
    Bitboard allowed_targets = (1ULL << from) & pinned ? pin_rays[from] : ~0ULL;
    Bitboard attacks = get_ray_attacks(from, QUEEN_DIRECTIONS, 8, occupied);
    Bitboard targets = attacks & ~our_pieces & allowed_targets;
    add_legal_moves(from, targets);
  }

  return legal_moves;
}