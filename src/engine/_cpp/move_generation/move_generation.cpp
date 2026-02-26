#include "move_generation.hpp"

#include <array>
#include <cstdint>
#include <vector>

#include "../board/board.hpp"

// ---------------------------------------------------------------------------
// Precomputed attack tables
// All tables are initialised once at program start via IIFE lambdas.
// ---------------------------------------------------------------------------

// Knight attack masks — one bitboard per square.
const std::array<Bitboard, 64> KNIGHT_ATTACKS = []() {
  std::array<Bitboard, 64> a{};
  for (int sq = 0; sq < 64; ++sq) {
    const int r = sq / 8, f = sq % 8;
    const int deltas[8][2] = {{2, 1}, {2, -1}, {-2, 1}, {-2, -1},
                              {1, 2}, {1, -2}, {-1, 2}, {-1, -2}};
    for (const auto &d : deltas) {
      const int nr = r + d[0], nf = f + d[1];
      if (nr >= 0 && nr < 8 && nf >= 0 && nf < 8)
        a[sq] |= 1ULL << (nr * 8 + nf);
    }
  }
  return a;
}();

// Pawn attack masks — indexed [color_index][square].
// color_index 0 = WHITE (attacks toward higher ranks), 1 = BLACK (lower ranks).
const std::array<std::array<Bitboard, 64>, 2> PAWN_ATTACKS = []() {
  std::array<std::array<Bitboard, 64>, 2> a{};
  for (int sq = 0; sq < 64; ++sq) {
    const int r = sq / 8, f = sq % 8;
    if (r < 7) {
      if (f > 0)
        a[0][sq] |= 1ULL << ((r + 1) * 8 + (f - 1));
      if (f < 7)
        a[0][sq] |= 1ULL << ((r + 1) * 8 + (f + 1));
    }
    if (r > 0) {
      if (f > 0)
        a[1][sq] |= 1ULL << ((r - 1) * 8 + (f - 1));
      if (f < 7)
        a[1][sq] |= 1ULL << ((r - 1) * 8 + (f + 1));
    }
  }
  return a;
}();

// King attack masks.
const std::array<Bitboard, 64> KING_ATTACKS = []() {
  std::array<Bitboard, 64> a{};
  for (int sq = 0; sq < 64; ++sq) {
    const int r = sq / 8, f = sq % 8;
    const int deltas[8][2] = {{1, 0}, {-1, 0}, {0, 1},  {0, -1},
                              {1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    for (const auto &d : deltas) {
      const int nr = r + d[0], nf = f + d[1];
      if (nr >= 0 && nr < 8 && nf >= 0 && nf < 8)
        a[sq] |= 1ULL << (nr * 8 + nf);
    }
  }
  return a;
}();

// Slider direction index sets.
// Direction ordering: 0=N(+8), 1=S(-8), 2=E(+1), 3=W(-1),
//                     4=NE(+9), 5=SW(-9), 6=NW(+7), 7=SE(-7).
// Even indices advance in positive bit-index direction → find closest blocker
// with LSB (ctz64).  Odd indices advance in negative direction → MSB
// (63-clz64).
const int ROOK_DIRECTIONS[4] = {0, 1, 2, 3};
const int BISHOP_DIRECTIONS[4] = {4, 5, 6, 7};
const int QUEEN_DIRECTIONS[8] = {0, 1, 2, 3, 4, 5, 6, 7};

// RAY_ATTACKS[sq][dir] — squares reachable from sq along dir, empty board,
// exclusive of sq itself.
const std::array<std::array<Bitboard, 8>, 64> RAY_ATTACKS = []() {
  std::array<std::array<Bitboard, 8>, 64> a{};
  const int offsets[8] = {8, -8, 1, -1, 9, -9, 7, -7};

  for (int sq = 0; sq < 64; ++sq) {
    for (int d = 0; d < 8; ++d) {
      const int off = offsets[d];
      int cur = sq;
      int cr = sq / 8, cf = sq % 8;

      while (true) {
        cur += off;
        if (cur < 0 || cur >= 64)
          break;
        const int tr = cur / 8, tf = cur % 8;
        // Guard against wrap-around (rank or file jumps more than 1).
        const int dr = tr - cr, df = tf - cf;
        const int adr = dr < 0 ? -dr : dr;
        const int adf = df < 0 ? -df : df;
        if (adr > 1 || adf > 1)
          break;
        a[sq][d] |= 1ULL << cur;
        cr = tr;
        cf = tf;
      }
    }
  }
  return a;
}();

// BETWEEN_SQUARES[a][b] — bitmask of squares strictly between a and b along
// their shared rank, file, or diagonal.  Zero when a and b are not aligned.
const std::array<std::array<Bitboard, 64>, 64> BETWEEN_SQUARES = []() {
  std::array<std::array<Bitboard, 64>, 64> b{};
  for (int s1 = 0; s1 < 64; ++s1) {
    for (int s2 = 0; s2 < 64; ++s2) {
      if (s1 == s2)
        continue;
      const int r1 = s1 / 8, f1 = s1 % 8;
      const int r2 = s2 / 8, f2 = s2 % 8;
      const int dr = r2 - r1, df = f2 - f1;
      const int adr = dr < 0 ? -dr : dr;
      const int adf = df < 0 ? -df : df;

      const bool aligned = (r1 == r2) || (f1 == f2) || (adr == adf);
      if (!aligned)
        continue;

      const int step_r = (dr == 0) ? 0 : (dr > 0 ? 1 : -1);
      const int step_f = (df == 0) ? 0 : (df > 0 ? 1 : -1);
      const int step = step_r * 8 + step_f;

      Bitboard mask = 0;
      int sq = s1 + step;
      while (sq != s2) {
        mask |= 1ULL << sq;
        sq += step;
      }
      b[s1][s2] = mask;
    }
  }
  return b;
}();

// ---------------------------------------------------------------------------
// Sliding-piece ray attacks with blocker handling
// ---------------------------------------------------------------------------

// Returns the set of squares attacked along the given directions from sq,
// treating the first occupied square in each direction as included (capturable)
// and blocking further travel.
Bitboard get_ray_attacks(int sq, const int *directions, int num_dirs,
                         Bitboard occupied) noexcept {
  Bitboard attacks = 0;
  for (int i = 0; i < num_dirs; ++i) {
    const int d = directions[i];
    Bitboard ray = RAY_ATTACKS[sq][d];
    const Bitboard blockers = ray & occupied;
    if (blockers) {
      // The closest blocker terminates the ray.  For positive directions
      // (even d) use the LSB; for negative directions (odd d) use the MSB.
      const int blocker_sq =
          (d % 2 == 0) ? ctz64(blockers) : (63 - clz64(blockers));
      // Keep the ray from sq up to and including the blocker, excluding the
      // rest of the shadow.  RAY_ATTACKS[blocker_sq][d] holds the "shadow"
      // (squares past the blocker in the same direction), so:
      //   attacks |= ray ^ shadow  = ray XOR (ray & shadow)
      attacks |= ray ^ RAY_ATTACKS[blocker_sq][d];
    } else {
      attacks |= ray;
    }
  }
  return attacks;
}

// ---------------------------------------------------------------------------
// Attacked-square computation (parametrised occupancy)
// ---------------------------------------------------------------------------

Bitboard attacked_by(const Board &board, Color by_color,
                     Bitboard occ) noexcept {
  Bitboard attacked = 0;
  const uint8_t ci = static_cast<uint8_t>(by_color);

  Bitboard pawns = board.get_piece_bb(PieceType::PAWN, by_color);
  while (pawns)
    attacked |= PAWN_ATTACKS[ci][pop_lsb(pawns)];

  Bitboard knights = board.get_piece_bb(PieceType::KNIGHT, by_color);
  while (knights)
    attacked |= KNIGHT_ATTACKS[pop_lsb(knights)];

  Bitboard bishops = board.get_piece_bb(PieceType::BISHOP, by_color);
  while (bishops)
    attacked |= get_ray_attacks(pop_lsb(bishops), BISHOP_DIRECTIONS, 4, occ);

  Bitboard rooks = board.get_piece_bb(PieceType::ROOK, by_color);
  while (rooks)
    attacked |= get_ray_attacks(pop_lsb(rooks), ROOK_DIRECTIONS, 4, occ);

  Bitboard queens = board.get_piece_bb(PieceType::QUEEN, by_color);
  while (queens)
    attacked |= get_ray_attacks(pop_lsb(queens), QUEEN_DIRECTIONS, 8, occ);

  const Bitboard king = board.get_piece_bb(PieceType::KING, by_color);
  if (king)
    attacked |= KING_ATTACKS[ctz64(king)];

  return attacked;
}

// ---------------------------------------------------------------------------
// Check and checker detection
// ---------------------------------------------------------------------------

// Determines whether the king of color us is in check by casting attack rays
// outward from the king square.  One set of ray casts covers all slider threats
// regardless of how many enemy sliders there are.
bool is_in_check(const Board &board, Color us) noexcept {
  const Bitboard king_bb = board.get_piece_bb(PieceType::KING, us);
  if (!king_bb)
    return false;
  const uint8_t king_sq = ctz64(king_bb);
  const Color them = (us == Color::WHITE) ? Color::BLACK : Color::WHITE;
  const Bitboard occ = board.get_all_pieces_bb();

  if (PAWN_ATTACKS[static_cast<uint8_t>(us)][king_sq] &
      board.get_piece_bb(PieceType::PAWN, them))
    return true;

  if (KNIGHT_ATTACKS[king_sq] & board.get_piece_bb(PieceType::KNIGHT, them))
    return true;

  const Bitboard diag_enemies = board.get_piece_bb(PieceType::BISHOP, them) |
                                board.get_piece_bb(PieceType::QUEEN, them);
  if (get_ray_attacks(king_sq, BISHOP_DIRECTIONS, 4, occ) & diag_enemies)
    return true;

  const Bitboard ortho_enemies = board.get_piece_bb(PieceType::ROOK, them) |
                                 board.get_piece_bb(PieceType::QUEEN, them);
  if (get_ray_attacks(king_sq, ROOK_DIRECTIONS, 4, occ) & ortho_enemies)
    return true;

  if (KING_ATTACKS[king_sq] & board.get_piece_bb(PieceType::KING, them))
    return true;

  return false;
}

// Returns a bitboard of all enemy pieces that are currently giving check to
// the king.  Callers must supply king_sq and occ to avoid redundant queries.
Bitboard get_checkers(const Board &board, Color us, uint8_t king_sq,
                      Bitboard occ) noexcept {
  const Color them = (us == Color::WHITE) ? Color::BLACK : Color::WHITE;
  const uint8_t ui = static_cast<uint8_t>(us);
  Bitboard checkers = 0;

  checkers |=
      PAWN_ATTACKS[ui][king_sq] & board.get_piece_bb(PieceType::PAWN, them);
  checkers |=
      KNIGHT_ATTACKS[king_sq] & board.get_piece_bb(PieceType::KNIGHT, them);

  const Bitboard diag_att = get_ray_attacks(king_sq, BISHOP_DIRECTIONS, 4, occ);
  checkers |= diag_att & (board.get_piece_bb(PieceType::BISHOP, them) |
                          board.get_piece_bb(PieceType::QUEEN, them));

  const Bitboard ortho_att = get_ray_attacks(king_sq, ROOK_DIRECTIONS, 4, occ);
  checkers |= ortho_att & (board.get_piece_bb(PieceType::ROOK, them) |
                           board.get_piece_bb(PieceType::QUEEN, them));

  return checkers;
}

// ---------------------------------------------------------------------------
// Castling legality
// ---------------------------------------------------------------------------

bool is_castling_legal(const Board &board, Color us, bool kingside) noexcept {
  if (is_in_check(board, us))
    return false;

  const Color them = (us == Color::WHITE) ? Color::BLACK : Color::WHITE;
  const Bitboard occ = board.get_all_pieces_bb();

  if (kingside) {
    const Bitboard path =
        (us == Color::WHITE) ? 0x0000000000000060ULL : 0x6000000000000000ULL;
    return (path & occ) == 0 &&
           (path & compute_attacked_squares(board, them)) == 0;
  } else {
    const Bitboard empty_path =
        (us == Color::WHITE) ? 0x000000000000000EULL : 0x0E00000000000000ULL;
    const Bitboard safe_path =
        (us == Color::WHITE) ? 0x000000000000000CULL : 0x0C00000000000000ULL;
    return (empty_path & occ) == 0 &&
           (safe_path & compute_attacked_squares(board, them)) == 0;
  }
}

// ---------------------------------------------------------------------------
// Pin detection using outward ray scans from the king
// ---------------------------------------------------------------------------

std::pair<Bitboard, std::array<Bitboard, 64>>
compute_pinned_pieces(const Board &board, Color us) noexcept {
  Bitboard pinned = 0;
  std::array<Bitboard, 64> pin_rays{};

  const Color them = (us == Color::WHITE) ? Color::BLACK : Color::WHITE;
  const Bitboard our_bb = board.get_color_bb(us);
  const Bitboard occ = board.get_all_pieces_bb();
  const Bitboard king_bb = board.get_piece_bb(PieceType::KING, us);
  if (!king_bb)
    return {0, pin_rays};
  const uint8_t king_sq = ctz64(king_bb);

  // Orthogonal pin rays (rook / queen pinners).
  const Bitboard rook_attackers = board.get_piece_bb(PieceType::ROOK, them) |
                                  board.get_piece_bb(PieceType::QUEEN, them);
  for (int d = 0; d < 4; ++d) {
    const Bitboard ray = RAY_ATTACKS[king_sq][d];
    const Bitboard blockers = ray & occ;
    if (!blockers)
      continue;

    // Closest piece in direction d.
    const uint8_t first_sq = (d % 2 == 0)
                                 ? ctz64(blockers)
                                 : static_cast<uint8_t>(63 - clz64(blockers));
    if (!((1ULL << first_sq) & our_bb))
      continue; // enemy piece — not a pin candidate

    // Next piece past first_sq in the same direction.
    const Bitboard ray2 = RAY_ATTACKS[first_sq][d];
    const Bitboard blockers2 = ray2 & occ;
    if (!blockers2)
      continue;

    const uint8_t second_sq = (d % 2 == 0)
                                  ? ctz64(blockers2)
                                  : static_cast<uint8_t>(63 - clz64(blockers2));
    if (!((1ULL << second_sq) & rook_attackers))
      continue;

    pinned |= 1ULL << first_sq;
    pin_rays[first_sq] =
        BETWEEN_SQUARES[king_sq][second_sq] | (1ULL << second_sq);
  }

  // Diagonal pin rays (bishop / queen pinners).
  const Bitboard bishop_attackers =
      board.get_piece_bb(PieceType::BISHOP, them) |
      board.get_piece_bb(PieceType::QUEEN, them);
  for (int d = 0; d < 4; ++d) {
    const int dir_idx = BISHOP_DIRECTIONS[d]; // 4, 5, 6, or 7
    const Bitboard ray = RAY_ATTACKS[king_sq][dir_idx];
    const Bitboard blockers = ray & occ;
    if (!blockers)
      continue;

    const uint8_t first_sq = (dir_idx % 2 == 0)
                                 ? ctz64(blockers)
                                 : static_cast<uint8_t>(63 - clz64(blockers));
    if (!((1ULL << first_sq) & our_bb))
      continue;

    const Bitboard ray2 = RAY_ATTACKS[first_sq][dir_idx];
    const Bitboard blockers2 = ray2 & occ;
    if (!blockers2)
      continue;

    const uint8_t second_sq = (dir_idx % 2 == 0)
                                  ? ctz64(blockers2)
                                  : static_cast<uint8_t>(63 - clz64(blockers2));
    if (!((1ULL << second_sq) & bishop_attackers))
      continue;

    pinned |= 1ULL << first_sq;
    pin_rays[first_sq] =
        BETWEEN_SQUARES[king_sq][second_sq] | (1ULL << second_sq);
  }

  return {pinned, pin_rays};
}

// ---------------------------------------------------------------------------
// Legal move generation — no board copies required for the common case
// ---------------------------------------------------------------------------

std::vector<Move> Board::generate_legal_moves() const noexcept {
  std::vector<Move> moves;
  moves.reserve(48);

  const Color us = side_to_move ? Color::WHITE : Color::BLACK;
  const Color them = (us == Color::WHITE) ? Color::BLACK : Color::WHITE;
  const uint8_t us_idx = static_cast<uint8_t>(us);

  const Bitboard our_bb = get_color_bb(us);
  const Bitboard their_bb = get_color_bb(them);
  const Bitboard occ = get_all_pieces_bb();
  const Bitboard empty = ~occ;

  // Locate our king.
  const Bitboard king_bb = get_piece_bb(PieceType::KING, us);
  if (!king_bb)
    return moves; // Defensive: no king means no legal moves
  const uint8_t king_sq = ctz64(king_bb);

  // Compute danger squares for the king.  The king is removed from the
  // occupancy so that sliding attackers extend through its square, preventing
  // the king from stepping into the line of fire of a rook or queen even when
  // it moves along the attack ray.
  const Bitboard king_danger = attacked_by(*this, them, occ ^ king_bb);

  // King moves — added before the check-count test so they are always present.
  {
    Bitboard tgts = KING_ATTACKS[king_sq] & ~our_bb & ~king_danger;
    while (tgts)
      moves.push_back({king_sq, pop_lsb(tgts), 0});
  }

  // Determine the number of pieces giving check.
  const Bitboard checkers = get_checkers(*this, us, king_sq, occ);
  const int num_checkers = popcount(checkers);

  // Double check: only king moves are legal (already generated above).
  if (num_checkers == 2)
    return moves;

  // Build the target mask for non-king moves.
  // Not in check: any square that is not occupied by a friendly piece.
  // Single check: must capture the checker or interpose on the check ray.
  Bitboard target_mask = ~our_bb;
  if (num_checkers == 1) {
    const uint8_t checker_sq = ctz64(checkers);
    target_mask = checkers | BETWEEN_SQUARES[king_sq][checker_sq];
  }

  // Compute absolute pins.
  auto [pinned, pin_rays] = compute_pinned_pieces(*this, us);

  // Helper that adds moves from `from` to each square in `raw`, filtered by
  // both the target mask and any pin constraint.
  const auto add_moves = [&](uint8_t from, Bitboard raw, uint8_t promo = 0) {
    Bitboard tgts = raw & target_mask;
    if ((1ULL << from) & pinned)
      tgts &= pin_rays[from];
    while (tgts)
      moves.push_back({from, pop_lsb(tgts), promo});
  };

  // --- Pawn moves ---
  {
    Bitboard pawns = get_piece_bb(PieceType::PAWN, us);
    const int push_dir = (us_idx == 0u) ? 8 : -8;
    const int start_rank = (us_idx == 0u) ? 1 : 6;
    const int promo_rank = (us_idx == 0u) ? 6 : 1;

    while (pawns) {
      const uint8_t from = pop_lsb(pawns);
      const int from_rank = from / 8;
      const bool on_promo = (from_rank == promo_rank);

      // Pin limit: a pinned pawn may only move along its pin ray.
      const Bitboard pin_limit =
          ((1ULL << from) & pinned) ? pin_rays[from] : ~0ULL;

      // Single push.
      // push1 must be empty (and on the pin ray if pinned).  Whether it is a
      // valid *target* (i.e. in target_mask) determines if the single push is
      // legal.  For the double push the transit through push1 requires only
      // that push1 is empty; the destination push2 must satisfy target_mask.
      const uint8_t push1 =
          static_cast<uint8_t>(static_cast<int>(from) + push_dir);
      const Bitboard push1_bb = 1ULL << push1;
      const bool push1_clear = (push1_bb & empty & pin_limit) != 0;
      if (push1_clear) {
        if (on_promo) {
          // Promotion: push1 must be in target_mask.
          if (push1_bb & target_mask) {
            for (uint8_t pt = 1u; pt <= 4u; ++pt)
              moves.push_back({from, push1, pt});
          }
        } else {
          // Single push: push1 must be in target_mask.
          if (push1_bb & target_mask)
            moves.push_back({from, push1, 0u});
          // Double push from the starting rank: only the destination needs to
          // satisfy target_mask.  The transit square (push1) just needs to be
          // empty, which we already verified above.
          if (from_rank == start_rank) {
            const uint8_t push2 =
                static_cast<uint8_t>(static_cast<int>(push1) + push_dir);
            if ((1ULL << push2) & empty & target_mask & pin_limit)
              moves.push_back({from, push2, 0u});
          }
        }
      }

      // Diagonal captures.
      Bitboard capt =
          PAWN_ATTACKS[us_idx][from] & their_bb & target_mask & pin_limit;
      if (on_promo) {
        while (capt) {
          const uint8_t to = pop_lsb(capt);
          for (uint8_t pt = 1u; pt <= 4u; ++pt)
            moves.push_back({from, to, pt});
        }
      } else {
        while (capt)
          moves.push_back({from, pop_lsb(capt), 0u});
      }

      // En-passant capture.
      if (en_passant_square == -1)
        continue;
      {
        const uint8_t ep_sq = static_cast<uint8_t>(en_passant_square);
        const Bitboard ep_att = PAWN_ATTACKS[us_idx][from] & (1ULL << ep_sq);
        if (!ep_att)
          continue;

        const uint8_t ep_capt_sq =
            static_cast<uint8_t>(ep_sq + (us_idx == 0u ? -8 : 8));
        const Bitboard ep_capt_bb = 1ULL << ep_capt_sq;
        const Bitboard ep_sq_bb = 1ULL << ep_sq;

        // Standard pin constraint.
        if (((1ULL << from) & pinned) && !(pin_rays[from] & ep_sq_bb))
          continue;

        // Single-check evasion constraint: the en-passant must either capture
        // the checking pawn or land on the evasion ray.
        if (num_checkers == 1 && !((ep_capt_bb | ep_sq_bb) & target_mask))
          continue;

        // Simulate the en-passant occupancy to detect horizontal discovered
        // attacks on the king.  Standard pin computation cannot detect this
        // case because both the capturing pawn and the captured pawn are
        // removed from the same rank simultaneously.
        const Bitboard new_occ = (occ ^ (1ULL << from) ^ ep_capt_bb) | ep_sq_bb;
        const Bitboard ortho_att =
            get_ray_attacks(king_sq, ROOK_DIRECTIONS, 4, new_occ);
        const Bitboard diag_att =
            get_ray_attacks(king_sq, BISHOP_DIRECTIONS, 4, new_occ);
        const Bitboard enemy_rq = get_piece_bb(PieceType::ROOK, them) |
                                  get_piece_bb(PieceType::QUEEN, them);
        const Bitboard enemy_bq = get_piece_bb(PieceType::BISHOP, them) |
                                  get_piece_bb(PieceType::QUEEN, them);
        if ((ortho_att & enemy_rq) || (diag_att & enemy_bq))
          continue;

        moves.push_back({from, ep_sq, 0u});
      }
    }
  }

  // --- Knights ---
  // A pinned knight can never move along any pin ray, so pinned knights are
  // skipped entirely.
  {
    Bitboard knights = get_piece_bb(PieceType::KNIGHT, us);
    while (knights) {
      const uint8_t from = pop_lsb(knights);
      if ((1ULL << from) & pinned)
        continue;
      Bitboard tgts = KNIGHT_ATTACKS[from] & ~our_bb & target_mask;
      while (tgts)
        moves.push_back({from, pop_lsb(tgts), 0u});
    }
  }

  // --- Bishops ---
  {
    Bitboard bishops = get_piece_bb(PieceType::BISHOP, us);
    while (bishops) {
      const uint8_t from = pop_lsb(bishops);
      add_moves(from,
                get_ray_attacks(from, BISHOP_DIRECTIONS, 4, occ) & ~our_bb);
    }
  }

  // --- Rooks ---
  {
    Bitboard rooks = get_piece_bb(PieceType::ROOK, us);
    while (rooks) {
      const uint8_t from = pop_lsb(rooks);
      add_moves(from, get_ray_attacks(from, ROOK_DIRECTIONS, 4, occ) & ~our_bb);
    }
  }

  // --- Queens ---
  {
    Bitboard queens = get_piece_bb(PieceType::QUEEN, us);
    while (queens) {
      const uint8_t from = pop_lsb(queens);
      add_moves(from,
                get_ray_attacks(from, QUEEN_DIRECTIONS, 8, occ) & ~our_bb);
    }
  }

  // --- Castling (only when not in check) ---
  if (num_checkers == 0) {
    const uint8_t king_home = (us_idx == 0u) ? 4u : 60u;
    if (king_sq == king_home) {
      // Kingside: f1/g1 (white) or f8/g8 (black) must be empty and safe.
      const uint8_t ks_right = (us_idx == 0u) ? 1u : 4u;
      if (castling_rights & ks_right) {
        const Bitboard ks_path =
            (us_idx == 0u) ? 0x0000000000000060ULL : 0x6000000000000000ULL;
        if ((ks_path & occ) == 0 && (ks_path & king_danger) == 0)
          moves.push_back(
              {king_home, static_cast<uint8_t>(king_home + 2u), 0u});
      }
      // Queenside: b/c/d files must be unoccupied; c/d files must not be
      // attacked.
      const uint8_t qs_right = (us_idx == 0u) ? 2u : 8u;
      if (castling_rights & qs_right) {
        const Bitboard qs_empty =
            (us_idx == 0u) ? 0x000000000000000EULL : 0x0E00000000000000ULL;
        const Bitboard qs_safe =
            (us_idx == 0u) ? 0x000000000000000CULL : 0x0C00000000000000ULL;
        if ((qs_empty & occ) == 0 && (qs_safe & king_danger) == 0)
          moves.push_back(
              {king_home, static_cast<uint8_t>(king_home - 2u), 0u});
      }
    }
  }

  return moves;
}
