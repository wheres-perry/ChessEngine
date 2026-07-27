#include "evaluators.hpp"

#include <algorithm>
#include <array>
#include <cstdint>

namespace evaluators {

namespace {

// ── Piece values (centipawns) ─────────────────────────────────────
constexpr std::array<int, 6> MATERIAL_CP = {
    100, // PAWN
    320, // KNIGHT
    330, // BISHOP
    500, // ROOK
    900, // QUEEN
    0    // KING
};

// ── Phase weights (mirror Python _PHASE_WEIGHTS) ────────────────────
constexpr int PHASE_WEIGHT_QUEEN = 9;
constexpr int PHASE_WEIGHT_ROOK = 5;
constexpr int PHASE_WEIGHT_BISHOP = 3;
constexpr int PHASE_WEIGHT_KNIGHT = 3;
constexpr int MAX_PHASE_MATERIAL = 62;

[[nodiscard]] constexpr double lerp(double mg, double eg,
                                    double phase) noexcept {
  return mg * phase + eg * (1.0 - phase);
}

// Expand a half-board (32 values, ranks 1-4) into a 64-square table in
// pawns, matching Python's make_table helper.
constexpr std::array<double, 64>
make_table(const std::array<int, 32> &values) noexcept {
  std::array<double, 64> result{};
  for (int i = 0; i < 32; ++i) {
    result[i] = static_cast<double>(values[i]) / 100.0;
  }
  for (int i = 0; i < 32; ++i) {
    result[32 + i] = static_cast<double>(values[31 - i]) / 100.0;
  }
  return result;
}

// clang-format off
constexpr std::array<int, 32> MG_PAWN = {
    0, 0, 0, 0, 0, 0, 0, 0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5, 5, 10, 25, 25, 10, 5, 5,
};
constexpr std::array<int, 32> MG_KNIGHT = {
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
};
constexpr std::array<int, 32> MG_BISHOP = {
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
};
constexpr std::array<int, 32> MG_ROOK = {
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 10, 10, 10, 10, 10, 10, 5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
};
constexpr std::array<int, 32> MG_QUEEN = {
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -5, 0, 5, 5, 5, 5, 0, -5,
};
constexpr std::array<int, 32> MG_KING = {
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
};

constexpr std::array<int, 32> EG_PAWN = {
    0, 0, 0, 0, 0, 0, 0, 0,
    80, 80, 80, 80, 80, 80, 80, 80,
    50, 50, 50, 50, 50, 50, 50, 50,
    30, 30, 30, 30, 30, 30, 30, 30,
};
constexpr std::array<int, 32> EG_KNIGHT = MG_KNIGHT;
constexpr std::array<int, 32> EG_BISHOP = MG_BISHOP;
constexpr std::array<int, 32> EG_ROOK = MG_ROOK;
constexpr std::array<int, 32> EG_QUEEN = MG_QUEEN;
constexpr std::array<int, 32> EG_KING = {
    -50, -40, -30, -20, -20, -30, -40, -50,
    -30, -20, -10, 0, 0, -10, -20, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
};
// clang-format on

// ── File masks for pawn-structure lookups ──────────────────────────
constexpr std::array<Bitboard, 8> FILE_MASKS = []() {
  std::array<Bitboard, 8> m{};
  for (int f = 0; f < 8; ++f) {
    Bitboard bb = 0;
    for (int r = 0; r < 8; ++r) {
      bb |= 1ULL << (r * 8 + f);
    }
    m[f] = bb;
  }
  return m;
}();

constexpr std::array<Bitboard, 8> ADJACENT_FILE_MASKS = []() {
  std::array<Bitboard, 8> m{};
  for (int f = 0; f < 8; ++f) {
    Bitboard bb = 0;
    if (f > 0) {
      for (int r = 0; r < 8; ++r)
        bb |= 1ULL << (r * 8 + (f - 1));
    }
    if (f < 7) {
      for (int r = 0; r < 8; ++r)
        bb |= 1ULL << (r * 8 + (f + 1));
    }
    m[f] = bb;
  }
  return m;
}();

// Mask of all squares strictly ahead of `square` in the given color's
// direction on the file + adjacent files (used for the passed-pawn check).
constexpr std::array<std::array<Bitboard, 64>, 2> PASSED_PAWN_MASKS = []() {
  std::array<std::array<Bitboard, 64>, 2> m{};
  for (int sq = 0; sq < 64; ++sq) {
    const int f = sq % 8;
    const int r = sq / 8;
    Bitboard white_mask = 0;
    Bitboard black_mask = 0;
    for (int rr = 0; rr < 8; ++rr) {
      for (int df = -1; df <= 1; ++df) {
        const int nf = f + df;
        if (nf < 0 || nf > 7)
          continue;
        const Bitboard bit = 1ULL << (rr * 8 + nf);
        if (rr > r)
          white_mask |= bit;
        if (rr < r)
          black_mask |= bit;
      }
    }
    m[0][sq] = white_mask; // WHITE
    m[1][sq] = black_mask; // BLACK
  }
  return m;
}();

} // namespace

// The tables are aligned on a cache line so repeated lookups in the tight
// PST loop stay out of the way of the move sorter's history arrays.
alignas(64) const std::array<std::array<double, 64>, 6> PST_MG = {
    make_table(MG_PAWN), make_table(MG_KNIGHT), make_table(MG_BISHOP),
    make_table(MG_ROOK), make_table(MG_QUEEN),  make_table(MG_KING),
};

alignas(64) const std::array<std::array<double, 64>, 6> PST_EG = {
    make_table(EG_PAWN), make_table(EG_KNIGHT), make_table(EG_BISHOP),
    make_table(EG_ROOK), make_table(EG_QUEEN),  make_table(EG_KING),
};

// ── compute_game_phase ──────────────────────────────────────────────
double compute_game_phase(const Board &board) noexcept {
  const int total =
      popcount(board.get_piece_bb(PieceType::QUEEN)) * PHASE_WEIGHT_QUEEN +
      popcount(board.get_piece_bb(PieceType::ROOK)) * PHASE_WEIGHT_ROOK +
      popcount(board.get_piece_bb(PieceType::BISHOP)) * PHASE_WEIGHT_BISHOP +
      popcount(board.get_piece_bb(PieceType::KNIGHT)) * PHASE_WEIGHT_KNIGHT;
  const double phase =
      static_cast<double>(total) / static_cast<double>(MAX_PHASE_MATERIAL);
  return std::min(1.0, phase);
}

// ── MaterialComponent ──────────────────────────────────────────────
double MaterialComponent::score(const Board &board,
                                double /*phase*/) const noexcept {
  int total = 0;
  for (int pt = 0; pt < 5; ++pt) { // skip king (value 0)
    const int value = MATERIAL_CP[pt];
    const int w =
        popcount(board.get_piece_bb(static_cast<PieceType>(pt), Color::WHITE));
    const int b =
        popcount(board.get_piece_bb(static_cast<PieceType>(pt), Color::BLACK));
    total += (w - b) * value;
  }
  return static_cast<double>(total);
}

double MaterialComponent::go(const Board &board) const {
  return score(board, 0.0);
}

// ── PSTComponent ──────────────────────────────────────────────────
double PSTComponent::score(const Board &board, double phase) const noexcept {
  double total = 0.0;
  for (int pt = 0; pt < 6; ++pt) {
    const auto &mg_table = PST_MG[pt];
    const auto &eg_table = PST_EG[pt];
    Bitboard white =
        board.get_piece_bb(static_cast<PieceType>(pt), Color::WHITE);
    Bitboard black =
        board.get_piece_bb(static_cast<PieceType>(pt), Color::BLACK);

    if (gsc_) {
      while (white != 0) {
        const uint8_t sq = pop_lsb(white);
        total += lerp(mg_table[sq], eg_table[sq], phase);
      }
      while (black != 0) {
        const uint8_t sq = pop_lsb(black);
        total -= lerp(mg_table[sq], eg_table[sq], phase);
      }
    } else {
      while (white != 0) {
        const uint8_t sq = pop_lsb(white);
        total += mg_table[sq];
      }
      while (black != 0) {
        const uint8_t sq = pop_lsb(black);
        total -= mg_table[sq];
      }
    }
  }
  return total;
}

double PSTComponent::go(const Board &board) const {
  return score(board, compute_game_phase(board));
}

// ── PawnStructureComponent ────────────────────────────────────────
namespace {

constexpr double DOUBLED_PENALTY_CP = 20.0;
constexpr double ISOLATED_PENALTY_CP = 25.0;
constexpr double PASSED_BASE_CP = 10.0;
constexpr double PASSED_PER_RANK_CP = 10.0;
constexpr double PAWN_STRUCT_GSC_MG = 0.6;
constexpr double PAWN_STRUCT_GSC_EG = 1.4;

} // namespace

double PawnStructureComponent::score(const Board &board,
                                     double phase) const noexcept {
  double raw = 0.0;
  const Bitboard white_pawns =
      board.get_piece_bb(PieceType::PAWN, Color::WHITE);
  const Bitboard black_pawns =
      board.get_piece_bb(PieceType::PAWN, Color::BLACK);

  for (int colour_idx = 0; colour_idx < 2; ++colour_idx) {
    const double sign = (colour_idx == 0) ? 1.0 : -1.0;
    Bitboard pawns = (colour_idx == 0) ? white_pawns : black_pawns;
    const Bitboard enemy = (colour_idx == 0) ? black_pawns : white_pawns;
    const bool is_white = (colour_idx == 0);
    Bitboard loop = pawns;

    // Precompute which files contain friendly pawns for the isolated test.
    uint8_t file_occupancy = 0;
    {
      Bitboard tmp = pawns;
      while (tmp != 0) {
        const uint8_t sq = pop_lsb(tmp);
        file_occupancy |= static_cast<uint8_t>(1 << (sq & 7));
      }
    }

    while (loop != 0) {
      const uint8_t sq = pop_lsb(loop);
      const int file = sq & 7;
      const int rank = sq >> 3;

      // Doubled: any friendly pawn on the same file strictly ahead of us.
      const Bitboard same_file = pawns & FILE_MASKS[file];
      Bitboard ahead;
      if (is_white) {
        // Clear rank bits at-or-below this rank.
        const Bitboard below =
            (rank < 7) ? ((1ULL << ((rank + 1) * 8)) - 1ULL) : ~0ULL;
        ahead = same_file & ~below;
      } else {
        const Bitboard below =
            (rank > 0) ? ((1ULL << (rank * 8)) - 1ULL) : 0ULL;
        ahead = same_file & below;
      }
      if (ahead != 0) {
        raw -= DOUBLED_PENALTY_CP * sign;
      }

      // Isolated: no friendly pawn on adjacent files.
      const uint8_t left =
          (file > 0) ? static_cast<uint8_t>(1 << (file - 1)) : 0;
      const uint8_t right =
          (file < 7) ? static_cast<uint8_t>(1 << (file + 1)) : 0;
      if ((file_occupancy & (left | right)) == 0) {
        raw -= ISOLATED_PENALTY_CP * sign;
      }

      // Passed: no enemy pawn on the same or adjacent files ahead of us.
      const Bitboard blockers = enemy & PASSED_PAWN_MASKS[colour_idx][sq];
      if (blockers == 0) {
        const int advancement = is_white ? rank : (7 - rank);
        raw += (PASSED_BASE_CP + PASSED_PER_RANK_CP * advancement) * sign;
      }
    }
  }

  if (gsc_) {
    const double weight = lerp(PAWN_STRUCT_GSC_MG, PAWN_STRUCT_GSC_EG, phase);
    return raw * weight;
  }
  return raw;
}

double PawnStructureComponent::go(const Board &board) const {
  return score(board, compute_game_phase(board));
}

// ── MobilityComponent ─────────────────────────────────────────────
namespace {

constexpr std::array<double, 6> MOBILITY_WEIGHTS = {
    1.0, 5.0, 5.0, 3.0, 2.0, 0.0 // PAWN, N, B, R, Q, K
};
constexpr double MOBILITY_GSC_MG = 0.3;
constexpr double MOBILITY_GSC_EG = 1.3;

[[nodiscard]] double raw_mobility_side(const Board &board) {
  double score = 0.0;
  for (const auto &move : board.generate_legal_moves()) {
    auto piece = board.piece_at(move.from);
    if (!piece || piece->type == PieceType::KING) {
      continue;
    }
    score += MOBILITY_WEIGHTS[static_cast<int>(piece->type)];
  }
  return score;
}

} // namespace

double MobilityComponent::score(const Board &board, double phase) const {
  // Python walks: side-to-move perspective plus one null-move flip.  We
  // mirror it bit-for-bit to stay parity-safe — this is the slowest
  // component and a future pass can replace it with a bitboard-based
  // pseudo-mobility heuristic that keeps the same ordering.
  Board mut = board;
  const bool current_is_white = mut.get_side_to_move();
  double total = 0.0;
  const double current = raw_mobility_side(mut);
  total += current_is_white ? current : -current;

  mut.push_null();
  const double other = raw_mobility_side(mut);
  total += current_is_white ? -other : other;
  (void)mut.pop();

  if (gsc_) {
    const double weight = lerp(MOBILITY_GSC_MG, MOBILITY_GSC_EG, phase);
    return total * weight;
  }
  return total;
}

double MobilityComponent::go(const Board &board) const {
  return score(board, compute_game_phase(board));
}

// ── KingSafetyComponent ──────────────────────────────────────────
namespace {

constexpr double PAWN_SHIELD_BONUS_CP = 15.0;
constexpr double OPEN_FILE_PENALTY_CP = 30.0;
constexpr double ATTACK_ZONE_WEIGHT_CP = 8.0;
constexpr double KING_SAFETY_GSC_MG = 1.3;
constexpr double KING_SAFETY_GSC_EG = 0.4;

[[nodiscard]] double pawn_shield(const Board &board, Color color, int king_file,
                                 int king_rank) noexcept {
  double bonus = 0.0;
  const int dir = (color == Color::WHITE) ? 1 : -1;
  const Bitboard friendly_pawns = board.get_piece_bb(PieceType::PAWN, color);
  for (const int df : {-1, 0, 1}) {
    const int f = king_file + df;
    const int r = king_rank + dir;
    if (f < 0 || f > 7 || r < 0 || r > 7)
      continue;
    const Bitboard bit = 1ULL << (r * 8 + f);
    if ((friendly_pawns & bit) != 0) {
      bonus += PAWN_SHIELD_BONUS_CP;
    }
  }
  return bonus;
}

[[nodiscard]] double open_file_penalty(const Board &board,
                                       int king_file) noexcept {
  const Bitboard file_mask = FILE_MASKS[king_file];
  if ((board.get_piece_bb(PieceType::PAWN) & file_mask) != 0) {
    return 0.0;
  }
  return -OPEN_FILE_PENALTY_CP;
}

[[nodiscard]] double attack_zone_pressure(const Board &board, Color color,
                                          int king_file,
                                          int king_rank) noexcept {
  const Color enemy = (color == Color::WHITE) ? Color::BLACK : Color::WHITE;
  const Bitboard enemy_pieces = board.get_color_bb(enemy);
  // Build the 3×3 king zone as a bitboard and intersect with enemy pieces.
  Bitboard zone = 0;
  for (int df = -1; df <= 1; ++df) {
    for (int dr = -1; dr <= 1; ++dr) {
      const int f = king_file + df;
      const int r = king_rank + dr;
      if (f < 0 || f > 7 || r < 0 || r > 7)
        continue;
      zone |= 1ULL << (r * 8 + f);
    }
  }
  return static_cast<double>(popcount(zone & enemy_pieces)) *
         ATTACK_ZONE_WEIGHT_CP;
}

} // namespace

double KingSafetyComponent::score(const Board &board,
                                  double phase) const noexcept {
  double raw = 0.0;
  for (const Color color : {Color::WHITE, Color::BLACK}) {
    const double sign = (color == Color::WHITE) ? 1.0 : -1.0;
    auto ks = board.king(color);
    if (!ks.has_value()) {
      continue;
    }
    const int kf = static_cast<int>(square_file(*ks));
    const int kr = static_cast<int>(square_rank(*ks));

    raw += pawn_shield(board, color, kf, kr) * sign;
    raw += open_file_penalty(board, kf) * sign;
    raw -= attack_zone_pressure(board, color, kf, kr) * sign;
  }
  if (gsc_) {
    const double weight = lerp(KING_SAFETY_GSC_MG, KING_SAFETY_GSC_EG, phase);
    return raw * weight;
  }
  return raw;
}

double KingSafetyComponent::go(const Board &board) const {
  return score(board, compute_game_phase(board));
}

// ── CompositeEvaluator ────────────────────────────────────────────
double CompositeEvaluator::go(const Board &board) const {
  const double phase = compute_game_phase(board);
  double total = 0.0;
  for (const auto &c : components_) {
    total += c->score(board, phase);
  }
  return total;
}

} // namespace evaluators
