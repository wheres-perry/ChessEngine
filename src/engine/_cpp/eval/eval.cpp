#include "eval.hpp"
#include "pst_tables.hpp"

#include <algorithm>
#include <cmath>

namespace eval {

// ── Material piece values (centipawns) ───────────────────────────────
static constexpr int MATERIAL_CP[6] = {
    100, // PAWN
    320, // KNIGHT
    330, // BISHOP
    500, // ROOK
    900, // QUEEN
    0,   // KING
};

// ── Game-phase weights for non-pawn, non-king material ──────────────
static constexpr int PHASE_WEIGHT[6] = {
    0, // PAWN
    3, // KNIGHT
    3, // BISHOP
    5, // ROOK
    9, // QUEEN
    0, // KING
};
static constexpr int MAX_PHASE_MATERIAL = 62;

// ── Pawn structure constants ────────────────────────────────────────
static constexpr int DOUBLED_PENALTY = 20;
static constexpr int ISOLATED_PENALTY = 25;
static constexpr int PASSED_BASE = 10;
static constexpr int PASSED_PER_RANK = 10;
static constexpr double PAWN_STRUCT_GSC_MG = 0.6;
static constexpr double PAWN_STRUCT_GSC_EG = 1.4;

// ── Mobility weights per piece type ─────────────────────────────────
static constexpr double MOBILITY_WEIGHT[6] = {
    1.0, // PAWN
    5.0, // KNIGHT
    5.0, // BISHOP
    3.0, // ROOK
    2.0, // QUEEN
    0.0, // KING  (ignored)
};
static constexpr double MOBILITY_GSC_MG = 0.3;
static constexpr double MOBILITY_GSC_EG = 1.3;

// ── King safety constants ───────────────────────────────────────────
static constexpr double PAWN_SHIELD_BONUS = 15.0;
static constexpr double OPEN_FILE_PENALTY = 30.0;
static constexpr double ATTACK_ZONE_WEIGHT = 8.0;
static constexpr double KING_SAFETY_GSC_MG = 1.3;
static constexpr double KING_SAFETY_GSC_EG = 0.4;

// ── Helpers ─────────────────────────────────────────────────────────
static inline constexpr double lerp(double mg, double eg,
                                    double phase) noexcept {
  return mg * phase + eg * (1.0 - phase);
}

// File masks: 8-bit columns pre-computed.
static constexpr Bitboard FILE_MASK[8] = {
    0x0101010101010101ULL << 0, 0x0101010101010101ULL << 1,
    0x0101010101010101ULL << 2, 0x0101010101010101ULL << 3,
    0x0101010101010101ULL << 4, 0x0101010101010101ULL << 5,
    0x0101010101010101ULL << 6, 0x0101010101010101ULL << 7,
};

// Adjacent file masks for isolated pawn detection (bitboard of adjacent files).
static constexpr Bitboard ADJACENT_FILES[8] = {
    FILE_MASK[1],                // file a: only file b
    FILE_MASK[0] | FILE_MASK[2], // file b: files a,c
    FILE_MASK[1] | FILE_MASK[3], // file c: files b,d
    FILE_MASK[2] | FILE_MASK[4], // file d: files c,e
    FILE_MASK[3] | FILE_MASK[5], // file e: files d,f
    FILE_MASK[4] | FILE_MASK[6], // file f: files e,g
    FILE_MASK[5] | FILE_MASK[7], // file g: files f,h
    FILE_MASK[6],                // file h: only file g
};

// Passed pawn masks: for each square/color, the set of squares where an enemy
// pawn would block a passed pawn.  Precomputed for fast lookup.
static constexpr Bitboard compute_white_passed_mask(int file,
                                                    int rank) noexcept {
  Bitboard mask = 0;
  for (int f = file - 1; f <= file + 1; ++f) {
    if (f < 0 || f > 7)
      continue;
    for (int r = rank + 1; r <= 7; ++r) {
      mask |= 1ULL << (r * 8 + f);
    }
  }
  return mask;
}

static constexpr Bitboard compute_black_passed_mask(int file,
                                                    int rank) noexcept {
  Bitboard mask = 0;
  for (int f = file - 1; f <= file + 1; ++f) {
    if (f < 0 || f > 7)
      continue;
    for (int r = rank - 1; r >= 0; --r) {
      mask |= 1ULL << (r * 8 + f);
    }
  }
  return mask;
}

// Precomputed passed pawn masks [color][square].
static constexpr auto make_passed_masks() noexcept {
  struct PassedMasks {
    Bitboard white[64]{};
    Bitboard black[64]{};
  };
  PassedMasks m{};
  for (int sq = 0; sq < 64; ++sq) {
    int file = sq % 8;
    int rank = sq / 8;
    m.white[sq] = compute_white_passed_mask(file, rank);
    m.black[sq] = compute_black_passed_mask(file, rank);
  }
  return m;
}

static constexpr auto PASSED_MASKS = make_passed_masks();

// King zone: 3x3 area around the king (precomputed).
static constexpr Bitboard compute_king_zone(int sq) noexcept {
  Bitboard zone = 0;
  int kf = sq % 8;
  int kr = sq / 8;
  for (int df = -1; df <= 1; ++df) {
    for (int dr = -1; dr <= 1; ++dr) {
      int f = kf + df;
      int r = kr + dr;
      if (f >= 0 && f <= 7 && r >= 0 && r <= 7) {
        zone |= 1ULL << (r * 8 + f);
      }
    }
  }
  return zone;
}

static constexpr auto make_king_zones() noexcept {
  struct KingZones {
    Bitboard zones[64]{};
  };
  KingZones kz{};
  for (int sq = 0; sq < 64; ++sq) {
    kz.zones[sq] = compute_king_zone(sq);
  }
  return kz;
}

static constexpr auto KING_ZONES = make_king_zones();

// Pawn shield masks: squares directly in front of king (one rank forward).
static constexpr Bitboard compute_pawn_shield(int sq, bool is_white) noexcept {
  Bitboard shield = 0;
  int kf = sq % 8;
  int kr = sq / 8;
  int dir = is_white ? 1 : -1;
  int r = kr + dir;
  if (r < 0 || r > 7)
    return 0;
  for (int df = -1; df <= 1; ++df) {
    int f = kf + df;
    if (f >= 0 && f <= 7) {
      shield |= 1ULL << (r * 8 + f);
    }
  }
  return shield;
}

static constexpr auto make_pawn_shields() noexcept {
  struct PawnShields {
    Bitboard white[64]{};
    Bitboard black[64]{};
  };
  PawnShields ps{};
  for (int sq = 0; sq < 64; ++sq) {
    ps.white[sq] = compute_pawn_shield(sq, true);
    ps.black[sq] = compute_pawn_shield(sq, false);
  }
  return ps;
}

static constexpr auto PAWN_SHIELDS = make_pawn_shields();

// ── Constructor ─────────────────────────────────────────────────────
Evaluator::Evaluator(const EvalConfig &config) noexcept : config_(config) {}

// ── Main entry point ────────────────────────────────────────────────
int Evaluator::go(Board &board) const noexcept {
  double phase = compute_game_phase(board);

  double score = static_cast<double>(material_score(board));

  if (config_.use_pst) [[likely]] {
    score += pst_score(board, phase);
  }

  if (config_.use_pawn_structure) [[likely]] {
    score += pawn_structure_score(board, phase);
  }

  if (config_.use_mobility) [[likely]] {
    score += mobility_score(board, phase);
  }

  if (config_.use_king_safety) [[likely]] {
    score += king_safety_score(board, phase);
  }

  return static_cast<int>(score);
}

// ── Game phase (1.0 = opening, 0.0 = endgame) ──────────────────────
double Evaluator::compute_game_phase(const Board &board) const noexcept {
  int total = 0;
  for (int pt = 0; pt < 6; ++pt) {
    if (PHASE_WEIGHT[pt] == 0)
      continue;
    auto piece_type = static_cast<PieceType>(pt);
    int count = popcount(board.get_piece_bb(piece_type, Color::WHITE)) +
                popcount(board.get_piece_bb(piece_type, Color::BLACK));
    total += count * PHASE_WEIGHT[pt];
  }
  return std::min(1.0, static_cast<double>(total) / MAX_PHASE_MATERIAL);
}

// ── Material (using popcount intrinsics on bitboards) ───────────────
int Evaluator::material_score(const Board &board) const noexcept {
  int score = 0;
  for (int pt = 0; pt < 6; ++pt) {
    auto piece_type = static_cast<PieceType>(pt);
    int w = popcount(board.get_piece_bb(piece_type, Color::WHITE));
    int b = popcount(board.get_piece_bb(piece_type, Color::BLACK));
    score += (w - b) * MATERIAL_CP[pt];
  }
  return score;
}

// ── PST ─────────────────────────────────────────────────────────────
double Evaluator::pst_score(const Board &board, double phase) const noexcept {
  double total = 0.0;
  bool gsc = config_.game_stage_conscious;

  for (uint8_t sq = 0; sq < 64; ++sq) {
    auto piece_opt = board.piece_at(sq);
    if (!piece_opt.has_value() || !piece_opt->valid)
      continue;

    int pt = static_cast<int>(piece_opt->type);

    // PST values are in centipawns; Python divides by 100 before use.
    // We keep centipawns here and divide by 100 to match Python's pawn units.
    double mg = static_cast<double>((*MG_TABLES[pt])[sq]) / 100.0;

    double value;
    if (gsc) {
      double eg = static_cast<double>((*EG_TABLES[pt])[sq]) / 100.0;
      value = lerp(mg, eg, phase);
    } else {
      value = mg;
    }

    if (piece_opt->color == Color::WHITE)
      total += value;
    else
      total -= value;
  }

  return total;
}

// ── Pawn Structure (bitboard-based) ─────────────────────────────────

double Evaluator::pawn_structure_score(const Board &board,
                                       double phase) const noexcept {
  double raw = 0.0;

  for (int c = 0; c < 2; ++c) {
    auto color = static_cast<Color>(c);
    double sign = (color == Color::WHITE) ? 1.0 : -1.0;

    Bitboard pawns_bb = board.get_piece_bb(PieceType::PAWN, color);
    Color enemy = (color == Color::WHITE) ? Color::BLACK : Color::WHITE;
    Bitboard enemy_pawns = board.get_piece_bb(PieceType::PAWN, enemy);
    Bitboard iter = pawns_bb;

    while (iter) {
      uint8_t sq = pop_lsb(iter);
      int file = square_file(sq);
      int rank = square_rank(sq);

      // Doubled pawns: more than one pawn on the same file.
      Bitboard same_file = pawns_bb & FILE_MASK[file];
      // Check if there are pawns ahead on this file.
      Bitboard ahead = same_file;
      Bitboard tmp = ahead;
      while (tmp) {
        uint8_t p = pop_lsb(tmp);
        int pr = square_rank(p);
        if (color == Color::WHITE) {
          if (pr > rank) {
            raw -= DOUBLED_PENALTY * sign;
            break;
          }
        } else {
          if (pr < rank) {
            raw -= DOUBLED_PENALTY * sign;
            break;
          }
        }
      }

      // Isolated pawns: no friendly pawns on adjacent files (bitboard check).
      if (!(pawns_bb & ADJACENT_FILES[file])) {
        raw -= ISOLATED_PENALTY * sign;
      }

      // Passed pawns: no enemy pawns ahead on same or adjacent files.
      Bitboard pass_mask = (color == Color::WHITE) ? PASSED_MASKS.white[sq]
                                                   : PASSED_MASKS.black[sq];
      if (!(enemy_pawns & pass_mask)) {
        int advancement = (color == Color::WHITE) ? rank : (7 - rank);
        raw += (PASSED_BASE + PASSED_PER_RANK * advancement) * sign;
      }
    }
  }

  if (config_.game_stage_conscious) {
    double weight = lerp(PAWN_STRUCT_GSC_MG, PAWN_STRUCT_GSC_EG, phase);
    return raw * weight;
  }
  return raw;
}

// ── Mobility ────────────────────────────────────────────────────────
double Evaluator::mobility_score(Board &board, double phase) const noexcept {
  double total = 0.0;

  // Evaluate legal moves for the side to move.
  auto evaluate_moves = [&](bool is_white) -> double {
    double score = 0.0;
    auto moves = board.generate_legal_moves();
    for (const auto &m : moves) {
      auto piece_opt = board.piece_at(m.from);
      if (!piece_opt.has_value() || !piece_opt->valid)
        continue;
      int pt = static_cast<int>(piece_opt->type);
      if (pt == static_cast<int>(PieceType::KING))
        continue;
      score += MOBILITY_WEIGHT[pt];
    }
    return is_white ? score : -score;
  };

  bool current_is_white = board.get_side_to_move();
  total += evaluate_moves(current_is_white);

  // Other side.
  board.push_null();
  total += evaluate_moves(!current_is_white);
  board.pop();

  if (config_.game_stage_conscious) {
    double weight = lerp(MOBILITY_GSC_MG, MOBILITY_GSC_EG, phase);
    return total * weight;
  }
  return total;
}

// ── King Safety (bitboard-based with precomputed masks) ─────────────

double Evaluator::king_safety_score(const Board &board,
                                    double phase) const noexcept {
  double raw = 0.0;

  for (int c = 0; c < 2; ++c) {
    auto color = static_cast<Color>(c);
    double sign = (color == Color::WHITE) ? 1.0 : -1.0;

    auto king_opt = board.king(color);
    if (!king_opt.has_value()) [[unlikely]] {
      continue;
    }

    uint8_t ks = *king_opt;
    int kf = square_file(ks);

    // Pawn shield: use precomputed shield mask and bitboard intersection.
    Bitboard my_pawns = board.get_piece_bb(PieceType::PAWN, color);
    Bitboard shield_mask = (color == Color::WHITE) ? PAWN_SHIELDS.white[ks]
                                                   : PAWN_SHIELDS.black[ks];
    int shield_count = popcount(my_pawns & shield_mask);
    raw += shield_count * PAWN_SHIELD_BONUS * sign;

    // Open file penalty.
    Bitboard all_pawns = board.get_piece_bb(PieceType::PAWN);
    if (!(all_pawns & FILE_MASK[kf])) {
      raw -= OPEN_FILE_PENALTY * sign;
    }

    // Attack zone pressure: use precomputed king zone mask,
    // intersect with enemy pieces.
    Color enemy = (color == Color::WHITE) ? Color::BLACK : Color::WHITE;
    Bitboard enemy_pieces = board.get_color_bb(enemy);
    Bitboard king_zone = KING_ZONES.zones[ks];
    int attackers_in_zone = popcount(enemy_pieces & king_zone);
    raw -= attackers_in_zone * ATTACK_ZONE_WEIGHT * sign;
  }

  if (config_.game_stage_conscious) {
    double weight = lerp(KING_SAFETY_GSC_MG, KING_SAFETY_GSC_EG, phase);
    return raw * weight;
  }
  return raw;
}

} // namespace eval
