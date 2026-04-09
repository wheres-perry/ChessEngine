#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <tuple>
#include <vector>

#include "../board/board.hpp"

namespace move_ordering {

static constexpr int MAX_PLY = 128;

static constexpr int HASH_MOVE_SCORE = 100'000'000;
static constexpr int TACTICAL_BASE = 10'000'000;
static constexpr int KILLER_BASE = 1'000'000;
static constexpr int COUNTERMOVE_SCORE = 850'000;

// Number of top moves to partial-sort in sort_moves().  In alpha-beta,
// we typically cut off after the first few moves.
static constexpr int PARTIAL_SORT_N = 8;

// Piece values in centipawns, indexed by PieceType enum (0-5).
static constexpr std::array<int, 6> PIECE_VALUES_CP = {
    100,   // PAWN
    320,   // KNIGHT
    330,   // BISHOP
    500,   // ROOK
    900,   // QUEEN
    20'000 // KING
};

struct MoveSorterConfig {
  bool use_move_ordering = true;
  bool use_mvv_lva = true;
  bool use_history_heuristic = true;
  bool use_countermove_heuristic = true;
  bool use_see_ordering = true;
  bool use_killer_moves = true;
  bool use_hash_move_ordering = true;
  int history_max_score = 16384;
  int killer_slots_per_ply = 2;
  int see_capture_threshold = 0;
};

// A hashable move key: (from, to, promotion).
using MoveKey = std::tuple<uint8_t, uint8_t, uint8_t>;

inline constexpr MoveKey make_move_key(const Move &m) noexcept {
  return {m.from, m.to, m.promotion};
}

// Sentinel move meaning "no move".
static constexpr Move NO_MOVE{0, 0, 0};

inline constexpr bool is_no_move(const Move &m) noexcept {
  return m.from == 0 && m.to == 0 && m.promotion == 0;
}

class MoveSorter {
public:
  explicit MoveSorter(const MoveSorterConfig &config) noexcept
      : config_(config) {
    reset(true, true);
  }

  /// Sort all moves in descending priority order.
  /// Uses partial sort for the top N moves (better for alpha-beta cutoff).
  [[nodiscard]] std::vector<Move> sort_moves(Board &board,
                                             const std::vector<Move> &moves,
                                             int ply, const Move &hash_move,
                                             const Move &previous_move) const;

  /// Sort captures/promotions for quiescence search.
  [[nodiscard]] std::vector<Move>
  sort_tactical(Board &board, const std::vector<Move> &moves) const;

  /// Static Exchange Evaluation.
  [[nodiscard]] int see(Board &board, const Move &move) const;

  /// Update killer, history, and countermove tables after a beta cutoff.
  void on_beta_cutoff(const Move &move, int ply, int depth,
                      const Move &previous_move, bool is_tactical) noexcept;

  /// Return history table saturation percentage (0-100).
  [[nodiscard]] double history_saturation() const noexcept;

  /// Selective reset of heuristic tables.
  void reset(bool clear_history, bool clear_killers) noexcept;

  // --- Accessors for Python wrapper ---

  /// Get killer moves for a given ply. Returns empty vector if ply is invalid.
  [[nodiscard]] const std::vector<Move> &get_killers(int ply) const noexcept {
    static const std::vector<Move> empty;
    if (ply < 0 || ply >= MAX_PLY) [[unlikely]] {
      return empty;
    }
    return killer_moves_[ply];
  }

  /// Get history score for a move key (flat array lookup).
  [[nodiscard]] int get_history(uint8_t from, uint8_t to,
                                uint8_t /*promo*/) const noexcept {
    if (from >= 64 || to >= 64) [[unlikely]] {
      return 0;
    }
    // History is indexed by [side][from][to]. Since we don't know side here,
    // return max of both sides for the Python accessor.
    return std::max(history_table_[0][from][to], history_table_[1][from][to]);
  }

  /// Get full history table as a vector of ((from,to,promo), score) pairs.
  /// Builds a map on-the-fly from the flat arrays for Python compatibility.
  [[nodiscard]] std::vector<std::pair<MoveKey, int>>
  get_history_entries() const noexcept {
    std::vector<std::pair<MoveKey, int>> entries;
    for (int side = 0; side < 2; ++side) {
      for (int from = 0; from < 64; ++from) {
        for (int to = 0; to < 64; ++to) {
          int val = history_table_[side][from][to];
          if (val != 0) {
            entries.emplace_back(MoveKey{static_cast<uint8_t>(from),
                                         static_cast<uint8_t>(to), 0},
                                 val);
          }
        }
      }
    }
    return entries;
  }

  /// Get the config (read-only).
  [[nodiscard]] const MoveSorterConfig &config() const noexcept {
    return config_;
  }

private:
  [[nodiscard]] int score_move(Board &board, const Move &move, int ply,
                               const Move &hash_move,
                               const Move &previous_move) const;

  [[nodiscard]] int score_tactical_move(Board &board, const Move &move) const;

  [[nodiscard]] int mvv_lva(Board &board, const Move &move) const noexcept;

  [[nodiscard]] static constexpr bool is_promotion(const Move &move) noexcept {
    return move.promotion != 0;
  }

  MoveSorterConfig config_;
  std::array<std::vector<Move>, MAX_PLY> killer_moves_;

  // Flat 2D history table: history_table_[side][from][to].
  // side: 0=WHITE, 1=BLACK.  Eliminates all map overhead.
  int history_table_[2][64][64]{};

  // Flat countermove table: countermove_[from][to] = the counter-move.
  Move countermove_table_[64][64]{};

  // Track how many non-zero entries exist for saturation calculation.
  int history_entry_count_ = 0;
  int64_t history_sum_ = 0;
};

} // namespace move_ordering
