#pragma once
// ---------------------------------------------------------------------------
// minimax.hpp — C++ port of engine.search.minimax.Minimax
//
// A near 1:1 port of the Python IDDFS negamax search.  The layering matches
// the Python reference so that each feature can be toggled independently via
// CppSearchConfig.
// ---------------------------------------------------------------------------

#include <chrono>
#include <cstdint>
#include <optional>

#include "../board/board.hpp"
#include "../evaluators/evaluators.hpp"
#include "move_sorter.hpp"
#include "search_config.hpp"
#include "transposition_table.hpp"
#include "zobrist.hpp"

#if defined(__GNUC__) || defined(__clang__)
#define MM_LIKELY(x) __builtin_expect(!!(x), 1)
#define MM_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define MM_LIKELY(x) (x)
#define MM_UNLIKELY(x) (x)
#endif

namespace search {

// Statistics collected during a single search.  Mirror of the Python
// SearchStats dataclass so the same fields are visible to callers.
struct MinimaxStats {
  uint64_t nodes = 0;
  int depth = 0;
  int seldepth = 0;
  uint64_t tt_hits = 0;
  int hashfull = 0;
  uint64_t beta_cutoffs = 0;
  uint64_t first_move_cuts = 0;
  uint64_t killer_cuts = 0;
  uint64_t history_cuts = 0;
  uint64_t qsearch_nodes = 0;
  uint64_t null_move_cuts = 0;
  uint64_t pvs_researches = 0;
  uint64_t lmr_researches = 0;
  uint64_t qs_see_pruning = 0;
  uint64_t qs_delta_pruning = 0;
  uint64_t check_extensions = 0;
  uint64_t iid_searches = 0;
  uint64_t root_move_changes = 0;
  double history_saturation = 0.0;
  int score = 0;

  void reset() noexcept { *this = MinimaxStats{}; }
};

class Minimax {
public:
  static constexpr int MATE_SCORE = 100'000;
  static constexpr int TIME_CHECK_INTERVAL = 2048;

  Minimax(Board &board, evaluators::IEvaluator &evaluator,
          TranspositionTable *tt, MoveSorter *sorter, Zobrist *zobrist,
          const CppSearchConfig &config) noexcept;

  // Reset per-search state (stats, node counter, time flag).  Optionally
  // clear the TT and move-sorter history/killer tables.
  void reset_state(bool clear_tt, bool clear_history,
                   bool clear_killers) noexcept;

  // Run IDDFS up to *depth*.  Returns (score, best_move) from White's
  // perspective.  Either component of the pair can be empty if the search
  // was unable to find anything before the time limit.
  struct Result {
    std::optional<double> score;
    std::optional<Move> best_move;
  };

  Result find_best_move(int depth);

  [[nodiscard]] const MinimaxStats &stats() const noexcept { return stats_; }
  [[nodiscard]] uint64_t node_count() const noexcept { return stats_.nodes; }

  // Time-limit helpers — exposed so the Python wrapper can forward the
  // same API used by tests.
  [[nodiscard]] bool time_up() const noexcept { return time_up_; }
  void set_time_up(bool value) noexcept { time_up_ = value; }
  [[nodiscard]] bool check_time_limit() noexcept;
  void reset_clock() noexcept;

  [[nodiscard]] std::optional<Move> root_best_move() const noexcept {
    return root_best_move_;
  }

private:
  using Clock = std::chrono::steady_clock;

  Board &board_;
  evaluators::IEvaluator &evaluator_;
  TranspositionTable *tt_;
  MoveSorter *move_sorter_;
  Zobrist *zobrist_;
  const CppSearchConfig &config_;
  MinimaxStats stats_;

  std::optional<Clock::time_point> start_time_;
  bool time_up_ = false;
  std::optional<Move> root_best_move_;

  // Core search functions.
  [[nodiscard]] double search_with_window(int depth, double alpha, double beta);
  [[nodiscard]] double negamax(int depth, double alpha, double beta, int ply,
                               std::optional<Move> previous_move,
                               int extensions_left);
  [[nodiscard]] double search_child(int index, int next_depth, double alpha,
                                    double beta, int ply, const Move &move,
                                    bool in_check, bool gives_check,
                                    bool is_tactical, int extensions_left);
  [[nodiscard]] double quiescence(double alpha, double beta, int ply,
                                  int qs_depth);
  [[nodiscard]] double null_move_search(int depth, double beta, int ply,
                                        int extensions_left);

  // Helpers.
  [[nodiscard]] inline double relative_eval() {
    const double white_perspective = evaluator_.go(board_);
    return board_.get_side_to_move() ? white_perspective : -white_perspective;
  }
  [[nodiscard]] inline double terminal_score(GameState state,
                                             int ply) const noexcept {
    if (state == GameState::CHECKMATE) {
      return -MATE_SCORE + ply;
    }
    return 0.0;
  }
  [[nodiscard]] inline std::optional<uint64_t> current_hash() const noexcept {
    if (zobrist_ == nullptr) {
      return std::nullopt;
    }
    return zobrist_->get_current_hash();
  }
  [[nodiscard]] inline std::optional<uint64_t>
  push_move_with_hash(const Move &move) {
    auto saved_hash = current_hash();
    std::optional<uint64_t> next_hash;
    if (zobrist_ != nullptr && saved_hash.has_value()) {
      next_hash = zobrist_->make_move_hash(board_, move);
    }
    board_.push(move);
    if (zobrist_ != nullptr) {
      if (next_hash.has_value()) {
        zobrist_->set_current_hash(next_hash);
      } else {
        (void)zobrist_->hash_board(board_);
      }
    }
    return saved_hash;
  }
  inline void pop_move_with_hash(std::optional<uint64_t> saved_hash) noexcept {
    (void)board_.pop();
    if (zobrist_ != nullptr && saved_hash.has_value()) {
      zobrist_->set_current_hash(saved_hash);
    }
  }
  [[nodiscard]] inline int capture_gain(const Move &move) const noexcept {
    auto piece = board_.piece_at(move.to);
    if (!piece && board_.is_en_passant(move)) {
      return MoveSorter::PIECE_VALUES_CP[0];
    }
    if (!piece) {
      return 0;
    }
    return MoveSorter::PIECE_VALUES_CP[static_cast<int>(piece->type)];
  }
  [[nodiscard]] bool has_non_pawn_material() const noexcept;
  [[nodiscard]] inline bool is_tactical_move(const Move &move) const noexcept {
    return board_.is_capture(move) || move.promotion != 0;
  }
  [[nodiscard]] inline bool
  can_apply_futility(int depth, double static_eval, double alpha, bool in_check,
                     bool is_tactical) const noexcept {
    if (!config_.use_alpha_beta || in_check || is_tactical) {
      return false;
    }
    if (config_.use_futility_pruning && depth == 1 &&
        static_eval + static_cast<double>(config_.futility_margin_standard) <=
            alpha) {
      return true;
    }
    return config_.use_extended_futility_pruning && depth == 2 &&
           static_eval +
                   static_cast<double>(config_.futility_margin_extended) <=
               alpha;
  }
  [[nodiscard]] inline bool can_apply_lmr(int move_index, int depth,
                                          bool in_check, bool gives_check,
                                          bool is_tactical) const noexcept {
    if (!config_.use_lmr || in_check || gives_check || is_tactical) {
      return false;
    }
    if (depth < config_.lmr_min_depth) {
      return false;
    }
    return move_index >= config_.lmr_min_move_number;
  }
  [[nodiscard]] static int lmr_reduction(int depth, int move_index) noexcept;
  [[nodiscard]] inline TTBound determine_bound(double best_score,
                                               double original_alpha,
                                               double beta) const noexcept {
    if (!config_.use_alpha_beta) {
      return TTBound::EXACT;
    }
    if (best_score <= original_alpha) {
      return TTBound::UPPER;
    }
    if (best_score >= beta) {
      return TTBound::LOWER;
    }
    return TTBound::EXACT;
  }
};

} // namespace search
