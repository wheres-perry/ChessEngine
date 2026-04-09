#pragma once

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "../board/board.hpp"
#include "../eval/eval.hpp"
#include "../syzygy/syzygy.hpp"
#include "move_ordering.hpp"
#include "transposition_table.hpp"
#include "zobrist.hpp"

namespace search {

// ── Precomputed LMR reduction table ────────────────────────────────────
// Computed at compile time.  LMR_TABLE[depth][move_index] gives the
// reduction to apply.  Uses the standard formula:
//   0.75 * ln(max(2,depth)) * ln(max(2, move_index+1))
// clamped to [1, 3].

static constexpr int LMR_MAX_DEPTH = 128;
static constexpr int LMR_MAX_MOVES = 256;

// constexpr-compatible natural log approximation (Taylor series around 1).
// For LMR we only need rough values for small integers; this is accurate
// enough and avoids std::log() which isn't constexpr in C++17.
static constexpr double cx_log(double x) noexcept {
  // Use the identity: ln(x) = 2 * atanh((x-1)/(x+1))
  // where atanh(z) = z + z^3/3 + z^5/5 + ...
  if (x <= 0.0)
    return -1000.0;
  // Reduce to [1,2) range
  double result = 0.0;
  double v = x;
  while (v >= 2.0) {
    v /= 2.718281828459045; // divide by e
    result += 1.0;
  }
  while (v < 1.0) {
    v *= 2.718281828459045;
    result -= 1.0;
  }
  // Now v is in [1, e), use Taylor series for ln(v)
  double z = (v - 1.0) / (v + 1.0);
  double z2 = z * z;
  double term = z;
  double sum = z;
  for (int i = 1; i < 20; ++i) {
    term *= z2;
    sum += term / (2.0 * i + 1.0);
  }
  return result + 2.0 * sum;
}

static constexpr auto make_lmr_table() noexcept {
  struct LMRTable {
    int data[LMR_MAX_DEPTH][LMR_MAX_MOVES]{};
  };
  LMRTable t{};
  for (int d = 0; d < LMR_MAX_DEPTH; ++d) {
    for (int m = 0; m < LMR_MAX_MOVES; ++m) {
      double base = 0.75 * cx_log(d < 2 ? 2.0 : static_cast<double>(d)) *
                    cx_log(m < 1 ? 2.0 : static_cast<double>(m + 1));
      int val = static_cast<int>(base);
      if (val < 1)
        val = 1;
      if (val > 3)
        val = 3;
      t.data[d][m] = val;
    }
  }
  return t;
}

static constexpr auto LMR_TABLE = make_lmr_table();

// ── SearchConfig ────────────────────────────────────────────────────────

struct SearchConfig {
  // Move ordering
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

  // Search algorithms
  bool use_alpha_beta = true;
  bool use_pvs = true;
  bool use_quiescence_search = true;
  int qs_max_depth = 16;
  bool use_iid = true;
  int iid_min_depth = 5;
  int iid_depth_reduction = 2;

  // Pruning
  bool use_null_move_pruning = true;
  int nmp_reduction_r = 3;
  int nmp_min_depth = 3;
  bool use_lmr = true;
  int lmr_min_depth = 3;
  int lmr_min_move_number = 4;
  bool use_futility_pruning = true;
  int futility_margin_standard = 300;
  bool use_extended_futility_pruning = true;
  int futility_margin_extended = 500;
  bool use_reverse_futility_pruning = true;
  int rfp_margin_multiplier = 120;
  int rfp_max_depth = 8;
  bool use_delta_pruning = true;
  int delta_margin = 200;
  bool use_see_pruning_in_qs = true;

  // State & hashing
  bool use_aspiration_windows = true;
  int aspiration_window_margin = 50;
  bool use_check_extensions = true;
  int max_check_extensions = 16;
  bool use_transposition_table = true;
  int tt_size_mb = 64;
  bool use_tt_aging = true;

  // Syzygy
  bool use_syzygy = false;
  bool use_50_move_rule = true;

  // Lazy SMP
  bool use_lazy_smp = false;
  int smp_num_threads = 1;

  // Time
  double max_time = 250.0;
  bool has_max_time = true; // false when max_time is None in Python
};

// ── SearchStats ─────────────────────────────────────────────────────────

struct SearchStats {
  std::atomic<uint64_t> nodes{0};
  int depth = 0;
  int seldepth = 0;
  std::atomic<uint64_t> tt_hits{0};
  int hashfull = 0;
  std::atomic<uint64_t> beta_cutoffs{0};
  std::atomic<uint64_t> first_move_cuts{0};
  std::atomic<uint64_t> killer_cuts{0};
  std::atomic<uint64_t> history_cuts{0};
  std::atomic<uint64_t> qsearch_nodes{0};
  std::atomic<uint64_t> null_move_cuts{0};
  std::atomic<uint64_t> pvs_researches{0};
  std::atomic<uint64_t> lmr_researches{0};
  std::atomic<uint64_t> qs_see_pruning{0};
  std::atomic<uint64_t> qs_delta_pruning{0};
  std::atomic<uint64_t> check_extensions{0};
  std::atomic<uint64_t> iid_searches{0};
  std::atomic<uint64_t> root_move_changes{0};
  double history_saturation = 0.0;
  int score = 0;
  Move best_move{};

  void reset() noexcept {
    nodes.store(0, std::memory_order_relaxed);
    depth = 0;
    seldepth = 0;
    tt_hits.store(0, std::memory_order_relaxed);
    hashfull = 0;
    beta_cutoffs.store(0, std::memory_order_relaxed);
    first_move_cuts.store(0, std::memory_order_relaxed);
    killer_cuts.store(0, std::memory_order_relaxed);
    history_cuts.store(0, std::memory_order_relaxed);
    qsearch_nodes.store(0, std::memory_order_relaxed);
    null_move_cuts.store(0, std::memory_order_relaxed);
    pvs_researches.store(0, std::memory_order_relaxed);
    lmr_researches.store(0, std::memory_order_relaxed);
    qs_see_pruning.store(0, std::memory_order_relaxed);
    qs_delta_pruning.store(0, std::memory_order_relaxed);
    check_extensions.store(0, std::memory_order_relaxed);
    iid_searches.store(0, std::memory_order_relaxed);
    root_move_changes.store(0, std::memory_order_relaxed);
    history_saturation = 0.0;
    score = 0;
    best_move = Move{};
  }

  // Non-copyable due to atomics; provide explicit copy semantics for snapshots
  SearchStats() = default;

  SearchStats(const SearchStats &other)
      : nodes(other.nodes.load(std::memory_order_relaxed)), depth(other.depth),
        seldepth(other.seldepth),
        tt_hits(other.tt_hits.load(std::memory_order_relaxed)),
        hashfull(other.hashfull),
        beta_cutoffs(other.beta_cutoffs.load(std::memory_order_relaxed)),
        first_move_cuts(other.first_move_cuts.load(std::memory_order_relaxed)),
        killer_cuts(other.killer_cuts.load(std::memory_order_relaxed)),
        history_cuts(other.history_cuts.load(std::memory_order_relaxed)),
        qsearch_nodes(other.qsearch_nodes.load(std::memory_order_relaxed)),
        null_move_cuts(other.null_move_cuts.load(std::memory_order_relaxed)),
        pvs_researches(other.pvs_researches.load(std::memory_order_relaxed)),
        lmr_researches(other.lmr_researches.load(std::memory_order_relaxed)),
        qs_see_pruning(other.qs_see_pruning.load(std::memory_order_relaxed)),
        qs_delta_pruning(
            other.qs_delta_pruning.load(std::memory_order_relaxed)),
        check_extensions(
            other.check_extensions.load(std::memory_order_relaxed)),
        iid_searches(other.iid_searches.load(std::memory_order_relaxed)),
        root_move_changes(
            other.root_move_changes.load(std::memory_order_relaxed)),
        history_saturation(other.history_saturation), score(other.score),
        best_move(other.best_move) {}

  SearchStats &operator=(const SearchStats &other) {
    if (this != &other) {
      nodes.store(other.nodes.load(std::memory_order_relaxed),
                  std::memory_order_relaxed);
      depth = other.depth;
      seldepth = other.seldepth;
      tt_hits.store(other.tt_hits.load(std::memory_order_relaxed),
                    std::memory_order_relaxed);
      hashfull = other.hashfull;
      beta_cutoffs.store(other.beta_cutoffs.load(std::memory_order_relaxed),
                         std::memory_order_relaxed);
      first_move_cuts.store(
          other.first_move_cuts.load(std::memory_order_relaxed),
          std::memory_order_relaxed);
      killer_cuts.store(other.killer_cuts.load(std::memory_order_relaxed),
                        std::memory_order_relaxed);
      history_cuts.store(other.history_cuts.load(std::memory_order_relaxed),
                         std::memory_order_relaxed);
      qsearch_nodes.store(other.qsearch_nodes.load(std::memory_order_relaxed),
                          std::memory_order_relaxed);
      null_move_cuts.store(other.null_move_cuts.load(std::memory_order_relaxed),
                           std::memory_order_relaxed);
      pvs_researches.store(other.pvs_researches.load(std::memory_order_relaxed),
                           std::memory_order_relaxed);
      lmr_researches.store(other.lmr_researches.load(std::memory_order_relaxed),
                           std::memory_order_relaxed);
      qs_see_pruning.store(other.qs_see_pruning.load(std::memory_order_relaxed),
                           std::memory_order_relaxed);
      qs_delta_pruning.store(
          other.qs_delta_pruning.load(std::memory_order_relaxed),
          std::memory_order_relaxed);
      check_extensions.store(
          other.check_extensions.load(std::memory_order_relaxed),
          std::memory_order_relaxed);
      iid_searches.store(other.iid_searches.load(std::memory_order_relaxed),
                         std::memory_order_relaxed);
      root_move_changes.store(
          other.root_move_changes.load(std::memory_order_relaxed),
          std::memory_order_relaxed);
      history_saturation = other.history_saturation;
      score = other.score;
      best_move = other.best_move;
    }
    return *this;
  }
};

// ── Search ──────────────────────────────────────────────────────────────

// Forward declaration for friend access.
class WorkerSearch;

class Search {
  friend class WorkerSearch;

public:
  static constexpr int NEG_INF = -200000;
  static constexpr int POS_INF = 200000;
  static constexpr int MATE_SCORE = 100000;
  static constexpr int TIME_CHECK_INTERVAL = 2048;

  Search(Board &board, const SearchConfig &config, eval::Evaluator *evaluator,
         syzygy::SyzygyProber *syzygy = nullptr);

  /// Main entry: runs IDDFS up to depth. Returns (score, best_move).
  /// Score is from White's perspective.
  std::pair<int, Move> find_best_move(int depth);

  /// Access search statistics (read-only snapshot).
  [[nodiscard]] const SearchStats &get_stats() const noexcept { return stats_; }

  /// Reset search state (TT, history, killers).
  void reset_state(bool clear_tt = true, bool clear_history = true,
                   bool clear_killers = true);

private:
  // Shared state
  Board &board_;
  SearchConfig config_;
  eval::Evaluator *evaluator_;
  syzygy::SyzygyProber *syzygy_;

  TranspositionTable tt_;
  Zobrist zobrist_;
  move_ordering::MoveSorter move_sorter_;

  SearchStats stats_;
  std::atomic<bool> time_up_{false};
  std::chrono::time_point<std::chrono::steady_clock> start_time_;
  Move root_best_move_{};
  bool has_root_best_move_ = false;

  // Core search functions
  int negamax(int depth, int alpha, int beta, int ply,
              const Move &previous_move, int extensions_left);
  int quiescence(int alpha, int beta, int ply, int qs_depth);
  int search_with_window(int depth, int alpha, int beta);
  int search_child(int index, int next_depth, int alpha, int beta, int ply,
                   const Move &move, bool in_check, bool gives_check,
                   bool is_tactical, int extensions_left);

  // Inlined helpers for critical paths
  inline int relative_eval() noexcept {
    int white_perspective = evaluator_->go(board_);
    return board_.get_side_to_move() ? white_perspective : -white_perspective;
  }

  static constexpr int terminal_score(GameState state, int ply) noexcept {
    if (state == GameState::CHECKMATE) {
      return -MATE_SCORE + ply;
    }
    return 0;
  }

  inline bool check_time_limit() noexcept {
    if (!config_.has_max_time) [[likely]] {
      return false;
    }
    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - start_time_).count();
    if (elapsed >= config_.max_time) [[unlikely]] {
      time_up_.store(true, std::memory_order_release);
      return true;
    }
    return false;
  }

  inline bool has_non_pawn_material() const noexcept {
    Color color = board_.get_side_to_move() ? Color::WHITE : Color::BLACK;
    Bitboard non_pawn = board_.get_piece_bb(PieceType::KNIGHT, color) |
                        board_.get_piece_bb(PieceType::BISHOP, color) |
                        board_.get_piece_bb(PieceType::ROOK, color) |
                        board_.get_piece_bb(PieceType::QUEEN, color);
    return non_pawn != 0;
  }

  static inline bool is_tactical_move(const Board &b,
                                      const Move &move) noexcept {
    return b.is_capture(move) || (move.promotion != 0);
  }

  int capture_gain(const Move &move) noexcept;
  bool can_apply_futility(int depth, int static_eval, int alpha, bool in_check,
                          bool is_tactical) const noexcept;
  bool can_apply_lmr(int move_index, int depth, bool in_check, bool gives_check,
                     bool is_tactical) const noexcept;

  static constexpr int lmr_reduction(int depth, int move_index) noexcept {
    int d =
        (depth < 0) ? 0 : (depth >= LMR_MAX_DEPTH ? LMR_MAX_DEPTH - 1 : depth);
    int m = (move_index < 0) ? 0
                             : (move_index >= LMR_MAX_MOVES ? LMR_MAX_MOVES - 1
                                                            : move_index);
    return LMR_TABLE.data[d][m];
  }

  static BoundType determine_bound(int best_score, int original_alpha,
                                   int beta) noexcept;

  // Zobrist helpers (inlined for hot path)
  inline uint64_t current_hash() noexcept {
    if (!config_.use_transposition_table) [[unlikely]] {
      return 0;
    }
    auto h = zobrist_.get_current_hash();
    if (h.has_value()) [[likely]] {
      return h.value();
    }
    return zobrist_.hash_board(board_);
  }

  inline uint64_t push_move_with_hash(const Move &move) noexcept {
    uint64_t saved_hash = 0;
    if (config_.use_transposition_table) {
      auto h = zobrist_.get_current_hash();
      if (h.has_value()) [[likely]] {
        saved_hash = h.value();
      }
    }

    uint64_t next_hash = 0;
    bool has_next_hash = false;
    if (config_.use_transposition_table && saved_hash != 0) [[likely]] {
      next_hash = zobrist_.make_move_hash(board_, move);
      has_next_hash = true;
    }

    board_.push(move);

    if (config_.use_transposition_table) {
      if (has_next_hash) [[likely]] {
        zobrist_.set_current_hash(next_hash);
      } else {
        (void)zobrist_.hash_board(board_);
      }
    }

    return saved_hash;
  }

  inline void pop_move_with_hash(uint64_t saved_hash) noexcept {
    (void)board_.pop();
    if (config_.use_transposition_table && saved_hash != 0) [[likely]] {
      zobrist_.set_current_hash(saved_hash);
    }
  }

  int null_move_search(int depth, int beta, int ply, int extensions_left);

  // Syzygy probe
  std::optional<int> probe_syzygy(int ply);

  // Lazy SMP
  void smp_search(int depth);

  // Build a MoveSorterConfig from our SearchConfig
  static move_ordering::MoveSorterConfig
  make_move_sorter_config(const SearchConfig &cfg);
};

// ── WorkerSearch ─────────────────────────────────────────────────────
// Lightweight search instance for Lazy SMP helper threads.
// Each worker has its own Board, Zobrist, MoveSorter and Evaluator,
// but shares the TT with the main thread.  Reads/writes to the shared
// TT use key verification to detect torn writes (Lazy SMP standard).

class WorkerSearch {
public:
  WorkerSearch(const Board &root_board, const SearchConfig &config,
               const eval::EvalConfig &eval_config,
               TranspositionTable &shared_tt, std::atomic<bool> &time_up,
               std::atomic<bool> &stop, SearchStats &shared_stats,
               int worker_id);

  /// Run IDDFS up to target_depth, writing results to shared TT.
  void run(int target_depth);

private:
  Board board_;
  SearchConfig config_;
  eval::Evaluator evaluator_;
  TranspositionTable &tt_; // shared
  Zobrist zobrist_;
  move_ordering::MoveSorter move_sorter_;

  std::atomic<bool> &time_up_; // shared
  std::atomic<bool> &stop_;    // shared
  SearchStats &shared_stats_;  // shared (for node counts)
  int worker_id_;
  Move root_best_move_{};

  int negamax(int depth, int alpha, int beta, int ply,
              const Move &previous_move, int extensions_left);
  int quiescence(int alpha, int beta, int ply, int qs_depth);

  inline int relative_eval() noexcept {
    int white_perspective = evaluator_.go(board_);
    return board_.get_side_to_move() ? white_perspective : -white_perspective;
  }

  inline bool should_stop() const noexcept {
    return time_up_.load(std::memory_order_acquire) ||
           stop_.load(std::memory_order_acquire);
  }

  inline uint64_t current_hash() noexcept {
    auto h = zobrist_.get_current_hash();
    if (h.has_value()) [[likely]] {
      return h.value();
    }
    return zobrist_.hash_board(board_);
  }

  inline uint64_t push_move_with_hash(const Move &move) noexcept {
    uint64_t saved_hash = 0;
    auto h = zobrist_.get_current_hash();
    if (h.has_value()) [[likely]] {
      saved_hash = h.value();
    }
    uint64_t next_hash = 0;
    bool has_next = false;
    if (saved_hash != 0) [[likely]] {
      next_hash = zobrist_.make_move_hash(board_, move);
      has_next = true;
    }
    board_.push(move);
    if (has_next) [[likely]] {
      zobrist_.set_current_hash(next_hash);
    } else {
      (void)zobrist_.hash_board(board_);
    }
    return saved_hash;
  }

  inline void pop_move_with_hash(uint64_t saved_hash) noexcept {
    (void)board_.pop();
    if (saved_hash != 0) [[likely]] {
      zobrist_.set_current_hash(saved_hash);
    }
  }
};

} // namespace search
