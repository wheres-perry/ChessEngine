#include "search.hpp"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <thread>
#include <vector>

namespace search {

namespace {
inline uint8_t move_key_from(const Move &m) noexcept { return m.from; }
inline uint8_t move_key_to(const Move &m) noexcept { return m.to; }
inline uint8_t move_key_promo(const Move &m) noexcept { return m.promotion; }
} // namespace

// =========================================================================
// Helpers
// =========================================================================

move_ordering::MoveSorterConfig
Search::make_move_sorter_config(const SearchConfig &cfg) {
  move_ordering::MoveSorterConfig ms{};
  ms.use_move_ordering = cfg.use_move_ordering;
  ms.use_mvv_lva = cfg.use_mvv_lva;
  ms.use_history_heuristic = cfg.use_history_heuristic;
  ms.use_countermove_heuristic = cfg.use_countermove_heuristic;
  ms.use_see_ordering = cfg.use_see_ordering;
  ms.use_killer_moves = cfg.use_killer_moves;
  ms.use_hash_move_ordering = cfg.use_hash_move_ordering;
  ms.history_max_score = cfg.history_max_score;
  ms.killer_slots_per_ply = cfg.killer_slots_per_ply;
  ms.see_capture_threshold = cfg.see_capture_threshold;
  return ms;
}

// =========================================================================
// Constructor
// =========================================================================

Search::Search(Board &board, const SearchConfig &config,
               eval::Evaluator *evaluator, syzygy::SyzygyProber *syzygy)
    : board_(board), config_(config), evaluator_(evaluator), syzygy_(syzygy),
      tt_(config.use_transposition_table
              ? static_cast<size_t>(config.tt_size_mb)
              : 1),
      zobrist_(), move_sorter_(make_move_sorter_config(config)) {
  if (config_.use_transposition_table) {
    (void)zobrist_.hash_board(board_);
  }
}

// =========================================================================
// reset_state
// =========================================================================

void Search::reset_state(bool clear_tt, bool clear_history,
                         bool clear_killers) {
  if (clear_tt && config_.use_transposition_table) {
    tt_.clear();
  }
  if (config_.use_move_ordering) {
    move_sorter_.reset(clear_history, clear_killers);
  }
  stats_.reset();
  root_best_move_ = Move{};
  has_root_best_move_ = false;
}

// =========================================================================
// find_best_move - main entry point (IDDFS)
// =========================================================================

std::pair<int, Move> Search::find_best_move(int depth) {
  int target_depth = std::max(1, depth);

  stats_.reset();
  time_up_.store(false, std::memory_order_release);
  start_time_ = std::chrono::steady_clock::now();
  root_best_move_ = Move{};
  has_root_best_move_ = false;

  if (config_.use_transposition_table && config_.use_tt_aging) {
    tt_.increment_age();
  }

  if (config_.use_transposition_table) {
    (void)zobrist_.hash_board(board_);
  }

  bool root_turn_is_white = board_.get_side_to_move();
  int previous_score = 0;
  bool has_previous_score = false;
  int final_relative_score = 0;
  bool has_final_score = false;

  // Check for Lazy SMP
  if (config_.use_lazy_smp && config_.smp_num_threads > 1) [[unlikely]] {
    smp_search(target_depth);
    // After SMP, stats_ and root_best_move_ are updated
    if (!has_root_best_move_) {
      return {0, Move{}};
    }
    int white_score = root_turn_is_white ? stats_.score : -stats_.score;
    stats_.score = white_score;
    return {white_score, root_best_move_};
  }

  for (int current_depth = 1; current_depth <= target_depth; ++current_depth) {
    if (check_time_limit()) [[unlikely]] {
      break;
    }

    int alpha = NEG_INF;
    int beta = POS_INF;
    if (config_.use_alpha_beta && config_.use_aspiration_windows &&
        has_previous_score) {
      int margin = std::max(10, config_.aspiration_window_margin);
      alpha = previous_score - margin;
      beta = previous_score + margin;
    }

    int relative_score = search_with_window(current_depth, alpha, beta);
    if (time_up_.load(std::memory_order_acquire)) [[unlikely]] {
      break;
    }

    previous_score = relative_score;
    has_previous_score = true;
    final_relative_score = relative_score;
    has_final_score = true;
    stats_.depth = current_depth;
  }

  if (!has_final_score) [[unlikely]] {
    return {0, Move{}};
  }

  if (config_.use_transposition_table) {
    stats_.hashfull = tt_.hashfull();
  }

  if (config_.use_move_ordering) {
    stats_.history_saturation = move_sorter_.history_saturation();
  }

  int white_score =
      root_turn_is_white ? final_relative_score : -final_relative_score;
  stats_.score = white_score;
  stats_.best_move = root_best_move_;
  return {white_score, root_best_move_};
}

// =========================================================================
// search_with_window - aspiration windows with retry
// =========================================================================

int Search::search_with_window(int depth, int alpha, int beta) {
  if (!(config_.use_alpha_beta && config_.use_aspiration_windows) ||
      alpha == NEG_INF || beta == POS_INF) {
    int a = config_.use_alpha_beta ? alpha : NEG_INF;
    int b = config_.use_alpha_beta ? beta : POS_INF;
    return negamax(depth, a, b, 0, move_ordering::NO_MOVE,
                   config_.max_check_extensions);
  }

  int current_alpha = alpha;
  int current_beta = beta;

  for (int retry = 0; retry < 6; ++retry) {
    int score = negamax(depth, current_alpha, current_beta, 0,
                        move_ordering::NO_MOVE, config_.max_check_extensions);
    if (time_up_.load(std::memory_order_acquire)) [[unlikely]] {
      return score;
    }
    if (score <= current_alpha) [[unlikely]] {
      current_alpha -= std::max(50, config_.aspiration_window_margin);
      continue;
    }
    if (score >= current_beta) [[unlikely]] {
      current_beta += std::max(50, config_.aspiration_window_margin);
      continue;
    }
    return score;
  }

  // Fallback: full window
  return negamax(depth, NEG_INF, POS_INF, 0, move_ordering::NO_MOVE,
                 config_.max_check_extensions);
}

// =========================================================================
// negamax - core search with all features
// =========================================================================

int Search::negamax(int depth, int alpha, int beta, int ply,
                    const Move &previous_move, int extensions_left) {
  stats_.nodes.fetch_add(1, std::memory_order_relaxed);
  if (ply > stats_.seldepth) {
    stats_.seldepth = ply;
  }

  // Time check every TIME_CHECK_INTERVAL nodes (unlikely to trigger)
  if ((stats_.nodes.load(std::memory_order_relaxed) % TIME_CHECK_INTERVAL ==
       0) &&
      check_time_limit()) [[unlikely]] {
    return relative_eval();
  }

  // Terminal detection (unlikely during normal search)
  GameState game_state = board_.is_game_over();
  if (game_state != GameState::ONGOING) [[unlikely]] {
    return terminal_score(game_state, ply);
  }

  // Syzygy probing
  if (syzygy_ != nullptr && ply > 0) [[unlikely]] {
    auto tb_score = probe_syzygy(ply);
    if (tb_score.has_value()) {
      return tb_score.value();
    }
  }

  // Leaf node
  if (depth <= 0) {
    if (config_.use_quiescence_search) {
      return quiescence(alpha, beta, ply, 0);
    }
    return relative_eval();
  }

  bool in_check = board_.is_check();

  // Check extension (before entering the node)
  if (config_.use_check_extensions && in_check && extensions_left > 0 &&
      config_.use_alpha_beta) {
    depth += 1;
    extensions_left -= 1;
    stats_.check_extensions.fetch_add(1, std::memory_order_relaxed);
  }

  // TT probe with prefetch
  uint64_t key = current_hash();
  Move hash_move = move_ordering::NO_MOVE;
  if (config_.use_transposition_table) {
    tt_.prefetch(key);
    const TTEntry *entry = tt_.probe(key);
    if (entry != nullptr) [[likely]] {
      hash_move = entry->best_move;
      if (config_.use_alpha_beta) {
        auto hit_score = tt_.try_get_score(*entry, depth, alpha, beta);
        if (hit_score.has_value()) {
          if (ply == 0 && !(entry->best_move == move_ordering::NO_MOVE)) {
            root_best_move_ = entry->best_move;
            has_root_best_move_ = true;
          }
          stats_.tt_hits.fetch_add(1, std::memory_order_relaxed);
          return hit_score.value();
        }
      } else {
        // Non-alpha-beta: only use exact entries with sufficient depth
        auto bound = static_cast<BoundType>(entry->bound);
        if (bound == BoundType::EXACT &&
            entry->depth >= static_cast<int16_t>(depth)) {
          if (ply == 0 && !(entry->best_move == move_ordering::NO_MOVE)) {
            root_best_move_ = entry->best_move;
            has_root_best_move_ = true;
          }
          stats_.tt_hits.fetch_add(1, std::memory_order_relaxed);
          return entry->score;
        }
      }
    }
  }

  int static_eval = relative_eval();

  // Reverse futility pruning
  if (config_.use_alpha_beta && config_.use_reverse_futility_pruning &&
      !in_check && depth <= config_.rfp_max_depth && beta < POS_INF) {
    int margin = config_.rfp_margin_multiplier * depth;
    if (static_eval - margin >= beta) [[unlikely]] {
      return beta;
    }
  }

  // Null-move pruning
  if (config_.use_alpha_beta && config_.use_null_move_pruning && !in_check &&
      depth >= config_.nmp_min_depth && has_non_pawn_material() &&
      beta < POS_INF) {
    int null_score = null_move_search(depth, beta, ply, extensions_left);
    if (null_score >= beta) [[likely]] {
      stats_.null_move_cuts.fetch_add(1, std::memory_order_relaxed);
      return beta;
    }
  }

  // IID (Internal Iterative Deepening)
  if (config_.use_iid && config_.use_alpha_beta &&
      depth >= config_.iid_min_depth && (hash_move == move_ordering::NO_MOVE) &&
      config_.use_transposition_table) {
    stats_.iid_searches.fetch_add(1, std::memory_order_relaxed);
    int shallow_depth = std::max(1, depth - config_.iid_depth_reduction);
    negamax(shallow_depth, alpha, beta, ply, previous_move, extensions_left);
    const TTEntry *iid_entry = tt_.probe(key);
    if (iid_entry != nullptr) {
      hash_move = iid_entry->best_move;
    }
  }

  // Generate legal moves
  std::vector<Move> legal_moves = board_.generate_legal_moves();
  if (legal_moves.empty()) [[unlikely]] {
    return in_check ? -MATE_SCORE + ply : 0;
  }

  // Sort moves
  if (config_.use_move_ordering) {
    legal_moves = move_sorter_.sort_moves(board_, legal_moves, ply, hash_move,
                                          previous_move);
  }

  int original_alpha = alpha;
  int best_score = NEG_INF;
  Move best_move = move_ordering::NO_MOVE;

  for (int index = 0; index < static_cast<int>(legal_moves.size()); ++index) {
    if (time_up_.load(std::memory_order_acquire)) [[unlikely]] {
      break;
    }

    const Move &move = legal_moves[index];
    bool is_tactical = is_tactical_move(board_, move);

    // Futility pruning
    if (can_apply_futility(depth, static_eval, alpha, in_check, is_tactical)) {
      continue;
    }

    uint64_t saved_hash = push_move_with_hash(move);
    bool gives_check = board_.is_check();

    int child_extensions = extensions_left;
    int next_depth = depth - 1;

    // Check extension for the child
    if (config_.use_check_extensions && gives_check && child_extensions > 0 &&
        config_.use_alpha_beta) {
      next_depth += 1;
      child_extensions -= 1;
      stats_.check_extensions.fetch_add(1, std::memory_order_relaxed);
    }

    int score =
        search_child(index, next_depth, alpha, beta, ply, move, in_check,
                     gives_check, is_tactical, child_extensions);

    pop_move_with_hash(saved_hash);

    if (score > best_score) {
      best_score = score;
      best_move = move;
      if (ply == 0) {
        if (!has_root_best_move_ || !(root_best_move_ == move)) {
          stats_.root_move_changes.fetch_add(1, std::memory_order_relaxed);
        }
        root_best_move_ = move;
        has_root_best_move_ = true;
      }
    }

    if (config_.use_alpha_beta) {
      alpha = std::max(alpha, score);
      if (alpha >= beta) [[likely]] {
        stats_.beta_cutoffs.fetch_add(1, std::memory_order_relaxed);
        if (index == 0) {
          stats_.first_move_cuts.fetch_add(1, std::memory_order_relaxed);
        }
        if (config_.use_move_ordering) {
          auto move_key = move_ordering::make_move_key(move);
          if (config_.use_killer_moves) {
            const auto &killers = move_sorter_.get_killers(ply);
            for (const auto &k : killers) {
              if (k == move) {
                stats_.killer_cuts.fetch_add(1, std::memory_order_relaxed);
                break;
              }
            }
          }
          if (config_.use_history_heuristic) {
            int hist_score = move_sorter_.get_history(
                move_key_from(move), move_key_to(move), move_key_promo(move));
            if (hist_score > 0) {
              stats_.history_cuts.fetch_add(1, std::memory_order_relaxed);
            }
          }
          move_sorter_.on_beta_cutoff(move, ply, depth, previous_move,
                                      is_tactical);
        }
        break;
      }
    }
  }

  if (best_move == move_ordering::NO_MOVE) [[unlikely]] {
    return static_eval;
  }

  // TT store
  if (config_.use_transposition_table) {
    BoundType bound = determine_bound(best_score, original_alpha, beta);
    tt_.store(key, depth, static_cast<int32_t>(best_score), best_move, bound);
  }

  return best_score;
}

// =========================================================================
// search_child - PVS + LMR dispatch
// =========================================================================

int Search::search_child(int index, int next_depth, int alpha, int beta,
                         int ply, const Move &move, bool in_check,
                         bool gives_check, bool is_tactical,
                         int extensions_left) {
  if (!config_.use_alpha_beta) [[unlikely]] {
    return -negamax(next_depth, NEG_INF, POS_INF, ply + 1, move,
                    extensions_left);
  }

  if (config_.use_pvs && index > 0) {
    int score = alpha + 1; // Will be overwritten

    // Try LMR first
    if (can_apply_lmr(index, next_depth, in_check, gives_check, is_tactical)) {
      int reduction = lmr_reduction(next_depth, index);
      int reduced_depth = std::max(0, next_depth - reduction);
      score = -negamax(reduced_depth, -alpha - 1, -alpha, ply + 1, move,
                       extensions_left);
      if (score > alpha) {
        stats_.lmr_researches.fetch_add(1, std::memory_order_relaxed);
      }
    }

    // Null-window search
    if (score > alpha) {
      score = -negamax(next_depth, -alpha - 1, -alpha, ply + 1, move,
                       extensions_left);
      // Full re-search if score is between alpha and beta
      if (alpha < score && score < beta) {
        stats_.pvs_researches.fetch_add(1, std::memory_order_relaxed);
        score =
            -negamax(next_depth, -beta, -alpha, ply + 1, move, extensions_left);
      }
    }

    return score;
  }

  // First move or non-PVS: full window
  return -negamax(next_depth, -beta, -alpha, ply + 1, move, extensions_left);
}

// =========================================================================
// quiescence - captures/promotions search
// =========================================================================

int Search::quiescence(int alpha, int beta, int ply, int qs_depth) {
  stats_.qsearch_nodes.fetch_add(1, std::memory_order_relaxed);
  if (ply > stats_.seldepth) {
    stats_.seldepth = ply;
  }

  if (qs_depth >= config_.qs_max_depth) [[unlikely]] {
    return relative_eval();
  }

  GameState game_state = board_.is_game_over();
  if (game_state != GameState::ONGOING) [[unlikely]] {
    return terminal_score(game_state, ply);
  }

  int stand_pat = relative_eval();
  if (config_.use_alpha_beta) {
    if (stand_pat >= beta) [[likely]] {
      return beta;
    }
    alpha = std::max(alpha, stand_pat);
  } else {
    alpha = std::max(alpha, stand_pat);
  }

  // Generate only tactical moves
  std::vector<Move> all_moves = board_.generate_legal_moves();
  std::vector<Move> tactical_moves;
  tactical_moves.reserve(all_moves.size());
  for (const auto &move : all_moves) {
    if (is_tactical_move(board_, move)) {
      tactical_moves.push_back(move);
    }
  }

  if (tactical_moves.empty()) [[likely]] {
    return alpha;
  }

  // Sort tactical moves
  if (config_.use_move_ordering) {
    tactical_moves = move_sorter_.sort_tactical(board_, tactical_moves);
  }

  for (const auto &move : tactical_moves) {
    // Delta pruning
    if (config_.use_delta_pruning && config_.use_alpha_beta) {
      int gain = capture_gain(move);
      if (stand_pat + gain + config_.delta_margin < alpha) [[unlikely]] {
        stats_.qs_delta_pruning.fetch_add(1, std::memory_order_relaxed);
        continue;
      }
    }

    // SEE pruning
    if (config_.use_see_pruning_in_qs && config_.use_move_ordering &&
        board_.is_capture(move) && move_sorter_.see(board_, move) < 0)
        [[unlikely]] {
      stats_.qs_see_pruning.fetch_add(1, std::memory_order_relaxed);
      continue;
    }

    uint64_t saved_hash = push_move_with_hash(move);
    int score = -quiescence(-beta, -alpha, ply + 1, qs_depth + 1);
    pop_move_with_hash(saved_hash);

    if (config_.use_alpha_beta && score >= beta) [[likely]] {
      return beta;
    }
    alpha = std::max(alpha, score);
  }

  return alpha;
}

// =========================================================================
// null_move_search
// =========================================================================

int Search::null_move_search(int depth, int beta, int ply,
                             int extensions_left) {
  uint64_t saved_hash = 0;
  bool had_hash = false;
  if (config_.use_transposition_table) {
    auto h = zobrist_.get_current_hash();
    if (h.has_value()) [[likely]] {
      saved_hash = h.value();
      had_hash = true;
    }
  }

  uint64_t null_hash = 0;
  bool has_null_hash = false;
  if (config_.use_transposition_table && had_hash) [[likely]] {
    null_hash = zobrist_.make_null_move_hash(board_);
    has_null_hash = true;
  }

  board_.push_null();

  if (config_.use_transposition_table) {
    if (has_null_hash) [[likely]] {
      zobrist_.set_current_hash(null_hash);
    } else {
      (void)zobrist_.hash_board(board_);
    }
  }

  int reduction = std::max(1, config_.nmp_reduction_r);
  int score = -negamax(std::max(0, depth - 1 - reduction), -beta, -beta + 1,
                       ply + 1, move_ordering::NO_MOVE, extensions_left);

  (void)board_.pop();
  if (config_.use_transposition_table && had_hash) [[likely]] {
    zobrist_.set_current_hash(saved_hash);
  }

  return score;
}

// =========================================================================
// Material helpers
// =========================================================================

int Search::capture_gain(const Move &move) noexcept {
  auto piece = board_.piece_at(move.to);
  if (!piece.has_value() && board_.is_en_passant(move)) {
    return move_ordering::PIECE_VALUES_CP[static_cast<int>(PieceType::PAWN)];
  }
  if (!piece.has_value()) {
    return 0;
  }
  return move_ordering::PIECE_VALUES_CP[static_cast<int>(piece->type)];
}

// =========================================================================
// Pruning helpers
// =========================================================================

bool Search::can_apply_futility(int depth, int static_eval, int alpha,
                                bool in_check,
                                bool is_tactical) const noexcept {
  if (!config_.use_alpha_beta) {
    return false;
  }
  if (in_check || is_tactical) {
    return false;
  }
  if (config_.use_futility_pruning && depth == 1 &&
      static_eval + config_.futility_margin_standard <= alpha) {
    return true;
  }
  return config_.use_extended_futility_pruning && depth == 2 &&
         static_eval + config_.futility_margin_extended <= alpha;
}

bool Search::can_apply_lmr(int move_index, int depth, bool in_check,
                           bool gives_check, bool is_tactical) const noexcept {
  if (!config_.use_lmr) {
    return false;
  }
  if (in_check || gives_check || is_tactical) {
    return false;
  }
  if (depth < config_.lmr_min_depth) {
    return false;
  }
  return move_index >= config_.lmr_min_move_number;
}

BoundType Search::determine_bound(int best_score, int original_alpha,
                                  int beta) noexcept {
  if (best_score <= original_alpha) {
    return BoundType::UPPER;
  }
  if (best_score >= beta) {
    return BoundType::LOWER;
  }
  return BoundType::EXACT;
}

// =========================================================================
// Syzygy probe
// =========================================================================

std::optional<int> Search::probe_syzygy(int ply) {
  if (syzygy_ == nullptr) [[unlikely]] {
    return std::nullopt;
  }
  auto wdl = syzygy_->probe_wdl(board_);
  if (!wdl.has_value()) {
    return std::nullopt;
  }
  if (wdl.value() > 0) {
    auto dtz = syzygy_->probe_dtz(board_);
    int distance = dtz.has_value() ? std::abs(dtz.value()) : 100;
    return MATE_SCORE - distance - ply;
  }
  if (wdl.value() < 0) {
    auto dtz = syzygy_->probe_dtz(board_);
    int distance = dtz.has_value() ? std::abs(dtz.value()) : 100;
    return -MATE_SCORE + distance + ply;
  }
  return 0;
}

// =========================================================================
// Lazy SMP
// =========================================================================

void Search::smp_search(int target_depth) {
  // The main thread and N-1 worker threads search independently.
  // They share the TT (reads/writes are naturally safe enough for Lazy SMP).
  // Each worker gets its own Board copy, MoveSorter, Zobrist, and Evaluator.

  int num_threads = std::max(1, config_.smp_num_threads);
  std::atomic<bool> stop{false};

  std::vector<std::thread> threads;
  threads.reserve(num_threads - 1);

  // Get the eval config from the evaluator for constructing worker evaluators.
  eval::EvalConfig eval_cfg = evaluator_->config();

  // Launch helper threads — each runs a real alpha-beta search with IDDFS.
  // Odd workers search depth-1, even workers search depth+1 for diversity.
  for (int i = 1; i < num_threads; ++i) {
    threads.emplace_back([&, i, eval_cfg]() {
      WorkerSearch worker(board_, config_, eval_cfg, tt_, time_up_, stop,
                          stats_, i);
      worker.run(target_depth);
    });
  }

  // Main thread (worker 0) runs the full search with all features.
  int previous_score = 0;
  bool has_previous_score = false;
  int final_relative_score = 0;
  bool has_final_score = false;

  for (int current_depth = 1; current_depth <= target_depth; ++current_depth) {
    if (check_time_limit()) [[unlikely]] {
      break;
    }

    int alpha = NEG_INF;
    int beta = POS_INF;
    if (config_.use_alpha_beta && config_.use_aspiration_windows &&
        has_previous_score) {
      int margin = std::max(10, config_.aspiration_window_margin);
      alpha = previous_score - margin;
      beta = previous_score + margin;
    }

    int relative_score = search_with_window(current_depth, alpha, beta);
    if (time_up_.load(std::memory_order_acquire)) [[unlikely]] {
      break;
    }

    previous_score = relative_score;
    has_previous_score = true;
    final_relative_score = relative_score;
    has_final_score = true;
    stats_.depth = current_depth;
  }

  // Signal workers to stop
  stop.store(true, std::memory_order_release);

  // Wait for workers
  for (auto &t : threads) {
    if (t.joinable()) {
      t.join();
    }
  }

  // Use main thread result
  if (has_final_score) {
    stats_.score = final_relative_score;
    stats_.best_move = root_best_move_;
  }
}

// =========================================================================
// WorkerSearch — Lazy SMP helper thread search
// =========================================================================

WorkerSearch::WorkerSearch(const Board &root_board, const SearchConfig &config,
                           const eval::EvalConfig &eval_config,
                           TranspositionTable &shared_tt,
                           std::atomic<bool> &time_up, std::atomic<bool> &stop,
                           SearchStats &shared_stats, int worker_id)
    : board_(root_board.copy()), config_(config), evaluator_(eval_config),
      tt_(shared_tt), zobrist_(),
      move_sorter_(Search::make_move_sorter_config(config)), time_up_(time_up),
      stop_(stop), shared_stats_(shared_stats), worker_id_(worker_id) {
  if (config_.use_transposition_table) {
    (void)zobrist_.hash_board(board_);
  }
}

void WorkerSearch::run(int target_depth) {
  for (int d = 1; d <= target_depth; ++d) {
    if (should_stop())
      break;

    // Depth variation for diversity:
    // Odd workers search depth+1, even workers search depth-1 (when d>1).
    int search_depth = d;
    if ((worker_id_ % 2) == 0 && d > 1) {
      search_depth = d - 1;
    } else if ((worker_id_ % 2) == 1 && d < target_depth) {
      search_depth = d + 1;
    }
    search_depth = std::max(1, search_depth);

    // Full-window search (no aspiration for helper threads).
    negamax(search_depth, Search::NEG_INF, Search::POS_INF, 0,
            move_ordering::NO_MOVE, config_.max_check_extensions);
  }
}

int WorkerSearch::negamax(int depth, int alpha, int beta, int ply,
                          const Move &previous_move, int extensions_left) {
  shared_stats_.nodes.fetch_add(1, std::memory_order_relaxed);

  // Check stop condition periodically
  if ((shared_stats_.nodes.load(std::memory_order_relaxed) %
       Search::TIME_CHECK_INTERVAL) == 0) {
    if (should_stop()) {
      return relative_eval();
    }
  }

  // Terminal detection
  GameState game_state = board_.is_game_over();
  if (game_state != GameState::ONGOING) [[unlikely]] {
    return Search::terminal_score(game_state, ply);
  }

  // Leaf node
  if (depth <= 0) {
    if (config_.use_quiescence_search) {
      return quiescence(alpha, beta, ply, 0);
    }
    return relative_eval();
  }

  bool in_check = board_.is_check();

  // Check extension
  if (config_.use_check_extensions && in_check && extensions_left > 0) {
    depth += 1;
    extensions_left -= 1;
  }

  // TT probe (shared, with key verification for Lazy SMP)
  uint64_t key = current_hash();
  Move hash_move = move_ordering::NO_MOVE;
  if (config_.use_transposition_table) {
    tt_.prefetch(key);
    const TTEntry *entry = tt_.probe(key);
    if (entry != nullptr) [[likely]] {
      // Key verification: entry.key == key already checked by probe().
      // In Lazy SMP, torn writes may produce a wrong entry, but
      // the key check catches most of these.
      hash_move = entry->best_move;
      auto hit_score = tt_.try_get_score(*entry, depth, alpha, beta);
      if (hit_score.has_value()) {
        shared_stats_.tt_hits.fetch_add(1, std::memory_order_relaxed);
        return hit_score.value();
      }
    }
  }

  int static_eval = relative_eval();

  // Reverse futility pruning
  if (config_.use_reverse_futility_pruning && !in_check &&
      depth <= config_.rfp_max_depth && beta < Search::POS_INF) {
    int margin = config_.rfp_margin_multiplier * depth;
    if (static_eval - margin >= beta) {
      return beta;
    }
  }

  // Null-move pruning
  if (config_.use_null_move_pruning && !in_check &&
      depth >= config_.nmp_min_depth && beta < Search::POS_INF) {
    // Check for non-pawn material
    Color color = board_.get_side_to_move() ? Color::WHITE : Color::BLACK;
    Bitboard non_pawn = board_.get_piece_bb(PieceType::KNIGHT, color) |
                        board_.get_piece_bb(PieceType::BISHOP, color) |
                        board_.get_piece_bb(PieceType::ROOK, color) |
                        board_.get_piece_bb(PieceType::QUEEN, color);
    if (non_pawn != 0) {
      // Null move search
      uint64_t saved_hash = 0;
      auto h = zobrist_.get_current_hash();
      if (h.has_value()) [[likely]] {
        saved_hash = h.value();
      }
      uint64_t null_hash = 0;
      bool has_null = false;
      if (saved_hash != 0) [[likely]] {
        null_hash = zobrist_.make_null_move_hash(board_);
        has_null = true;
      }
      board_.push_null();
      if (has_null) [[likely]] {
        zobrist_.set_current_hash(null_hash);
      } else {
        (void)zobrist_.hash_board(board_);
      }
      int reduction = std::max(1, config_.nmp_reduction_r);
      int null_score =
          -negamax(std::max(0, depth - 1 - reduction), -beta, -beta + 1,
                   ply + 1, move_ordering::NO_MOVE, extensions_left);
      (void)board_.pop();
      if (saved_hash != 0) [[likely]] {
        zobrist_.set_current_hash(saved_hash);
      }
      if (null_score >= beta) {
        shared_stats_.null_move_cuts.fetch_add(1, std::memory_order_relaxed);
        return beta;
      }
    }
  }

  // Generate legal moves
  std::vector<Move> legal_moves = board_.generate_legal_moves();
  if (legal_moves.empty()) [[unlikely]] {
    return in_check ? -Search::MATE_SCORE + ply : 0;
  }

  // Sort moves
  if (config_.use_move_ordering) {
    legal_moves = move_sorter_.sort_moves(board_, legal_moves, ply, hash_move,
                                          previous_move);
  }

  int original_alpha = alpha;
  int best_score = Search::NEG_INF;
  Move best_move = move_ordering::NO_MOVE;

  for (int index = 0; index < static_cast<int>(legal_moves.size()); ++index) {
    if (should_stop()) [[unlikely]] {
      break;
    }

    const Move &move = legal_moves[index];
    bool is_tactical = board_.is_capture(move) || (move.promotion != 0);

    uint64_t saved_hash = push_move_with_hash(move);
    bool gives_check = board_.is_check();

    int child_ext = extensions_left;
    int next_depth = depth - 1;
    if (config_.use_check_extensions && gives_check && child_ext > 0) {
      next_depth += 1;
      child_ext -= 1;
    }

    int score;
    if (config_.use_pvs && index > 0) {
      // LMR
      score = alpha + 1;
      if (config_.use_lmr && !in_check && !gives_check && !is_tactical &&
          depth >= config_.lmr_min_depth &&
          index >= config_.lmr_min_move_number) {
        int reduction = Search::lmr_reduction(next_depth, index);
        int reduced = std::max(0, next_depth - reduction);
        score = -negamax(reduced, -alpha - 1, -alpha, ply + 1, move, child_ext);
      }
      // Null-window
      if (score > alpha) {
        score =
            -negamax(next_depth, -alpha - 1, -alpha, ply + 1, move, child_ext);
        if (alpha < score && score < beta) {
          score = -negamax(next_depth, -beta, -alpha, ply + 1, move, child_ext);
        }
      }
    } else {
      score = -negamax(next_depth, -beta, -alpha, ply + 1, move, child_ext);
    }

    pop_move_with_hash(saved_hash);

    if (score > best_score) {
      best_score = score;
      best_move = move;
      if (ply == 0) {
        root_best_move_ = move;
      }
    }

    alpha = std::max(alpha, score);
    if (alpha >= beta) [[likely]] {
      shared_stats_.beta_cutoffs.fetch_add(1, std::memory_order_relaxed);
      if (config_.use_move_ordering) {
        move_sorter_.on_beta_cutoff(move, ply, depth, previous_move,
                                    is_tactical);
      }
      break;
    }
  }

  if (best_move == move_ordering::NO_MOVE) [[unlikely]] {
    return static_eval;
  }

  // TT store (shared — Lazy SMP tolerates torn writes with key verification)
  if (config_.use_transposition_table) {
    BoundType bound;
    if (best_score <= original_alpha) {
      bound = BoundType::UPPER;
    } else if (best_score >= beta) {
      bound = BoundType::LOWER;
    } else {
      bound = BoundType::EXACT;
    }
    tt_.store(key, depth, static_cast<int32_t>(best_score), best_move, bound);
  }

  return best_score;
}

int WorkerSearch::quiescence(int alpha, int beta, int ply, int qs_depth) {
  shared_stats_.qsearch_nodes.fetch_add(1, std::memory_order_relaxed);

  if (qs_depth >= config_.qs_max_depth) [[unlikely]] {
    return relative_eval();
  }

  GameState game_state = board_.is_game_over();
  if (game_state != GameState::ONGOING) [[unlikely]] {
    return Search::terminal_score(game_state, ply);
  }

  int stand_pat = relative_eval();
  if (stand_pat >= beta) [[likely]] {
    return beta;
  }
  alpha = std::max(alpha, stand_pat);

  // Generate only tactical moves
  std::vector<Move> all_moves = board_.generate_legal_moves();
  std::vector<Move> tactical_moves;
  tactical_moves.reserve(all_moves.size());
  for (const auto &move : all_moves) {
    if (board_.is_capture(move) || move.promotion != 0) {
      tactical_moves.push_back(move);
    }
  }

  if (tactical_moves.empty()) [[likely]] {
    return alpha;
  }

  if (config_.use_move_ordering) {
    tactical_moves = move_sorter_.sort_tactical(board_, tactical_moves);
  }

  for (const auto &move : tactical_moves) {
    if (should_stop()) [[unlikely]] {
      break;
    }

    // Delta pruning
    if (config_.use_delta_pruning) {
      auto piece = board_.piece_at(move.to);
      int gain = 0;
      if (!piece.has_value() && board_.is_en_passant(move)) {
        gain =
            move_ordering::PIECE_VALUES_CP[static_cast<int>(PieceType::PAWN)];
      } else if (piece.has_value()) {
        gain = move_ordering::PIECE_VALUES_CP[static_cast<int>(piece->type)];
      }
      if (stand_pat + gain + config_.delta_margin < alpha) {
        continue;
      }
    }

    // SEE pruning
    if (config_.use_see_pruning_in_qs && config_.use_move_ordering &&
        board_.is_capture(move) && move_sorter_.see(board_, move) < 0) {
      continue;
    }

    uint64_t saved_hash = push_move_with_hash(move);
    int score = -quiescence(-beta, -alpha, ply + 1, qs_depth + 1);
    pop_move_with_hash(saved_hash);

    if (score >= beta) [[likely]] {
      return beta;
    }
    alpha = std::max(alpha, score);
  }

  return alpha;
}

} // namespace search
