#include "minimax.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace search {

namespace {

constexpr double NEG_INF = -std::numeric_limits<double>::infinity();
constexpr double POS_INF = std::numeric_limits<double>::infinity();

[[nodiscard]] constexpr bool is_finite(double v) noexcept {
  return v != NEG_INF && v != POS_INF;
}

} // namespace

Minimax::Minimax(Board &board, evaluators::IEvaluator &evaluator,
                 TranspositionTable *tt, MoveSorter *sorter, Zobrist *zobrist,
                 const CppSearchConfig &config) noexcept
    : board_(board), evaluator_(evaluator), tt_(tt), move_sorter_(sorter),
      zobrist_(zobrist), config_(config) {}

void Minimax::reset_state(bool clear_tt, bool clear_history,
                          bool clear_killers) noexcept {
  if (clear_tt && tt_ != nullptr) {
    tt_->clear();
  }
  if (move_sorter_ != nullptr) {
    move_sorter_->reset(clear_history, clear_killers);
  }
  stats_.reset();
  root_best_move_.reset();
}

void Minimax::reset_clock() noexcept {
  start_time_ = Clock::now();
  time_up_ = false;
}

bool Minimax::check_time_limit() noexcept {
  if (!config_.max_time.has_value() || !start_time_.has_value()) {
    return false;
  }
  const auto elapsed =
      std::chrono::duration<double>(Clock::now() - *start_time_);
  if (elapsed.count() >= *config_.max_time) {
    time_up_ = true;
    return true;
  }
  return false;
}

Minimax::Result Minimax::find_best_move(int depth) {
  const int target_depth = std::max(1, depth);
  stats_.reset();
  time_up_ = false;
  start_time_ = Clock::now();
  root_best_move_.reset();

  if (tt_ != nullptr) {
    tt_->increment_age();
  }
  if (zobrist_ != nullptr) {
    (void)zobrist_->hash_board(board_);
  }

  const bool root_turn_is_white = board_.get_side_to_move();
  std::optional<double> previous_score;
  std::optional<double> final_relative_score;

  for (int current_depth = 1; current_depth <= target_depth; ++current_depth) {
    if (check_time_limit()) {
      break;
    }

    double alpha = NEG_INF;
    double beta = POS_INF;
    if (config_.use_alpha_beta && config_.use_aspiration_windows &&
        previous_score.has_value()) {
      const double margin =
          static_cast<double>(std::max(10, config_.aspiration_window_margin));
      alpha = *previous_score - margin;
      beta = *previous_score + margin;
    }

    const double relative_score =
        search_with_window(current_depth, alpha, beta);
    if (time_up_) {
      break;
    }

    previous_score = relative_score;
    final_relative_score = relative_score;
    stats_.depth = current_depth;
  }

  Result result;
  if (!final_relative_score.has_value()) {
    return result;
  }

  const std::vector<Move> legal_moves = board_.generate_legal_moves();
  if (legal_moves.empty()) {
    return result;
  }

  // Mandatory safeguard: if search completed without setting a root best move,
  // pick the first legal move as a fallback so the engine never returns nullopt
  // / 0000.
  if (!root_best_move_.has_value()) {
    root_best_move_ = legal_moves[0];
  }

  if (tt_ != nullptr && tt_->max_entries() > 0) {
    stats_.hashfull =
        static_cast<int>((tt_->size() * 1000) / tt_->max_entries());
  }
  if (move_sorter_ != nullptr) {
    stats_.history_saturation = move_sorter_->history_saturation();
  }

  const double white_score =
      root_turn_is_white ? *final_relative_score : -*final_relative_score;
  stats_.score = static_cast<int>(white_score);

  result.score = white_score;
  result.best_move = root_best_move_;
  return result;
}

double Minimax::search_with_window(int depth, double alpha, double beta) {
  const bool use_aspiration = config_.use_alpha_beta &&
                              config_.use_aspiration_windows &&
                              is_finite(alpha) && is_finite(beta);

  if (!use_aspiration) {
    const double a = config_.use_alpha_beta ? alpha : NEG_INF;
    const double b = config_.use_alpha_beta ? beta : POS_INF;
    return negamax(depth, a, b, 0, std::nullopt, config_.max_check_extensions);
  }

  double current_alpha = alpha;
  double current_beta = beta;

  for (int attempt = 0; attempt < 6; ++attempt) {
    const double score = negamax(depth, current_alpha, current_beta, 0,
                                 std::nullopt, config_.max_check_extensions);
    if (time_up_) {
      return score;
    }
    if (score <= current_alpha) {
      current_alpha -=
          static_cast<double>(std::max(50, config_.aspiration_window_margin));
      continue;
    }
    if (score >= current_beta) {
      current_beta +=
          static_cast<double>(std::max(50, config_.aspiration_window_margin));
      continue;
    }
    return score;
  }

  return negamax(depth, NEG_INF, POS_INF, 0, std::nullopt,
                 config_.max_check_extensions);
}

double Minimax::negamax(int depth, double alpha, double beta, int ply,
                        std::optional<Move> previous_move,
                        int extensions_left) {
  stats_.nodes += 1;
  if (ply > stats_.seldepth) {
    stats_.seldepth = ply;
  }

  if (stats_.nodes % TIME_CHECK_INTERVAL == 0 && check_time_limit()) {
    return relative_eval();
  }

  const GameState game_state = board_.is_game_over();
  if (game_state != GameState::ONGOING) {
    return terminal_score(game_state, ply);
  }

  if (depth <= 0) {
    if (config_.use_quiescence_search) {
      return quiescence(alpha, beta, ply, 0);
    }
    return relative_eval();
  }

  bool in_check = board_.is_check();
  if (config_.use_check_extensions && in_check && extensions_left > 0 &&
      config_.use_alpha_beta) {
    depth += 1;
    extensions_left -= 1;
    stats_.check_extensions += 1;
  }

  const auto key_opt = current_hash();
  std::optional<Move> hash_move;
  if (tt_ != nullptr && key_opt.has_value()) {
    TTEntry *entry = tt_->probe(*key_opt);
    if (entry != nullptr) {
      if (entry->has_best_move) {
        hash_move = entry->best_move;
      }
      if (config_.use_alpha_beta) {
        auto hit_score = tt_->try_get_score(*entry, depth, alpha, beta);
        if (hit_score.has_value()) {
          if (ply == 0) {
            if (entry->has_best_move) {
              root_best_move_ = entry->best_move;
              stats_.tt_hits += 1;
              return *hit_score;
            }
            // If at root and entry has no best move, do not cut off so
            // root_best_move_ is populated.
          } else {
            stats_.tt_hits += 1;
            return *hit_score;
          }
        }
      } else if (entry->bound == TTBound::EXACT && entry->depth >= depth) {
        if (ply == 0) {
          if (entry->has_best_move) {
            root_best_move_ = entry->best_move;
            stats_.tt_hits += 1;
            return entry->score;
          }
        } else {
          stats_.tt_hits += 1;
          return entry->score;
        }
      }
    }
  }

  const double static_eval = relative_eval();

  if (config_.use_alpha_beta && config_.use_reverse_futility_pruning &&
      !in_check && depth <= config_.rfp_max_depth && beta < POS_INF) {
    const double margin =
        static_cast<double>(config_.rfp_margin_multiplier * depth);
    if (static_eval - margin >= beta) {
      return beta;
    }
  }

  if (config_.use_alpha_beta && config_.use_null_move_pruning && !in_check &&
      depth >= config_.nmp_min_depth && has_non_pawn_material() &&
      beta < POS_INF) {
    const double null_score =
        null_move_search(depth, beta, ply, extensions_left);
    if (null_score >= beta) {
      stats_.null_move_cuts += 1;
      return beta;
    }
  }

  if (config_.use_iid && config_.use_alpha_beta &&
      depth >= config_.iid_min_depth && !hash_move.has_value() &&
      tt_ != nullptr && key_opt.has_value()) {
    stats_.iid_searches += 1;
    const int shallow_depth = std::max(1, depth - config_.iid_depth_reduction);
    (void)negamax(shallow_depth, alpha, beta, ply, previous_move,
                  extensions_left);
    TTEntry *iid_entry = tt_->probe(*key_opt);
    if (iid_entry != nullptr && iid_entry->has_best_move) {
      hash_move = iid_entry->best_move;
    }
  }

  std::vector<Move> legal_moves = board_.generate_legal_moves();
  if (legal_moves.empty()) {
    return in_check ? -MATE_SCORE + ply : 0.0;
  }

  if (move_sorter_ != nullptr) {
    legal_moves = move_sorter_->sort_moves(board_, legal_moves, ply, hash_move,
                                           previous_move);
  }

  const double original_alpha = alpha;
  double best_score = NEG_INF;
  std::optional<Move> best_move;

  for (size_t index = 0; index < legal_moves.size(); ++index) {
    if (time_up_) {
      break;
    }

    const Move &move = legal_moves[index];
    const bool is_tactical = is_tactical_move(move);
    if (can_apply_futility(depth, static_eval, alpha, in_check, is_tactical)) {
      continue;
    }

    auto saved_hash = push_move_with_hash(move);
    const bool gives_check = board_.is_check();

    int child_extensions = extensions_left;
    int next_depth = depth - 1;
    if (config_.use_check_extensions && gives_check && child_extensions > 0 &&
        config_.use_alpha_beta) {
      next_depth += 1;
      child_extensions -= 1;
      stats_.check_extensions += 1;
    }

    const double score = search_child(
        static_cast<int>(index), next_depth, alpha, beta, ply, move, in_check,
        gives_check, is_tactical, child_extensions);

    pop_move_with_hash(saved_hash);

    if (score > best_score) {
      best_score = score;
      best_move = move;
      if (ply == 0) {
        if (!root_best_move_.has_value() || *root_best_move_ != move) {
          stats_.root_move_changes += 1;
        }
        root_best_move_ = move;
      }
    }

    if (config_.use_alpha_beta) {
      alpha = std::max(alpha, score);
      if (alpha >= beta) {
        stats_.beta_cutoffs += 1;
        if (index == 0) {
          stats_.first_move_cuts += 1;
        }
        if (move_sorter_ != nullptr) {
          if (config_.use_killer_moves && move_sorter_->is_killer(ply, move)) {
            stats_.killer_cuts += 1;
          }
          if (config_.use_history_heuristic &&
              move_sorter_->history_get(move.from, move.to, move.promotion) >
                  0) {
            stats_.history_cuts += 1;
          }
          move_sorter_->on_beta_cutoff(move, ply, depth, previous_move,
                                       is_tactical);
        }
        break;
      }
    }
  }

  if (!best_move.has_value()) {
    return static_eval;
  }

  if (tt_ != nullptr && key_opt.has_value()) {
    const TTBound bound = determine_bound(best_score, original_alpha, beta);
    tt_->store(*key_opt, depth, best_score, best_move, bound);
  }

  return best_score;
}

double Minimax::search_child(int index, int next_depth, double alpha,
                             double beta, int ply, const Move &move,
                             bool in_check, bool gives_check, bool is_tactical,
                             int extensions_left) {
  if (!config_.use_alpha_beta) {
    return -negamax(next_depth, NEG_INF, POS_INF, ply + 1, move,
                    extensions_left);
  }

  if (config_.use_pvs && index > 0) {
    double score = alpha + 1.0;
    if (can_apply_lmr(index, next_depth, in_check, gives_check, is_tactical)) {
      const int reduction = lmr_reduction(next_depth, index);
      const int reduced_depth = std::max(0, next_depth - reduction);
      score = -negamax(reduced_depth, -alpha - 1.0, -alpha, ply + 1, move,
                       extensions_left);
      if (score > alpha) {
        stats_.lmr_researches += 1;
      }
    }

    if (score > alpha) {
      score = -negamax(next_depth, -alpha - 1.0, -alpha, ply + 1, move,
                       extensions_left);
      if (alpha < score && score < beta) {
        stats_.pvs_researches += 1;
        score =
            -negamax(next_depth, -beta, -alpha, ply + 1, move, extensions_left);
      }
    }
    return score;
  }

  return -negamax(next_depth, -beta, -alpha, ply + 1, move, extensions_left);
}

double Minimax::quiescence(double alpha, double beta, int ply, int qs_depth) {
  stats_.qsearch_nodes += 1;
  if (ply > stats_.seldepth) {
    stats_.seldepth = ply;
  }

  if (qs_depth >= config_.qs_max_depth) {
    return relative_eval();
  }

  const GameState game_state = board_.is_game_over();
  if (game_state != GameState::ONGOING) {
    return terminal_score(game_state, ply);
  }

  const double stand_pat = relative_eval();
  if (config_.use_alpha_beta) {
    if (stand_pat >= beta) {
      return beta;
    }
    if (stand_pat > alpha) {
      alpha = stand_pat;
    }
  } else if (stand_pat > alpha) {
    alpha = stand_pat;
  }

  std::vector<Move> tactical;
  for (const auto &move : board_.generate_legal_moves()) {
    if (is_tactical_move(move)) {
      tactical.push_back(move);
    }
  }
  if (tactical.empty()) {
    return alpha;
  }

  if (move_sorter_ != nullptr) {
    tactical = move_sorter_->sort_tactical(board_, tactical);
  }

  for (const auto &move : tactical) {
    if (config_.use_delta_pruning && config_.use_alpha_beta) {
      const double delta_eval = stand_pat + capture_gain(move) +
                                static_cast<double>(config_.delta_margin);
      if (delta_eval < alpha) {
        stats_.qs_delta_pruning += 1;
        continue;
      }
    }

    if (config_.use_see_pruning_in_qs && move_sorter_ != nullptr &&
        board_.is_capture(move)) {
      if (move_sorter_->see(board_, move) < 0) {
        stats_.qs_see_pruning += 1;
        continue;
      }
    }

    auto saved_hash = push_move_with_hash(move);
    const double score = -quiescence(-beta, -alpha, ply + 1, qs_depth + 1);
    pop_move_with_hash(saved_hash);

    if (config_.use_alpha_beta && score >= beta) {
      return beta;
    }
    if (score > alpha) {
      alpha = score;
    }
  }

  return alpha;
}

double Minimax::null_move_search(int depth, double beta, int ply,
                                 int extensions_left) {
  std::optional<uint64_t> saved_hash;
  std::optional<uint64_t> null_hash;
  if (zobrist_ != nullptr) {
    saved_hash = zobrist_->get_current_hash();
    if (saved_hash.has_value()) {
      null_hash = zobrist_->make_null_move_hash(board_);
    }
  }

  board_.push_null();

  if (zobrist_ != nullptr) {
    if (null_hash.has_value()) {
      zobrist_->set_current_hash(null_hash);
    } else {
      (void)zobrist_->hash_board(board_);
    }
  }

  const int reduction = std::max(1, config_.nmp_reduction_r);
  const double score =
      -negamax(std::max(0, depth - 1 - reduction), -beta, -beta + 1.0, ply + 1,
               std::nullopt, extensions_left);

  (void)board_.pop();
  if (zobrist_ != nullptr && saved_hash.has_value()) {
    zobrist_->set_current_hash(saved_hash);
  }

  return score;
}

bool Minimax::has_non_pawn_material() const noexcept {
  const Color stm = board_.get_side_to_move() ? Color::WHITE : Color::BLACK;
  for (const PieceType pt : {PieceType::KNIGHT, PieceType::BISHOP,
                             PieceType::ROOK, PieceType::QUEEN}) {
    if (board_.get_piece_bb(pt, stm) != 0) {
      return true;
    }
  }
  return false;
}

int Minimax::lmr_reduction(int depth, int move_index) noexcept {
  const double base = 0.75 * std::log(std::max(2, depth)) *
                      std::log(std::max(2, move_index + 1));
  return std::max(1, std::min(3, static_cast<int>(base)));
}

} // namespace search
