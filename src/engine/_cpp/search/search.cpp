#include "search.hpp"

#include <algorithm>
#include <array>
#include <utility>
#include <vector>

namespace {

constexpr int INF = 30000;
constexpr int MATE_SCORE = 20000;
constexpr int NULL_MOVE_REDUCTION = 2;

constexpr std::array<int, 6> MATERIAL_VALUES = {
    100,  // pawn
    320,  // knight
    330,  // bishop
    500,  // rook
    900,  // queen
    20000 // king
};

constexpr int ROOT_BEST_BONUS = 1'000'000;
constexpr int CAPTURE_BASE_BONUS = 500'000;
constexpr int KILLER_1_BONUS = 30'000;
constexpr int KILLER_2_BONUS = 20'000;

[[nodiscard]] inline int evaluate(const Board &board) noexcept {
  int white_score = 0;
  int black_score = 0;

  for (int pt = 0; pt < 6; ++pt) {
    const auto piece_type = static_cast<PieceType>(pt);
    const int value = MATERIAL_VALUES[pt];
    white_score +=
        value * popcount(board.get_piece_bb(piece_type, Color::WHITE));
    black_score +=
        value * popcount(board.get_piece_bb(piece_type, Color::BLACK));
  }

  const int score = white_score - black_score;
  return board.get_side_to_move() ? score : -score;
}

} // namespace

int Search::search(int depth) {
  reset_stats();
  stats_.depth = depth;

  if (depth <= 0) {
    const int score = quiescence(-INF, INF, 0);
    stats_.score = score;
    return score;
  }

  int alpha = -INF;
  int beta = INF;

  const std::vector<Move> moves = board_.generate_legal_moves();
  if (moves.empty()) {
    stats_.score = board_.is_check() ? -MATE_SCORE : 0;
    return stats_.score;
  }

  std::vector<std::pair<int, int>> scored_moves;
  scored_moves.reserve(moves.size());
  for (size_t i = 0; i < moves.size(); ++i) {
    scored_moves.emplace_back(score_move(moves[i], 0, true),
                              static_cast<int>(i));
  }
  std::sort(scored_moves.begin(), scored_moves.end(),
            [](const auto &left, const auto &right) {
              return left.first > right.first;
            });

  int best_score = -INF;
  Move best_move{};
  int searched = 0;

  for (const auto &[_, idx] : scored_moves) {
    const Move &move = moves[idx];
    board_.push(move);

    int score;
    if (searched == 0) {
      score = -alpha_beta(depth - 1, -beta, -alpha, 1);
    } else {
      score = -alpha_beta(depth - 1, -alpha - 1, -alpha, 1);
      if (score > alpha && score < beta) {
        stats_.pvs_researches++;
        score = -alpha_beta(depth - 1, -beta, -alpha, 1);
      }
    }

    (void)board_.pop();
    ++searched;

    if (score > best_score) {
      best_score = score;
      best_move = move;
      if (score > alpha) {
        alpha = score;
        stats_.root_move_changes++;
      }
    }
  }

  stats_.score = best_score;
  stats_.best_move = best_move;
  last_best_move_ = best_move;
  has_last_best_move_ = true;
  return best_score;
}

int Search::alpha_beta(int depth, int alpha, int beta, int ply) {
  stats_.nodes++;
  if (ply > stats_.seldepth)
    stats_.seldepth = ply;

  if (ply >= MAX_PLY - 1) {
    return evaluate(board_);
  }

  if (depth <= 0) {
    return quiescence(alpha, beta, ply);
  }

  const bool in_check = board_.is_check();

  if (depth >= 3 && ply > 0 && !in_check) {
    board_.push_null();
    const int score =
        -alpha_beta(depth - 1 - NULL_MOVE_REDUCTION, -beta, -beta + 1, ply + 1);
    (void)board_.pop();
    if (score >= beta) {
      stats_.null_move_cuts++;
      return beta;
    }
  }

  const std::vector<Move> moves = board_.generate_legal_moves();
  if (moves.empty()) {
    return in_check ? -MATE_SCORE + ply : 0;
  }

  std::vector<std::pair<int, int>> scored_moves;
  scored_moves.reserve(moves.size());
  for (size_t i = 0; i < moves.size(); ++i) {
    scored_moves.emplace_back(score_move(moves[i], ply, false),
                              static_cast<int>(i));
  }
  std::sort(scored_moves.begin(), scored_moves.end(),
            [](const auto &left, const auto &right) {
              return left.first > right.first;
            });

  int moves_searched = 0;
  for (const auto &[_, idx] : scored_moves) {
    const Move &move = moves[idx];
    const bool is_capture = board_.is_capture(move);
    board_.push(move);

    int score;
    if (moves_searched == 0) {
      score = -alpha_beta(depth - 1, -beta, -alpha, ply + 1);
    } else {
      const bool do_lmr =
          !is_capture && !in_check && depth >= 3 && moves_searched >= 4;
      if (do_lmr) {
        score = -alpha_beta(depth - 2, -alpha - 1, -alpha, ply + 1);
        if (score > alpha) {
          score = -alpha_beta(depth - 1, -alpha - 1, -alpha, ply + 1);
        }
      } else {
        score = -alpha_beta(depth - 1, -alpha - 1, -alpha, ply + 1);
      }

      if (score > alpha && score < beta) {
        stats_.pvs_researches++;
        score = -alpha_beta(depth - 1, -beta, -alpha, ply + 1);
      }
    }

    (void)board_.pop();
    ++moves_searched;

    if (score >= beta) {
      stats_.beta_cutoffs++;
      if (moves_searched == 1) {
        stats_.first_move_cuts++;
      }
      if (!is_capture) {
        record_killer(ply, move);
        record_history(move, depth);
        stats_.killer_cuts++;
        stats_.history_cuts++;
      }
      return beta;
    }

    if (score > alpha) {
      alpha = score;
    }
  }

  return alpha;
}

int Search::quiescence(int alpha, int beta, int ply) {
  stats_.qsearch_nodes++;
  if (ply > stats_.seldepth)
    stats_.seldepth = ply;

  if (ply >= MAX_PLY - 1) {
    return evaluate(board_);
  }

  const int stand_pat = evaluate(board_);
  if (stand_pat >= beta)
    return beta;
  if (stand_pat > alpha)
    alpha = stand_pat;

  const std::vector<Move> moves = board_.generate_legal_moves();
  std::vector<std::pair<int, int>> captures;
  captures.reserve(moves.size());
  for (size_t i = 0; i < moves.size(); ++i) {
    if (board_.is_capture(moves[i])) {
      captures.emplace_back(score_capture(moves[i]), static_cast<int>(i));
    }
  }

  std::sort(captures.begin(), captures.end(),
            [](const auto &left, const auto &right) {
              return left.first > right.first;
            });

  for (const auto &[_, idx] : captures) {
    const Move &move = moves[idx];
    board_.push(move);
    const int score = -quiescence(-beta, -alpha, ply + 1);
    (void)board_.pop();

    if (score >= beta) {
      return beta;
    }
    if (score > alpha) {
      alpha = score;
    }
  }

  return alpha;
}

int Search::score_capture(const Move &move) const noexcept {
  auto attacker = board_.piece_at(move.from);
  if (!attacker)
    return 0;

  int victim_value = MATERIAL_VALUES[0];
  auto victim = board_.piece_at(move.to);
  if (victim) {
    victim_value = MATERIAL_VALUES[static_cast<int>(victim->type)];
  } else if (board_.is_en_passant(move)) {
    victim_value = MATERIAL_VALUES[static_cast<int>(PieceType::PAWN)];
  }

  const int attacker_value = MATERIAL_VALUES[static_cast<int>(attacker->type)];
  return (victim_value << 10) - attacker_value;
}

int Search::score_move(const Move &move, int ply, bool root) const noexcept {
  if (root && has_last_best_move_ && moves_equal(move, last_best_move_)) {
    return ROOT_BEST_BONUS;
  }

  if (board_.is_capture(move)) {
    return CAPTURE_BASE_BONUS + score_capture(move);
  }

  if (ply < MAX_PLY) {
    const Move &killer0 = killer_moves_[ply][0];
    const Move &killer1 = killer_moves_[ply][1];
    if (moves_equal(move, killer0))
      return KILLER_1_BONUS;
    if (moves_equal(move, killer1))
      return KILLER_2_BONUS;
  }

  const int side = board_.get_side_to_move() ? 0 : 1;
  return history_[side][move.from][move.to];
}

void Search::record_killer(int ply, const Move &move) noexcept {
  if (ply < 0 || ply >= MAX_PLY)
    return;
  Move &killer0 = killer_moves_[ply][0];
  Move &killer1 = killer_moves_[ply][1];
  if (moves_equal(move, killer0))
    return;
  killer1 = killer0;
  killer0 = move;
}

void Search::record_history(const Move &move, int depth) noexcept {
  const int side = board_.get_side_to_move() ? 0 : 1;
  int &slot = history_[side][move.from][move.to];
  const int bonus = depth * depth;
  slot += bonus;
  if (slot > 1'000'000) {
    slot >>= 1;
  }
}
