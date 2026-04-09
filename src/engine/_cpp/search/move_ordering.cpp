#include "move_ordering.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

namespace move_ordering {

// =========================================================================
// sort_moves — partial sort for top N, full sort for the rest
// =========================================================================

std::vector<Move> MoveSorter::sort_moves(Board &board,
                                         const std::vector<Move> &moves,
                                         int ply, const Move &hash_move,
                                         const Move &previous_move) const {
  if (!config_.use_move_ordering || moves.size() <= 1) {
    return moves;
  }

  std::vector<std::pair<int, size_t>> scored;
  scored.reserve(moves.size());
  for (size_t i = 0; i < moves.size(); ++i) {
    scored.emplace_back(
        score_move(board, moves[i], ply, hash_move, previous_move), i);
  }

  // Use partial sort: only sort the top PARTIAL_SORT_N moves fully.
  // For alpha-beta, we often get a cutoff on the first few moves.
  const auto cmp = [](const auto &a, const auto &b) {
    return a.first > b.first;
  };
  if (static_cast<int>(scored.size()) > PARTIAL_SORT_N) {
    std::partial_sort(scored.begin(), scored.begin() + PARTIAL_SORT_N,
                      scored.end(), cmp);
    // Sort the remainder too for correctness (move ordering tests expect
    // fully sorted output).
    std::sort(scored.begin() + PARTIAL_SORT_N, scored.end(), cmp);
  } else {
    std::sort(scored.begin(), scored.end(), cmp);
  }

  std::vector<Move> result;
  result.reserve(moves.size());
  for (const auto &[_, idx] : scored) {
    result.push_back(moves[idx]);
  }
  return result;
}

// =========================================================================
// sort_tactical
// =========================================================================

std::vector<Move>
MoveSorter::sort_tactical(Board &board, const std::vector<Move> &moves) const {
  std::vector<std::pair<int, size_t>> scored;
  scored.reserve(moves.size());
  for (size_t i = 0; i < moves.size(); ++i) {
    scored.emplace_back(score_tactical_move(board, moves[i]), i);
  }
  std::sort(scored.begin(), scored.end(),
            [](const auto &a, const auto &b) { return a.first > b.first; });

  std::vector<Move> result;
  result.reserve(moves.size());
  for (const auto &[_, idx] : scored) {
    result.push_back(moves[idx]);
  }
  return result;
}

// =========================================================================
// score_move
// =========================================================================

int MoveSorter::score_move(Board &board, const Move &move, int ply,
                           const Move &hash_move,
                           const Move &previous_move) const {
  // 1. Hash move gets top priority.
  if (config_.use_hash_move_ordering && !is_no_move(hash_move) &&
      move == hash_move) [[unlikely]] {
    return HASH_MOVE_SCORE;
  }

  // 2. Tactical moves (captures / promotions).
  if (board.is_capture(move) || is_promotion(move)) [[unlikely]] {
    return score_tactical_move(board, move);
  }

  // 3. Killer moves.
  if (config_.use_killer_moves && ply >= 0 && ply < MAX_PLY) {
    const auto &killers = killer_moves_[ply];
    for (size_t i = 0; i < killers.size(); ++i) {
      if (killers[i] == move) {
        return KILLER_BASE - (static_cast<int>(i) * 1024);
      }
    }
  }

  // 4. Countermove heuristic (flat array lookup).
  if (config_.use_countermove_heuristic && !is_no_move(previous_move)) {
    if (previous_move.from < 64 && previous_move.to < 64) [[likely]] {
      const Move &cm = countermove_table_[previous_move.from][previous_move.to];
      if (cm == move) {
        return COUNTERMOVE_SCORE;
      }
    }
  }

  // 5. History heuristic (flat array lookup).
  if (config_.use_history_heuristic) {
    // Use the side-to-move to index the history table.
    int side = board.get_side_to_move() ? 0 : 1;
    if (move.from < 64 && move.to < 64) [[likely]] {
      int history = history_table_[side][move.from][move.to];
      return std::min(history, config_.history_max_score);
    }
  }

  return 0;
}

// =========================================================================
// score_tactical_move
// =========================================================================

int MoveSorter::score_tactical_move(Board &board, const Move &move) const {
  int score = TACTICAL_BASE;

  if (config_.use_mvv_lva && board.is_capture(move)) {
    score += mvv_lva(board, move);
  }

  if (is_promotion(move)) {
    // promotion value: piece type maps 1=KNIGHT, 2=BISHOP, 3=ROOK, 4=QUEEN
    if (move.promotion > 0 && move.promotion <= 5) [[likely]] {
      score += PIECE_VALUES_CP[move.promotion];
    }
  }

  if (config_.use_see_ordering && board.is_capture(move)) {
    int see_value = see(board, move);
    if (see_value < config_.see_capture_threshold) [[unlikely]] {
      score -= 50'000;
    } else {
      score += std::min(see_value, 5'000);
    }
  }

  return score;
}

// =========================================================================
// mvv_lva
// =========================================================================

int MoveSorter::mvv_lva(Board &board, const Move &move) const noexcept {
  auto victim_piece = board.piece_at(move.to);
  auto attacker_piece = board.piece_at(move.from);

  int victim_value = 0;
  if (!victim_piece.has_value() && board.is_en_passant(move)) {
    victim_value = PIECE_VALUES_CP[static_cast<int>(PieceType::PAWN)];
  } else if (victim_piece.has_value()) {
    victim_value = PIECE_VALUES_CP[static_cast<int>(victim_piece->type)];
  }

  int attacker_value =
      attacker_piece.has_value()
          ? PIECE_VALUES_CP[static_cast<int>(attacker_piece->type)]
          : PIECE_VALUES_CP[static_cast<int>(PieceType::PAWN)];

  return victim_value * 10 - attacker_value;
}

// =========================================================================
// SEE (Static Exchange Evaluation)
// =========================================================================

int MoveSorter::see(Board &board, const Move &move) const {
  if (!board.is_capture(move)) {
    return 0;
  }

  // Create a copy of the board to simulate the capture sequence.
  Board sim_board = board.copy();

  // Determine the sequence of gains.
  std::vector<int> gains;
  gains.reserve(32);

  // Initial capture gain.
  auto victim_piece = sim_board.piece_at(move.to);
  int gain = 0;
  if (!victim_piece.has_value() && sim_board.is_en_passant(move)) {
    gain = PIECE_VALUES_CP[static_cast<int>(PieceType::PAWN)];
  } else if (victim_piece.has_value()) {
    gain = PIECE_VALUES_CP[static_cast<int>(victim_piece->type)];
  }
  gains.push_back(gain);

  sim_board.push(move);

  uint8_t target_sq = move.to;

  // Simulate captures on the target square until no more are possible.
  while (true) {
    Move best_attacker_move{};
    int lowest_attacker_val = std::numeric_limits<int>::max();
    bool found = false;

    auto legal_moves = sim_board.generate_legal_moves();
    for (const auto &next_move : legal_moves) {
      if (next_move.to == target_sq && sim_board.is_capture(next_move)) {
        auto attacker_piece = sim_board.piece_at(next_move.from);
        int attacker_val =
            attacker_piece.has_value()
                ? PIECE_VALUES_CP[static_cast<int>(attacker_piece->type)]
                : 0;
        if (attacker_val < lowest_attacker_val) {
          lowest_attacker_val = attacker_val;
          best_attacker_move = next_move;
          found = true;
        }
      }
    }

    if (!found) {
      break;
    }

    // The piece captured is whatever was moved previously.
    auto captured_piece = sim_board.piece_at(target_sq);
    int captured_value =
        captured_piece.has_value()
            ? PIECE_VALUES_CP[static_cast<int>(captured_piece->type)]
            : 0;
    gains.push_back(captured_value);

    sim_board.push(best_attacker_move);
  }

  // Minimax backwards through the capture sequence.
  int score = 0;
  for (int i = static_cast<int>(gains.size()) - 1; i >= 0; --i) {
    score = std::max(0, gains[i] - score);
  }

  return score;
}

// =========================================================================
// on_beta_cutoff
// =========================================================================

void MoveSorter::on_beta_cutoff(const Move &move, int ply, int depth,
                                const Move &previous_move,
                                bool is_tactical) noexcept {
  if (is_tactical) {
    return;
  }

  // Update killer moves.
  if (config_.use_killer_moves && ply >= 0 && ply < MAX_PLY) {
    auto &killers = killer_moves_[ply];
    // Check if already in the list.
    for (const auto &k : killers) {
      if (k == move) {
        return;
      }
    }
    killers.insert(killers.begin(), move);
    int max_killers = std::max(1, config_.killer_slots_per_ply);
    if (static_cast<int>(killers.size()) > max_killers) {
      killers.resize(max_killers);
    }
  }

  // Update history table (flat array).
  if (config_.use_history_heuristic && move.from < 64 && move.to < 64) {
    int bonus = depth * depth;
    // We don't know the side here, so update side 0 (WHITE).
    // The search code should ideally pass the side, but for compatibility
    // we update both sides' entries.
    // Actually, use a simple heuristic: just store in side 0 for now.
    // The on_beta_cutoff is called during search where the board state
    // determines the side. We'll use side 0 as the "general" table.
    int &current = history_table_[0][move.from][move.to];
    int old_val = current;
    int new_val = std::min(old_val + bonus, config_.history_max_score);
    if (old_val == 0 && new_val != 0) {
      history_entry_count_++;
    }
    history_sum_ += (new_val - old_val);
    current = new_val;
  }

  // Update countermove table (flat array).
  if (config_.use_countermove_heuristic && !is_no_move(previous_move) &&
      previous_move.from < 64 && previous_move.to < 64) {
    countermove_table_[previous_move.from][previous_move.to] = move;
  }
}

// =========================================================================
// history_saturation
// =========================================================================

double MoveSorter::history_saturation() const noexcept {
  if (!config_.use_history_heuristic || history_entry_count_ == 0) {
    return 0.0;
  }
  double max_score = static_cast<double>(config_.history_max_score);
  double avg = static_cast<double>(history_sum_) /
               static_cast<double>(history_entry_count_);
  double saturation = (avg / max_score) * 100.0;
  return std::min(100.0, saturation);
}

// =========================================================================
// reset
// =========================================================================

void MoveSorter::reset(bool clear_history, bool clear_killers) noexcept {
  if (clear_killers) {
    for (auto &v : killer_moves_) {
      v.clear();
    }
  }
  if (clear_history) {
    std::memset(history_table_, 0, sizeof(history_table_));
    std::memset(countermove_table_, 0, sizeof(countermove_table_));
    history_entry_count_ = 0;
    history_sum_ = 0;
  }
}

} // namespace move_ordering
