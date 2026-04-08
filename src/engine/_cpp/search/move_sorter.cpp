#include "move_sorter.hpp"

#include <algorithm>
#include <limits>

namespace search {

namespace {

constexpr int PAWN_VALUE = MoveSorter::PIECE_VALUES_CP[0];

[[nodiscard]] inline int piece_value(PieceType pt) noexcept {
  return MoveSorter::PIECE_VALUES_CP[static_cast<int>(pt)];
}

[[nodiscard]] inline int victim_value(const Board &board,
                                      const Move &move) noexcept {
  auto victim = board.piece_at(move.to);
  if (!victim) {
    if (board.is_en_passant(move)) {
      return PAWN_VALUE;
    }
    return 0;
  }
  return piece_value(victim->type);
}

[[nodiscard]] inline int attacker_value(const Board &board,
                                        const Move &move) noexcept {
  auto attacker = board.piece_at(move.from);
  if (!attacker) {
    return PAWN_VALUE;
  }
  return piece_value(attacker->type);
}

// Small stack buffer for the scored-move pairs so typical nodes never hit
// the heap.  256 covers the absolute legal-move maximum in chess.
constexpr size_t STACK_MOVE_BUF = 256;

} // namespace

MoveSorter::MoveSorter(const CppSearchConfig &config) noexcept
    : config_(config) {
  killer_counts_.fill(0);
  history_table_.fill(0);
  history_present_.fill(false);
  countermove_present_.fill(false);
}

void MoveSorter::reset(bool clear_history, bool clear_killers) noexcept {
  if (clear_killers) {
    killer_counts_.fill(0);
  }
  if (clear_history) {
    history_table_.fill(0);
    history_present_.fill(false);
    countermove_present_.fill(false);
  }
}

std::vector<Move>
MoveSorter::sort_moves(Board &board, const std::vector<Move> &moves, int ply,
                       std::optional<Move> hash_move,
                       std::optional<Move> previous_move) const {
  if (!config_.use_move_ordering || moves.size() <= 1) {
    return moves;
  }

  const size_t n = moves.size();
  // Stack buffers — avoids an allocation on the hot path.
  int scores[STACK_MOVE_BUF];
  int indices[STACK_MOVE_BUF];
  const size_t limit = std::min(n, STACK_MOVE_BUF);

  for (size_t i = 0; i < limit; ++i) {
    scores[i] = score_move(board, moves[i], ply, hash_move, previous_move);
    indices[i] = static_cast<int>(i);
  }

  std::sort(indices, indices + limit,
            [&](int a, int b) { return scores[a] > scores[b]; });

  std::vector<Move> out;
  out.reserve(n);
  for (size_t i = 0; i < limit; ++i) {
    out.push_back(moves[indices[i]]);
  }
  // Extremely rare overflow path for >256 legal moves (impossible in real
  // chess but kept for safety).
  if (MS_UNLIKELY(n > STACK_MOVE_BUF)) {
    for (size_t i = STACK_MOVE_BUF; i < n; ++i) {
      out.push_back(moves[i]);
    }
  }
  return out;
}

std::vector<Move>
MoveSorter::sort_tactical(Board &board, const std::vector<Move> &moves) const {
  const size_t n = moves.size();
  if (n == 0) {
    return {};
  }
  int scores[STACK_MOVE_BUF];
  int indices[STACK_MOVE_BUF];
  const size_t limit = std::min(n, STACK_MOVE_BUF);
  for (size_t i = 0; i < limit; ++i) {
    scores[i] = score_tactical_move(board, moves[i]);
    indices[i] = static_cast<int>(i);
  }
  std::sort(indices, indices + limit,
            [&](int a, int b) { return scores[a] > scores[b]; });

  std::vector<Move> out;
  out.reserve(n);
  for (size_t i = 0; i < limit; ++i) {
    out.push_back(moves[indices[i]]);
  }
  if (MS_UNLIKELY(n > STACK_MOVE_BUF)) {
    for (size_t i = STACK_MOVE_BUF; i < n; ++i) {
      out.push_back(moves[i]);
    }
  }
  return out;
}

int MoveSorter::score_move(Board &board, const Move &move, int ply,
                           std::optional<Move> hash_move,
                           std::optional<Move> previous_move) const noexcept {
  if (MS_UNLIKELY(config_.use_hash_move_ordering && hash_move.has_value() &&
                  move == *hash_move)) {
    return HASH_MOVE_SCORE;
  }

  if (board.is_capture(move) || is_promotion(move)) {
    return score_tactical_move(board, move);
  }

  if (config_.use_killer_moves && ply >= 0 && ply < MAX_PLY) {
    const int count = killer_counts_[ply];
    for (int slot = 0; slot < count; ++slot) {
      if (killer_moves_[ply][slot] == move) {
        return KILLER_BASE - (slot * 1024);
      }
    }
  }

  if (config_.use_countermove_heuristic && previous_move.has_value()) {
    const auto cm = countermove_get(previous_move->from, previous_move->to,
                                    previous_move->promotion);
    if (cm.has_value() && *cm == move) {
      return COUNTERMOVE_SCORE;
    }
  }

  if (config_.use_history_heuristic) {
    const int hist = history_get(move.from, move.to, move.promotion);
    return std::min(hist, config_.history_max_score);
  }

  return 0;
}

int MoveSorter::score_tactical_move(Board &board,
                                    const Move &move) const noexcept {
  int score = TACTICAL_BASE;
  const bool is_cap = board.is_capture(move);
  if (config_.use_mvv_lva && is_cap) {
    score += mvv_lva(board, move);
  }
  if (is_promotion(move)) {
    const int promo_idx = static_cast<int>(move.promotion);
    if (MS_LIKELY(promo_idx >= 0 && promo_idx < 6)) {
      score += PIECE_VALUES_CP[promo_idx];
    }
  }
  if (config_.use_see_ordering && is_cap) {
    const int see_value = see(board, move);
    if (see_value < config_.see_capture_threshold) {
      score -= 50'000;
    } else {
      score += std::min(see_value, 5'000);
    }
  }
  return score;
}

int MoveSorter::mvv_lva(const Board &board, const Move &move) const noexcept {
  const int victim = victim_value(board, move);
  const int attacker = attacker_value(board, move);
  return victim * 10 - attacker;
}

int MoveSorter::see(Board &board, const Move &move) const {
  if (!board.is_capture(move)) {
    return 0;
  }

  Board sim = board;

  int gains[32];
  int gain_count = 0;

  int gain = 0;
  auto victim_piece = sim.piece_at(move.to);
  if (!victim_piece && sim.is_en_passant(move)) {
    gain = PAWN_VALUE;
  } else if (victim_piece) {
    gain = piece_value(victim_piece->type);
  }
  gains[gain_count++] = gain;

  sim.push(move);
  const uint8_t target_sq = move.to;

  while (gain_count < 32) {
    Move best_attacker_move{};
    bool has_best = false;
    int lowest_attacker = std::numeric_limits<int>::max();

    for (const auto &next_move : sim.generate_legal_moves()) {
      if (next_move.to != target_sq) {
        continue;
      }
      if (!sim.is_capture(next_move)) {
        continue;
      }
      auto attacker = sim.piece_at(next_move.from);
      const int atk_val = attacker ? piece_value(attacker->type) : 0;
      if (atk_val < lowest_attacker) {
        lowest_attacker = atk_val;
        best_attacker_move = next_move;
        has_best = true;
      }
    }

    if (!has_best) {
      break;
    }

    auto captured = sim.piece_at(target_sq);
    const int capture_gain = captured ? piece_value(captured->type) : 0;
    gains[gain_count++] = capture_gain;

    sim.push(best_attacker_move);
  }

  int score = 0;
  for (int i = gain_count - 1; i >= 0; --i) {
    score = std::max(0, gains[i] - score);
  }
  return score;
}

void MoveSorter::on_beta_cutoff(const Move &move, int ply, int depth,
                                std::optional<Move> previous_move,
                                bool is_tactical) noexcept {
  if (is_tactical) {
    return;
  }

  if (config_.use_killer_moves && ply >= 0 && ply < MAX_PLY) {
    const int count = killer_counts_[ply];
    for (int i = 0; i < count; ++i) {
      if (killer_moves_[ply][i] == move) {
        // Mirror Python semantics: if the move is already a killer at this
        // ply, bail out entirely — no history/countermove update.
        return;
      }
    }
    const int max_killers = std::max(1, config_.killer_slots_per_ply);
    const int capacity = std::min(MAX_KILLER_SLOTS, max_killers);
    const int new_count = std::min(count + 1, capacity);
    // Shift existing killers right by one slot, then write `move` to slot 0.
    for (int i = new_count - 1; i > 0; --i) {
      killer_moves_[ply][i] = killer_moves_[ply][i - 1];
    }
    killer_moves_[ply][0] = move;
    killer_counts_[ply] = new_count;
  }

  if (config_.use_history_heuristic) {
    const int idx = history_index(move.from, move.to, move.promotion);
    const int bonus = depth * depth;
    int current = history_table_[idx];
    current += bonus;
    if (current > config_.history_max_score) {
      current = config_.history_max_score;
    }
    history_table_[idx] = current;
    history_present_[idx] = true;
  }

  if (config_.use_countermove_heuristic && previous_move.has_value()) {
    const int idx = history_index(previous_move->from, previous_move->to,
                                  previous_move->promotion);
    countermove_table_[idx] = move;
    countermove_present_[idx] = true;
  }
}

double MoveSorter::history_saturation() const noexcept {
  if (!config_.use_history_heuristic) {
    return 0.0;
  }
  int64_t total = 0;
  int count = 0;
  for (int i = 0; i < HISTORY_TABLE_SIZE; ++i) {
    if (history_present_[i]) {
      total += history_table_[i];
      ++count;
    }
  }
  if (count == 0) {
    return 0.0;
  }
  const double max_score = static_cast<double>(config_.history_max_score);
  const double avg = static_cast<double>(total) / static_cast<double>(count);
  const double saturation = (avg / max_score) * 100.0;
  return std::min(100.0, saturation);
}

void MoveSorter::history_for_each(
    const std::function<void(uint8_t, uint8_t, uint8_t, int)> &fn) const {
  for (int from = 0; from < 64; ++from) {
    for (int to = 0; to < 64; ++to) {
      for (int promo = 0; promo < PROMO_SLOTS; ++promo) {
        const int idx =
            history_index(static_cast<uint8_t>(from), static_cast<uint8_t>(to),
                          static_cast<uint8_t>(promo));
        if (history_present_[idx]) {
          fn(static_cast<uint8_t>(from), static_cast<uint8_t>(to),
             static_cast<uint8_t>(promo), history_table_[idx]);
        }
      }
    }
  }
}

void MoveSorter::countermove_for_each(
    const std::function<void(uint8_t, uint8_t, uint8_t, const Move &)> &fn)
    const {
  for (int from = 0; from < 64; ++from) {
    for (int to = 0; to < 64; ++to) {
      for (int promo = 0; promo < PROMO_SLOTS; ++promo) {
        const int idx =
            history_index(static_cast<uint8_t>(from), static_cast<uint8_t>(to),
                          static_cast<uint8_t>(promo));
        if (countermove_present_[idx]) {
          fn(static_cast<uint8_t>(from), static_cast<uint8_t>(to),
             static_cast<uint8_t>(promo), countermove_table_[idx]);
        }
      }
    }
  }
}

} // namespace search
