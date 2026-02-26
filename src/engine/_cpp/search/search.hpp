#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <vector>

#include "../board/board.hpp"

struct SearchStats {
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
  uint64_t root_move_changes = 0;

  // Stability
  int score = 0;
  Move best_move{};
};

class Search {
public:
  explicit Search(Board &board) noexcept : board_(board) {}

  // Basic fixed depth search
  // Returns evaluation score (centipawns)
  [[nodiscard]] int search(int depth);

  // Get statistics from the last search
  [[nodiscard]] const SearchStats &get_stats() const noexcept { return stats_; }

  // Reset stats
  void reset_stats() noexcept { stats_ = SearchStats{}; }

private:
  static constexpr int MAX_PLY = 128;

  Board &board_;
  SearchStats stats_;

  std::array<std::array<Move, 2>, MAX_PLY> killer_moves_{};
  std::array<std::array<std::array<int, 64>, 64>, 2> history_{};
  Move last_best_move_{};
  bool has_last_best_move_ = false;

  // Search functions
  [[nodiscard]] int alpha_beta(int depth, int alpha, int beta, int ply);
  [[nodiscard]] int quiescence(int alpha, int beta, int ply);

  // Move ordering and heuristics
  [[nodiscard]] static constexpr bool moves_equal(const Move &a,
                                                  const Move &b) noexcept {
    return a.from == b.from && a.to == b.to && a.promotion == b.promotion;
  }
  [[nodiscard]] int score_move(const Move &move, int ply,
                               bool root) const noexcept;
  [[nodiscard]] int score_capture(const Move &move) const noexcept;
  void record_killer(int ply, const Move &move) noexcept;
  void record_history(const Move &move, int depth) noexcept;
};
