#pragma once

#include "../board/board.hpp"
#include <cstdint>

namespace eval {

struct EvalConfig {
  bool use_pst = true;
  bool use_pawn_structure = true;
  bool use_mobility = true;
  bool use_king_safety = true;
  bool game_stage_conscious = true;
};

class Evaluator {
public:
  explicit Evaluator(const EvalConfig &config) noexcept;

  /// Returns centipawn score from White's perspective.
  /// Board is non-const because mobility evaluation uses push_null/pop.
  [[nodiscard]] int go(Board &board) const noexcept;

  /// Get the config (read-only).
  [[nodiscard]] const EvalConfig &config() const noexcept { return config_; }

private:
  EvalConfig config_;

  [[nodiscard]] double compute_game_phase(const Board &board) const noexcept;
  [[nodiscard]] int material_score(const Board &board) const noexcept;
  [[nodiscard]] double pst_score(const Board &board,
                                 double phase) const noexcept;
  [[nodiscard]] double pawn_structure_score(const Board &board,
                                            double phase) const noexcept;
  [[nodiscard]] double mobility_score(Board &board,
                                      double phase) const noexcept;
  [[nodiscard]] double king_safety_score(const Board &board,
                                         double phase) const noexcept;
};

} // namespace eval
