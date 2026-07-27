#pragma once
// ---------------------------------------------------------------------------
// evaluators.hpp — bitboard-driven position evaluation
//
// Phase-2 rewrite: the CompositeEvaluator computes the game phase exactly
// once per call and forwards it through `score(board, phase)`.  Components
// walk bitboards directly rather than iterating all 64 mailbox squares.
// ---------------------------------------------------------------------------

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include "../board/board.hpp"

namespace evaluators {

[[nodiscard]] double compute_game_phase(const Board &board) noexcept;

extern const std::array<std::array<double, 64>, 6> PST_MG;
extern const std::array<std::array<double, 64>, 6> PST_EG;

class IEvaluator {
public:
  virtual ~IEvaluator() = default;
  [[nodiscard]] virtual double go(const Board &board) const = 0;

  // Optional fast path — components receive the precomputed phase so a
  // composite can compute it once per node.  Default implementation
  // delegates to go(board).
  [[nodiscard]] virtual double score(const Board &board,
                                     double /*phase*/) const {
    return go(board);
  }
};

class MaterialComponent : public IEvaluator {
public:
  [[nodiscard]] double score(const Board &board,
                             double phase) const noexcept override;
  [[nodiscard]] double go(const Board &board) const override;
};

class PSTComponent : public IEvaluator {
public:
  explicit PSTComponent(bool gsc) noexcept : gsc_(gsc) {}
  [[nodiscard]] double score(const Board &board,
                             double phase) const noexcept override;
  [[nodiscard]] double go(const Board &board) const override;

private:
  bool gsc_;
};

class PawnStructureComponent : public IEvaluator {
public:
  explicit PawnStructureComponent(bool gsc) noexcept : gsc_(gsc) {}
  [[nodiscard]] double score(const Board &board,
                             double phase) const noexcept override;
  [[nodiscard]] double go(const Board &board) const override;

private:
  bool gsc_;
};

class MobilityComponent : public IEvaluator {
public:
  explicit MobilityComponent(bool gsc) noexcept : gsc_(gsc) {}
  [[nodiscard]] double score(const Board &board, double phase) const override;
  [[nodiscard]] double go(const Board &board) const override;

private:
  bool gsc_;
};

class KingSafetyComponent : public IEvaluator {
public:
  explicit KingSafetyComponent(bool gsc) noexcept : gsc_(gsc) {}
  [[nodiscard]] double score(const Board &board,
                             double phase) const noexcept override;
  [[nodiscard]] double go(const Board &board) const override;

private:
  bool gsc_;
};

// Composite evaluator — phase is computed once per go() and forwarded via
// score(board, phase), so components don't each recompute it.
class CompositeEvaluator : public IEvaluator {
public:
  CompositeEvaluator() = default;
  explicit CompositeEvaluator(
      std::vector<std::shared_ptr<IEvaluator>> components)
      : components_(std::move(components)) {}

  [[nodiscard]] double go(const Board &board) const override;

  [[nodiscard]] const std::vector<std::shared_ptr<IEvaluator>> &
  components() const noexcept {
    return components_;
  }

  void add_component(std::shared_ptr<IEvaluator> component) {
    components_.push_back(std::move(component));
  }

private:
  std::vector<std::shared_ptr<IEvaluator>> components_;
};

} // namespace evaluators
