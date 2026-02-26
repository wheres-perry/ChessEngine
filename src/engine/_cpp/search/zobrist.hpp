#pragma once
// ---------------------------------------------------------------------------
// zobrist.hpp — High-performance Zobrist hashing for chess search
//
// Uses compile-time generated SplitMix64 keys (zobrist_keys.hpp) instead of
// PolyGlot.  Optimised for:
//   • Cache-aligned key tables (fits L1)
//   • Single-lookup castling hash (16-entry table vs 4 conditional XORs)
//   • No EP pawn-can-capture validation (engine responsibility)
//   • O(1) null-move hashing via make_null_move_hash()
//   • [[likely]]/[[unlikely]] branch hints on cold fallback paths
// ---------------------------------------------------------------------------

#include <cstdint>
#include <optional>

#include "../board/board.hpp"

namespace search {

class Zobrist {
public:
  /// Construct Zobrist hasher.  @p seed is accepted for API compatibility
  /// but ignored — all keys are compile-time constants.
  explicit Zobrist(std::optional<uint64_t> seed = std::nullopt) noexcept;

  /// Compute full positional hash from scratch — O(popcount) via bitboard
  /// scan.  XORs side key when white to move.  Stores result internally.
  [[nodiscard]] uint64_t hash_board(const Board &board) noexcept;

  /// Incremental hash update for a normal move — O(1).
  /// Board must be in *pre-move* state.  Returns the hash of the position
  /// that would result from applying @p move.  Does NOT mutate the board.
  [[nodiscard]] uint64_t make_move_hash(Board &board,
                                        const Move &move) noexcept;

  /// Incremental hash update for a null move — O(1).
  /// Board must be in *pre-move* state.  Only toggles side-to-move and
  /// removes the old en-passant contribution.  Does NOT mutate the board.
  [[nodiscard]] uint64_t make_null_move_hash(const Board &board) noexcept;

  /// Get current internally stored hash (std::nullopt if uninitialised).
  [[nodiscard]] std::optional<uint64_t> get_current_hash() const noexcept;

  /// Overwrite the internally stored hash.
  void set_current_hash(std::optional<uint64_t> hash_val) noexcept;

  /// Clear the stored hash (forces full recomputation on next access).
  void invalidate_hash() noexcept;

private:
  std::optional<uint64_t> current_hash_;
};

} // namespace search
