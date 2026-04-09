#pragma once

#include <cstdint>
#include <cstring>
#include <optional>
#include <vector>

#include "../board/board.hpp"

namespace search {

enum class BoundType : uint8_t { EXACT = 0, LOWER = 1, UPPER = 2 };

// Cache-line friendly entry: padded to 32 bytes via alignas.
struct alignas(32) TTEntry {
  uint64_t key = 0;  // Zobrist hash (full key for verification)
  int32_t score = 0; // Centipawn score
  Move best_move{};  // 3 bytes: from, to, promotion
  int16_t depth = 0; // Search depth
  uint8_t bound = 0; // BoundType cast to uint8_t
  uint8_t age = 0;   // Search age for replacement
  // Total: 8 + 4 + 3 + 2 + 1 + 1 = 19 bytes, padded to 32
};

class TranspositionTable {
public:
  static constexpr size_t MIN_ENTRIES = 1024;

  explicit TranspositionTable(size_t size_mb = 64) noexcept;

  void resize(size_t size_mb) noexcept;
  void clear() noexcept;
  void increment_age() noexcept;

  // Prefetch the entry for the given key (call before probe for latency
  // hiding).
  void prefetch(uint64_t key) const noexcept {
    __builtin_prefetch(&table_[key & mask_], 0, 1);
  }

  // Probe: returns pointer to entry if key matches, nullptr otherwise
  [[nodiscard]] const TTEntry *probe(uint64_t key) const noexcept;

  // Mutable probe used internally by store to refresh age
  [[nodiscard]] TTEntry *probe_mut(uint64_t key) noexcept;

  // Try to get a usable score from a TT entry
  [[nodiscard]] std::optional<int32_t> try_get_score(const TTEntry &entry,
                                                     int depth, int alpha,
                                                     int beta) const noexcept;

  // Store with depth-preferred replacement
  void store(uint64_t key, int depth, int32_t score, const Move &best_move,
             BoundType bound) noexcept;

  [[nodiscard]] size_t size() const noexcept;
  [[nodiscard]] size_t capacity() const noexcept;
  [[nodiscard]] int hashfull() const noexcept;
  [[nodiscard]] uint8_t current_age() const noexcept { return age_; }

private:
  std::vector<TTEntry> table_;
  size_t mask_ = 0; // size - 1, for fast modulo via bitwise AND
  uint8_t age_ = 0;
  size_t entry_count_ = 0; // approximate count of non-empty entries
};

} // namespace search
