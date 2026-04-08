#include "transposition_table.hpp"

#include <algorithm>
#include <cstdlib>
#include <limits>

namespace search {

namespace {

constexpr int MIN_ENTRIES = 1024;

// Next power of two ≥ value, clamped to at least `floor_value`.
[[nodiscard]] constexpr size_t round_up_pow2(int64_t value,
                                             int64_t floor_value) noexcept {
  int64_t v = std::max<int64_t>(value, floor_value);
  size_t result = 1;
  while (static_cast<int64_t>(result) < v) {
    result <<= 1;
  }
  return result;
}

// calloc yields zero-initialized pages that Linux lazily maps on first
// touch, so constructing a 64 MB table is essentially free up front.  The
// TTBound::EMPTY enum is deliberately 0 so zeroed memory counts as empty.
[[nodiscard]] TTEntry *allocate_calloc(size_t capacity) noexcept {
  return static_cast<TTEntry *>(std::calloc(capacity, sizeof(TTEntry)));
}

} // namespace

TranspositionTable::TranspositionTable(const CppSearchConfig &config)
    : config_(config) {
  const int64_t estimated = static_cast<int64_t>(config_.tt_size_mb) * 1024 *
                            1024 / ESTIMATED_ENTRY_SIZE_BYTES;
  const size_t capacity = round_up_pow2(estimated, MIN_ENTRIES);
  table_.reset(allocate_calloc(capacity));
  capacity_ = capacity;
  mask_ = capacity - 1;
  max_entries_ = static_cast<int>(capacity);
}

void TranspositionTable::set_max_entries(int value) noexcept {
  max_entries_ = value;
}

void TranspositionTable::store(uint64_t key, int depth, double score,
                               std::optional<Move> best_move,
                               TTBound bound) noexcept {
  TTEntry &slot = table_[bucket_index(key)];
  if (TT_UNLIKELY(slot.is_occupied() && slot.key == key)) {
    // Same-key replacement: keep the deeper entry within the current
    // generation; stale generations are always overwritten when aging is
    // enabled.
    if (config_.use_tt_aging) {
      if (slot.age == static_cast<uint16_t>(current_age_) &&
          slot.depth > depth) {
        return;
      }
    } else if (slot.depth > depth) {
      return;
    }
  } else if (!slot.is_occupied()) {
    // Target bucket is free.  If we've hit the (possibly test-set) logical
    // cap, evict the oldest entry first to make room.  This scan is cold:
    // in production the physical table is always sized so `size_` never
    // reaches `max_entries_`.
    if (TT_UNLIKELY(static_cast<int>(size_) >= max_entries_)) {
      uint16_t min_age = std::numeric_limits<uint16_t>::max();
      TTEntry *oldest = nullptr;
      for (size_t i = 0; i < capacity_; ++i) {
        TTEntry &entry = table_[i];
        if (entry.is_occupied() && entry.age < min_age) {
          min_age = entry.age;
          oldest = &entry;
        }
      }
      if (oldest != nullptr) {
        oldest->bound = TTBound::EMPTY;
        oldest->has_best_move = false;
        --size_;
      }
    }
    ++size_;
  }
  // Different-key collisions fall through without touching size_ — we
  // simply overwrite the existing slot.  This mirrors the "replace-always
  // on collision" behavior typical of modern engines.

  slot.key = key;
  slot.depth = static_cast<int16_t>(depth);
  slot.score = score;
  if (best_move.has_value()) {
    slot.best_move = *best_move;
    slot.has_best_move = true;
  } else {
    slot.has_best_move = false;
  }
  slot.bound = bound;
  slot.age = static_cast<uint16_t>(current_age_);
}

} // namespace search
