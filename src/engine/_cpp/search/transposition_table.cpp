#include "transposition_table.hpp"

#include <algorithm>
#include <cstring>

namespace search {

// Round up to the nearest power of 2.
static constexpr size_t next_power_of_two(size_t v) noexcept {
  if (v == 0)
    return 1;
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v |= v >> 32;
  return v + 1;
}

TranspositionTable::TranspositionTable(size_t size_mb) noexcept {
  resize(size_mb);
}

void TranspositionTable::resize(size_t size_mb) noexcept {
  const size_t bytes = size_mb * 1024ULL * 1024ULL;
  size_t raw_count = bytes / sizeof(TTEntry);
  // Minimum 1024 entries for production use
  raw_count = std::max<size_t>(raw_count, MIN_ENTRIES);
  // Round down to power of 2 for fast masking
  size_t count = next_power_of_two(raw_count);
  // next_power_of_two rounds up, so if it overshot, halve it
  if (count > raw_count)
    count >>= 1;
  if (count < MIN_ENTRIES)
    count = MIN_ENTRIES;

  table_.assign(count, TTEntry{});
  mask_ = count - 1;
  entry_count_ = 0;
  age_ = 0;
}

void TranspositionTable::clear() noexcept {
  std::memset(table_.data(), 0, table_.size() * sizeof(TTEntry));
  entry_count_ = 0;
}

void TranspositionTable::increment_age() noexcept { age_++; }

const TTEntry *TranspositionTable::probe(uint64_t key) const noexcept {
  const size_t index = key & mask_;
  const TTEntry &entry = table_[index];
  if (entry.key == key) [[likely]] {
    return &entry;
  }
  return nullptr;
}

TTEntry *TranspositionTable::probe_mut(uint64_t key) noexcept {
  const size_t index = key & mask_;
  TTEntry &entry = table_[index];
  if (entry.key == key) [[likely]] {
    return &entry;
  }
  return nullptr;
}

std::optional<int32_t>
TranspositionTable::try_get_score(const TTEntry &entry, int depth, int alpha,
                                  int beta) const noexcept {
  if (entry.depth < static_cast<int16_t>(depth)) [[unlikely]] {
    return std::nullopt;
  }

  const auto bound = static_cast<BoundType>(entry.bound);
  if (bound == BoundType::EXACT) [[likely]] {
    return entry.score;
  }
  if (bound == BoundType::LOWER && entry.score >= beta)
    return entry.score;
  if (bound == BoundType::UPPER && entry.score <= alpha)
    return entry.score;

  return std::nullopt;
}

void TranspositionTable::store(uint64_t key, int depth, int32_t score,
                               const Move &best_move,
                               BoundType bound) noexcept {
  const size_t index = key & mask_;
  TTEntry &slot = table_[index];

  // Depth-preferred replacement with aging
  if (slot.key != 0) [[likely]] {
    // Existing entry: only replace if new entry has >= depth OR different age
    if (slot.age == age_ && slot.depth > static_cast<int16_t>(depth))
      return;
  } else {
    // Empty slot: new entry
    entry_count_++;
  }

  slot.key = key;
  slot.depth = static_cast<int16_t>(depth);
  slot.score = score;
  slot.best_move = best_move;
  slot.bound = static_cast<uint8_t>(bound);
  slot.age = age_;
}

size_t TranspositionTable::size() const noexcept { return entry_count_; }

size_t TranspositionTable::capacity() const noexcept { return table_.size(); }

int TranspositionTable::hashfull() const noexcept {
  if (table_.empty()) [[unlikely]] {
    return 0;
  }
  return static_cast<int>(entry_count_ * 1000 / table_.size());
}

} // namespace search
