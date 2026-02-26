#pragma once
// ---------------------------------------------------------------------------
// zobrist_keys.hpp — Compile-time generated Zobrist key tables
//
// Replaces the PolyGlot random array with purpose-built keys generated via
// SplitMix64 (same PRNG quality as Stockfish's initializer).  Every key is
// a compile-time constant so the entire table lives in .rodata with zero
// runtime initialisation cost.
//
// Layout is optimised for cache locality during both full-board hashing
// (iterates [pt][color] then scans bits) and incremental updates (direct
// [color][pt][sq] indexing).
// ---------------------------------------------------------------------------

#include <array>
#include <cstdint>

namespace search {

// ---------------------------------------------------------------------------
// Compile-time PRNG — SplitMix64
// ---------------------------------------------------------------------------
constexpr uint64_t splitmix64(uint64_t &state) noexcept {
  state += 0x9e3779b97f4a7c15ULL;
  uint64_t z = state;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

// ---------------------------------------------------------------------------
// Key table structure — cache-line aligned, 6344 bytes total (fits L1)
// ---------------------------------------------------------------------------
struct alignas(64) ZobristKeys {
  // piece[color][piece_type][square] — 2 × 6 × 64 = 768 keys
  // Indexed as ZKEYS.piece[color][pt][sq] for direct O(1) access.
  uint64_t piece[2][6][64];

  // castling[rights_mask] — 16 entries indexed by the full 4-bit castling
  // rights bitmask.  Built from 4 independent base keys so that the XOR
  // differential property holds: castling[a] ^ castling[b] correctly
  // reflects the rights that changed.  Single table lookup replaces up to
  // 4 conditional XORs.
  uint64_t castling[16];

  // ep_file[file] — 8 keys, one per file.  Hashed when the EP square is
  // set.  No pawn-can-capture validation (engine sets EP only when legal).
  uint64_t ep_file[8];

  // side — single key XORed to distinguish side-to-move.
  uint64_t side;
};

// ---------------------------------------------------------------------------
// Compile-time key generation
// ---------------------------------------------------------------------------
constexpr ZobristKeys generate_keys() noexcept {
  ZobristKeys keys{};
  uint64_t state = 0x5D69D5B97F4A7C15ULL; // fixed seed — deterministic

  // --- Piece keys ---
  for (int c = 0; c < 2; ++c)
    for (int pt = 0; pt < 6; ++pt)
      for (int sq = 0; sq < 64; ++sq)
        keys.piece[c][pt][sq] = splitmix64(state);

  // --- Castling keys (build 16-entry table from 4 base keys) ---
  uint64_t castle_base[4];
  for (int i = 0; i < 4; ++i)
    castle_base[i] = splitmix64(state);

  for (int mask = 0; mask < 16; ++mask) {
    uint64_t h = 0;
    for (int bit = 0; bit < 4; ++bit)
      if (mask & (1 << bit))
        h ^= castle_base[bit];
    keys.castling[mask] = h;
  }

  // --- En-passant file keys ---
  for (int f = 0; f < 8; ++f)
    keys.ep_file[f] = splitmix64(state);

  // --- Side-to-move key ---
  keys.side = splitmix64(state);

  return keys;
}

// Single global constexpr instance — zero runtime cost, lives in .rodata.
inline constexpr ZobristKeys ZKEYS = generate_keys();

} // namespace search
