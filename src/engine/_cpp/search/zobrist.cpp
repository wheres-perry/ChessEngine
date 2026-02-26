// ---------------------------------------------------------------------------
// zobrist.cpp — High-performance Zobrist hashing (non-PolyGlot)
//
// Key design choices vs. the old PolyGlot implementation:
//   1. Compile-time SplitMix64 keys in zobrist_keys.hpp — zero init cost.
//   2. piece[color][pt][sq] direct 3-D indexing — no offset arithmetic.
//   3. castling[rights_mask] single-lookup 16-entry table — replaces 4
//      conditional XORs.
//   4. EP hashed unconditionally by file — no pawn-can-capture filter,
//      which was a PolyGlot spec detail irrelevant for engine use.
//   5. make_null_move_hash() — O(1) null-move hash (toggle side + remove
//      old EP).
//   6. [[unlikely]] on cold fallback paths for branch-prediction hints.
// ---------------------------------------------------------------------------

#include "zobrist.hpp"

#include "zobrist_keys.hpp"

namespace search {

Zobrist::Zobrist(std::optional<uint64_t> /*seed*/) noexcept
    : current_hash_(std::nullopt) {}

// ---------------------------------------------------------------------------
// hash_board — Full O(popcount) hash from scratch
// ---------------------------------------------------------------------------
uint64_t Zobrist::hash_board(const Board &board) noexcept {
  uint64_t h = 0;

  // 1. Pieces — iterate bitboards with hardware CTZ/BLSR
  for (uint8_t pt = 0; pt < NUM_PIECE_TYPES; ++pt) {
    for (uint8_t c = 0; c < NUM_COLORS; ++c) {
      Bitboard bb =
          board.get_piece_bb(static_cast<PieceType>(pt), static_cast<Color>(c));
      while (bb) {
        h ^= ZKEYS.piece[c][pt][pop_lsb(bb)];
      }
    }
  }

  // 2. Castling — single indexed load (no per-bit branching)
  h ^= ZKEYS.castling[board.get_castling_rights()];

  // 3. En passant — hash file unconditionally when EP is set
  const int8_t ep = board.get_en_passant_square();
  if (ep != -1) {
    h ^= ZKEYS.ep_file[square_file(static_cast<uint8_t>(ep))];
  }

  // 4. Side to move — XOR when white to move
  if (board.get_side_to_move()) {
    h ^= ZKEYS.side;
  }

  current_hash_ = h;
  return h;
}

// ---------------------------------------------------------------------------
// make_move_hash — O(1) incremental update
//
// Given the current hash and a pre-move Board, computes the hash of the
// position that results from applying 'move'.  Uses the Board's mailbox
// arrays for O(1) piece identification.
// ---------------------------------------------------------------------------
uint64_t Zobrist::make_move_hash(Board &board, const Move &move) noexcept {
  // Cold fallback: no hash cached yet.
  if (!current_hash_.has_value()) [[unlikely]] {
    board.push(move);
    const uint64_t h = hash_board(board);
    (void)board.pop();
    return h;
  }

  uint64_t h = current_hash_.value();

  const uint8_t from = move.from;
  const uint8_t to = move.to;
  const uint8_t promo = move.promotion;

  // Identify moving piece via mailbox — O(1)
  const auto piece_opt = board.piece_at(from);
  if (!piece_opt.has_value()) [[unlikely]] {
    board.push(move);
    const uint64_t fh = hash_board(board);
    (void)board.pop();
    return fh;
  }

  const uint8_t moving_pt = static_cast<uint8_t>(piece_opt->type);
  const uint8_t us = static_cast<uint8_t>(piece_opt->color);
  const uint8_t them = us ^ 1u;

  const int8_t old_ep = board.get_en_passant_square();
  const uint8_t old_cr = board.get_castling_rights();

  // --- 1. Toggle side to move (always flips) ---
  h ^= ZKEYS.side;

  // --- 2. Remove old en-passant contribution ---
  if (old_ep != -1) {
    h ^= ZKEYS.ep_file[square_file(static_cast<uint8_t>(old_ep))];
  }

  // --- 3. Remove old castling contribution (single lookup) ---
  h ^= ZKEYS.castling[old_cr];

  // --- 4. XOR out moving piece from source ---
  h ^= ZKEYS.piece[us][moving_pt][from];

  // --- 5. Handle captures ---
  bool is_ep = false;
  bool is_capt = false;
  uint8_t capt_sq = to;

  // Check en-passant capture
  if (moving_pt == static_cast<uint8_t>(PieceType::PAWN) && old_ep != -1 &&
      to == static_cast<uint8_t>(old_ep)) {
    const auto target_piece = board.piece_at(to);
    if (!target_piece.has_value() || !target_piece->valid) {
      is_ep = true;
      is_capt = true;
      capt_sq = static_cast<uint8_t>(old_ep + (us == 0u ? -8 : 8));
    }
  }

  if (is_ep) {
    // XOR out the captured pawn
    h ^= ZKEYS.piece[them][static_cast<uint8_t>(PieceType::PAWN)][capt_sq];
  } else {
    // Normal capture — check mailbox at destination
    const auto captured = board.piece_at(to);
    if (captured.has_value() && captured->valid) {
      is_capt = true;
      h ^= ZKEYS.piece[static_cast<uint8_t>(captured->color)]
                      [static_cast<uint8_t>(captured->type)][to];
    }
  }

  // --- 6. XOR in the piece at destination ---
  if (promo != 0) {
    // Promotion: arriving piece is the promoted type
    h ^= ZKEYS.piece[us][promo][to];
  } else {
    h ^= ZKEYS.piece[us][moving_pt][to];
  }

  // --- 7. Handle castling rook movement ---
  const int delta = static_cast<int>(to) - static_cast<int>(from);
  const bool is_castling =
      (moving_pt == static_cast<uint8_t>(PieceType::KING)) &&
      (delta == 2 || delta == -2);

  if (is_castling) {
    const bool kingside = (to > from);
    const uint8_t rook_from =
        kingside ? (us == 0u ? 7u : 63u) : (us == 0u ? 0u : 56u);
    const uint8_t rook_to = kingside ? static_cast<uint8_t>(from + 1u)
                                     : static_cast<uint8_t>(from - 1u);
    constexpr uint8_t ROOK_PT = static_cast<uint8_t>(PieceType::ROOK);
    h ^= ZKEYS.piece[us][ROOK_PT][rook_from];
    h ^= ZKEYS.piece[us][ROOK_PT][rook_to];
  }

  // --- 8. Compute new castling rights ---
  uint8_t new_cr = old_cr;
  if (new_cr) {
    if (moving_pt == static_cast<uint8_t>(PieceType::KING)) {
      new_cr &= static_cast<uint8_t>(us == 0u ? ~0x03u : ~0x0Cu);
    }
    // Rook leaving its home corner or captured on its home corner
    auto revoke = [&](uint8_t sq, uint8_t mask) noexcept {
      if (from == sq || (is_capt && capt_sq == sq))
        new_cr &= static_cast<uint8_t>(~mask);
    };
    revoke(0u, 0x02u);  // a1 — white queenside
    revoke(7u, 0x01u);  // h1 — white kingside
    revoke(56u, 0x08u); // a8 — black queenside
    revoke(63u, 0x04u); // h8 — black kingside
  }
  h ^= ZKEYS.castling[new_cr];

  // --- 9. Compute new en-passant square ---
  // Double pawn push → EP square; otherwise cleared.
  if (moving_pt == static_cast<uint8_t>(PieceType::PAWN) &&
      (delta == 16 || delta == -16)) {
    const auto new_ep = static_cast<uint8_t>(from + (us == 0u ? 8 : -8));
    h ^= ZKEYS.ep_file[square_file(new_ep)];
  }

  return h;
}

// ---------------------------------------------------------------------------
// make_null_move_hash — O(1) incremental null-move update
//
// A null move only flips side-to-move and clears the EP square.  No piece,
// castling, or other state changes.  This avoids the full O(popcount)
// hash_board() call that the old code used after push_null().
// ---------------------------------------------------------------------------
uint64_t Zobrist::make_null_move_hash(const Board &board) noexcept {
  if (!current_hash_.has_value()) [[unlikely]] {
    // No hash available — compute from scratch for current position,
    // then apply the null-move delta.
    // (This cast is safe: hash_board only reads the board.)
    uint64_t h = hash_board(const_cast<Board &>(
        board)); // NOLINT(cppcoreguidelines-pro-type-const-cast)
    // Now apply null-move: toggle side, remove EP.
    h ^= ZKEYS.side;
    const int8_t ep = board.get_en_passant_square();
    if (ep != -1) {
      h ^= ZKEYS.ep_file[square_file(static_cast<uint8_t>(ep))];
    }
    return h;
  }

  uint64_t h = current_hash_.value();

  // Toggle side to move
  h ^= ZKEYS.side;

  // Remove old EP contribution (null move clears EP — no new EP to add)
  const int8_t ep = board.get_en_passant_square();
  if (ep != -1) {
    h ^= ZKEYS.ep_file[square_file(static_cast<uint8_t>(ep))];
  }

  return h;
}

// ---------------------------------------------------------------------------
// Accessors / mutators
// ---------------------------------------------------------------------------

std::optional<uint64_t> Zobrist::get_current_hash() const noexcept {
  return current_hash_;
}

void Zobrist::set_current_hash(std::optional<uint64_t> hash_val) noexcept {
  current_hash_ = hash_val;
}

void Zobrist::invalidate_hash() noexcept { current_hash_ = std::nullopt; }

} // namespace search
