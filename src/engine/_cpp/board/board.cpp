#include "board.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "../move_generation/move_generation.hpp"

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// Converts a piece letter (case-insensitive) to its PieceType.  Defaults to
// PAWN for unrecognised input (callers must validate before invoking).
[[nodiscard]] static constexpr PieceType piece_type_from_char(char c) noexcept {
  switch (c | 32) { // fold to lowercase
  case 'n':
    return PieceType::KNIGHT;
  case 'b':
    return PieceType::BISHOP;
  case 'r':
    return PieceType::ROOK;
  case 'q':
    return PieceType::QUEEN;
  case 'k':
    return PieceType::KING;
  default:
    return PieceType::PAWN;
  }
}

[[nodiscard]] static constexpr bool is_file_char(char c) noexcept {
  return c >= 'a' && c <= 'h';
}
[[nodiscard]] static constexpr bool is_rank_char(char c) noexcept {
  return c >= '1' && c <= '8';
}

// ---------------------------------------------------------------------------
// Piece
// ---------------------------------------------------------------------------

char Piece::symbol() const noexcept {
  if (!valid)
    return '.';
  static const char symbols[] = {'p', 'n', 'b', 'r', 'q', 'k'};
  char c = symbols[static_cast<uint8_t>(type)];
  return (color == Color::WHITE) ? static_cast<char>(c & ~32) : c;
}

// ---------------------------------------------------------------------------
// Board — piece queries  (O(1) via mailbox)
// ---------------------------------------------------------------------------

std::optional<Piece> Board::piece_at(uint8_t square) const noexcept {
  const uint8_t pt = piece_on[square];
  if (pt == EMPTY_SQ)
    return std::nullopt;
  return Piece{static_cast<PieceType>(pt), static_cast<Color>(color_on[square]),
               true};
}

std::vector<uint8_t> Board::pieces(PieceType pt, Color color) const noexcept {
  Bitboard bb = get_piece_bb(pt, color);
  std::vector<uint8_t> squares;
  squares.reserve(popcount(bb));
  while (bb)
    squares.push_back(pop_lsb(bb));
  return squares;
}

std::optional<uint8_t> Board::king(Color color) const noexcept {
  const Bitboard bb = get_piece_bb(PieceType::KING, color);
  if (!bb)
    return std::nullopt;
  return static_cast<uint8_t>(ctz64(bb));
}

// ---------------------------------------------------------------------------
// Board — castling / en-passant queries
// ---------------------------------------------------------------------------

bool Board::has_kingside_castling_rights(Color color) const noexcept {
  return (castling_rights & (color == Color::WHITE ? 0x01u : 0x04u)) != 0;
}

bool Board::has_queenside_castling_rights(Color color) const noexcept {
  return (castling_rights & (color == Color::WHITE ? 0x02u : 0x08u)) != 0;
}

bool Board::is_en_passant(const Move &move) const noexcept {
  if (en_passant_square == -1)
    return false;
  if (move.to != static_cast<uint8_t>(en_passant_square))
    return false;
  // The destination must be empty (the captured pawn is removed from a
  // different square, so the landing square is unoccupied before the move).
  return piece_on[move.from] == static_cast<uint8_t>(PieceType::PAWN) &&
         piece_on[move.to] == EMPTY_SQ;
}

bool Board::is_capture(const Move &move) const noexcept {
  const uint8_t our_color = color_on[move.from];
  const uint8_t dest_color = color_on[move.to];
  if (piece_on[move.to] != EMPTY_SQ && dest_color != our_color)
    return true;
  return is_en_passant(move);
}

bool Board::is_castling(const Move &move) const noexcept {
  if (piece_on[move.from] != static_cast<uint8_t>(PieceType::KING))
    return false;
  const int delta = static_cast<int>(move.to) - static_cast<int>(move.from);
  return delta == 2 || delta == -2;
}

bool Board::is_kingside_castling(const Move &move) const noexcept {
  return is_castling(move) && move.to > move.from;
}

bool Board::is_queenside_castling(const Move &move) const noexcept {
  return is_castling(move) && move.to < move.from;
}

bool Board::is_check() const noexcept {
  const Color us = side_to_move ? Color::WHITE : Color::BLACK;
  return is_in_check(*this, us);
}

std::string Board::print_move(const Move &move) const {
  return ::move_to_string(move, *this);
}

// ---------------------------------------------------------------------------
// Board — load FEN
// ---------------------------------------------------------------------------

void Board::load_fen(const std::string &fen) {
  clear();

  const char *ptr = fen.c_str();
  int8_t rank = 7, file = 0;

  // Piece placement
  while (*ptr && *ptr != ' ') {
    const char c = *ptr++;
    if (c == '/') {
      --rank;
      file = 0;
    } else if (c >= '1' && c <= '8') {
      file += c - '0';
    } else {
      const uint8_t square = static_cast<uint8_t>(rank * 8 + file);
      const Color color = (c >= 'A' && c <= 'Z') ? Color::WHITE : Color::BLACK;
      const uint8_t c_idx = static_cast<uint8_t>(color);
      const Bitboard sq_bb = 1ULL << square;

      PieceType pt;
      switch (c | 32) {
      case 'p':
        pt = PieceType::PAWN;
        break;
      case 'n':
        pt = PieceType::KNIGHT;
        break;
      case 'b':
        pt = PieceType::BISHOP;
        break;
      case 'r':
        pt = PieceType::ROOK;
        break;
      case 'q':
        pt = PieceType::QUEEN;
        break;
      case 'k':
        pt = PieceType::KING;
        break;
      default:
        ++file;
        continue; // skip invalid characters
      }

      const uint8_t pt_idx = static_cast<uint8_t>(pt);
      piece_bitboards[pt_idx] |= sq_bb;
      color_bitboards[c_idx] |= sq_bb;
      piece_on[square] = pt_idx;
      color_on[square] = c_idx;
      ++file;
    }
  }

  if (*ptr == ' ')
    ++ptr;

  // Active color
  side_to_move = (*ptr == 'w');
  ptr += 2; // skip colour char + space

  // Castling rights
  castling_rights = 0;
  while (*ptr && *ptr != ' ') {
    switch (*ptr++) {
    case 'K':
      castling_rights |= 1;
      break;
    case 'Q':
      castling_rights |= 2;
      break;
    case 'k':
      castling_rights |= 4;
      break;
    case 'q':
      castling_rights |= 8;
      break;
    default:
      break;
    }
  }
  ++ptr; // skip space

  // En passant
  if (*ptr == '-') {
    en_passant_square = -1;
    ptr += 2;
  } else {
    const int8_t ep_file = ptr[0] - 'a';
    const int8_t ep_rank = ptr[1] - '1';
    en_passant_square = static_cast<int8_t>(ep_rank * 8 + ep_file);
    ptr += 3;
  }

  // Halfmove clock
  halfmove_clock = 0;
  while (*ptr && *ptr != ' ')
    halfmove_clock = static_cast<uint8_t>(halfmove_clock * 10 + (*ptr++ - '0'));
  ++ptr;

  // Fullmove number
  fullmove_number = 0;
  while (*ptr && *ptr >= '0' && *ptr <= '9')
    fullmove_number =
        static_cast<uint16_t>(fullmove_number * 10 + (*ptr++ - '0'));
  if (fullmove_number == 0)
    fullmove_number = 1;
}

// ---------------------------------------------------------------------------
// Board — serialisation: to_fen / pretty
// ---------------------------------------------------------------------------

std::string Board::to_fen() const {
  std::string result;
  result.reserve(90);

  // Piece placement
  for (int8_t r = 7; r >= 0; --r) {
    uint8_t empty = 0;
    for (int8_t f = 0; f < 8; ++f) {
      const uint8_t sq = static_cast<uint8_t>(r * 8 + f);
      const uint8_t pt = piece_on[sq];
      if (pt == EMPTY_SQ) {
        ++empty;
      } else {
        if (empty > 0) {
          result += static_cast<char>('0' + empty);
          empty = 0;
        }
        static const char piece_chars[] = {'p', 'n', 'b', 'r', 'q', 'k'};
        char ch = piece_chars[pt];
        if (color_on[sq] == static_cast<uint8_t>(Color::WHITE))
          ch &= ~32;
        result += ch;
      }
    }
    if (empty > 0)
      result += static_cast<char>('0' + empty);
    if (r > 0)
      result += '/';
  }

  // Active colour
  result += side_to_move ? " w " : " b ";

  // Castling
  const bool any = castling_rights != 0;
  if (castling_rights & 1)
    result += 'K';
  if (castling_rights & 2)
    result += 'Q';
  if (castling_rights & 4)
    result += 'k';
  if (castling_rights & 8)
    result += 'q';
  if (!any)
    result += '-';

  // En passant
  result += ' ';
  if (en_passant_square == -1) {
    result += '-';
  } else {
    result += static_cast<char>('a' + (en_passant_square % 8));
    result += static_cast<char>('1' + (en_passant_square / 8));
  }

  result += ' ';
  result += std::to_string(halfmove_clock);
  result += ' ';
  result += std::to_string(fullmove_number);
  return result;
}

std::string Board::pretty() const {
  static const char piece_chars[] = {'p', 'n', 'b', 'r', 'q', 'k'};

  std::string result;
  result.reserve(200);

  for (int8_t r = 7; r >= 0; --r) {
    result += static_cast<char>('1' + r);
    result += "  ";
    for (int8_t f = 0; f < 8; ++f) {
      const uint8_t sq = static_cast<uint8_t>(r * 8 + f);
      const uint8_t pt = piece_on[sq];
      if (pt == EMPTY_SQ) {
        result += ". ";
      } else {
        char ch = piece_chars[pt];
        if (color_on[sq] == static_cast<uint8_t>(Color::WHITE))
          ch &= ~32;
        result += ch;
        result += ' ';
      }
    }
    result += '\n';
  }
  result += "\n   a b c d e f g h\n";
  return result;
}

// ---------------------------------------------------------------------------
// Board — move execution
// ---------------------------------------------------------------------------

void Board::push(const Move &move) {
  state_history.emplace_back();
  apply_move(move, &state_history.back());
}

void Board::push_null() {
  state_history.emplace_back();
  StateInfo &st = state_history.back();

  st.is_null_move = true;
  st.previous_castling_rights = castling_rights;
  st.previous_en_passant_square = en_passant_square;
  st.previous_halfmove_clock = halfmove_clock;
  st.previous_fullmove_number = fullmove_number;
  st.previous_side_to_move = side_to_move;

  en_passant_square = -1;
  ++halfmove_clock;
  if (!side_to_move)
    ++fullmove_number;
  side_to_move = !side_to_move;

  attacked_squares_valid_[0] = false;
  attacked_squares_valid_[1] = false;
}

Move Board::pop() {
  if (state_history.empty())
    throw std::runtime_error("Cannot pop from an empty move stack");

  const Move move = state_history.back().move;
  undo_move(state_history.back());
  state_history.pop_back();
  return move;
}

Bitboard Board::get_attacked_squares(Color color) const noexcept {
  const uint8_t idx = static_cast<uint8_t>(color);
  if (!attacked_squares_valid_[idx]) {
    cached_attacked_by_[idx] = ::compute_attacked_squares(*this, color);
    attacked_squares_valid_[idx] = true;
  }
  return cached_attacked_by_[idx];
}

void Board::update_attacked_squares(Color color) const noexcept {
  (void)get_attacked_squares(color);
}

Move Board::push_san(const std::string &san) {
  const Move move = parse_san(san);
  push(move);
  return move;
}

// ---------------------------------------------------------------------------
// Board — game-completion queries
// ---------------------------------------------------------------------------

bool Board::has_insufficient_material() const noexcept {
  // Any pawns, rooks, or queens → sufficient material.
  if (piece_bitboards[static_cast<uint8_t>(PieceType::PAWN)] |
      piece_bitboards[static_cast<uint8_t>(PieceType::ROOK)] |
      piece_bitboards[static_cast<uint8_t>(PieceType::QUEEN)])
    return false;

  const int wN = popcount(get_piece_bb(PieceType::KNIGHT, Color::WHITE));
  const int wB = popcount(get_piece_bb(PieceType::BISHOP, Color::WHITE));
  const int bN = popcount(get_piece_bb(PieceType::KNIGHT, Color::BLACK));
  const int bB = popcount(get_piece_bb(PieceType::BISHOP, Color::BLACK));
  const int wMin = wN + wB;
  const int bMin = bN + bB;

  if (wMin == 0 && bMin == 0)
    return true; // K vs K
  if ((wMin == 1) != (bMin == 1) && (wMin + bMin == 1))
    return true; // K+minor vs K
  if (wMin == 1 && bMin == 1 && wN == 1 && bN == 1)
    return true; // KN vs KN

  return false;
}

GameState Board::is_game_over() const noexcept {
  // Legal move generation is required to distinguish checkmate from stalemate;
  // perform it first so subsequent conditions can skip it.
  const std::vector<Move> legal_moves = generate_legal_moves();

  if (legal_moves.empty()) {
    const Color us = side_to_move ? Color::WHITE : Color::BLACK;
    return is_in_check(*this, us) ? GameState::CHECKMATE : GameState::STALEMATE;
  }

  if (halfmove_clock >= 100)
    return GameState::DRAW_BY_FIFTY_MOVE;
  if (has_insufficient_material())
    return GameState::DRAW_BY_INSUFFICIENT_MATERIAL;

  return GameState::ONGOING;
}

// ---------------------------------------------------------------------------
// Board — apply_move  (incremental update of both bitboards and mailbox)
// ---------------------------------------------------------------------------

void Board::apply_move(const Move &move, StateInfo *state) noexcept {
  const uint8_t from = move.from;
  const uint8_t to = move.to;
  const uint8_t promotion = move.promotion;

  // Retrieve moving piece from mailbox — O(1), no bitboard iteration.
  const uint8_t moving_pt_idx = piece_on[from];
  const uint8_t us_idx = color_on[from];
  const uint8_t them_idx = us_idx ^ 1u;

  const PieceType moving_pt = static_cast<PieceType>(moving_pt_idx);

  const Bitboard from_bb = 1ULL << from;
  const Bitboard to_bb = 1ULL << to;
  const int8_t old_ep = en_passant_square;

  // Classify the move: standard capture, en-passant, or quiet.
  bool is_ep = false;
  bool is_capt = (color_bitboards[them_idx] & to_bb) != 0;
  uint8_t capt_sq = to;
  uint8_t capt_pt_idx = piece_on[to];

  if (moving_pt == PieceType::PAWN && old_ep != -1 &&
      to == static_cast<uint8_t>(old_ep) && !is_capt) {
    is_ep = true;
    is_capt = true;
    capt_sq = static_cast<uint8_t>(old_ep + (us_idx == 0u ? -8 : 8));
    capt_pt_idx = static_cast<uint8_t>(PieceType::PAWN);
  }

  // Detect castling: the king moves exactly two squares horizontally.
  const int king_delta = static_cast<int>(to) - static_cast<int>(from);
  const bool is_cast =
      (moving_pt == PieceType::KING) && (king_delta == 2 || king_delta == -2);
  const bool kingside = is_cast && (to > from);

  // Persist reversible state before any mutation.
  if (state) {
    state->move = move;
    state->moving_piece = moving_pt;
    state->mover = static_cast<Color>(us_idx);
    state->was_capture = is_capt;
    state->captured_piece = static_cast<PieceType>(capt_pt_idx);
    state->captured_color = static_cast<Color>(them_idx);
    state->captured_square = is_capt ? capt_sq : 64u;
    state->was_en_passant = is_ep;
    state->was_castling = is_cast;
    state->was_kingside_castle = kingside;
    state->was_promotion = (promotion != 0);
    state->previous_castling_rights = castling_rights;
    state->previous_en_passant_square = en_passant_square;
    state->previous_halfmove_clock = halfmove_clock;
    state->previous_fullmove_number = fullmove_number;
    state->previous_side_to_move = side_to_move;
    state->is_null_move = false;
  }

  // Update clocks.
  halfmove_clock = (is_capt || moving_pt == PieceType::PAWN)
                       ? 0u
                       : static_cast<uint8_t>(halfmove_clock + 1u);
  if (us_idx == static_cast<uint8_t>(Color::BLACK))
    ++fullmove_number;

  en_passant_square = -1;

  // Remove captured piece from bitboards and mailbox.
  if (is_capt) {
    const Bitboard capt_bb = 1ULL << capt_sq;
    piece_bitboards[capt_pt_idx] &= ~capt_bb;
    color_bitboards[them_idx] &= ~capt_bb;
    piece_on[capt_sq] = EMPTY_SQ;
  }

  // Lift the moving piece from its source square.
  piece_bitboards[moving_pt_idx] &= ~from_bb;
  color_bitboards[us_idx] &= ~from_bb;
  piece_on[from] = EMPTY_SQ;

  // Place the piece at its destination (promotion changes the piece identity).
  const uint8_t dest_pt_idx = (promotion != 0u) ? promotion : moving_pt_idx;
  piece_bitboards[dest_pt_idx] |= to_bb;
  color_bitboards[us_idx] |= to_bb;
  piece_on[to] = dest_pt_idx;
  color_on[to] = us_idx;

  // Set en-passant square for a double pawn push.
  if (moving_pt == PieceType::PAWN && (king_delta == 16 || king_delta == -16))
    en_passant_square = static_cast<int8_t>(from + (us_idx == 0u ? 8 : -8));

  // Relocate the rook when castling.
  if (is_cast) {
    const uint8_t rook_from =
        kingside ? (us_idx == 0u ? 7u : 63u) : (us_idx == 0u ? 0u : 56u);
    const uint8_t rook_to = kingside ? static_cast<uint8_t>(from + 1u)
                                     : static_cast<uint8_t>(from - 1u);
    const Bitboard rmask = (1ULL << rook_from) | (1ULL << rook_to);
    const uint8_t rook_pt = static_cast<uint8_t>(PieceType::ROOK);
    piece_bitboards[rook_pt] ^= rmask;
    color_bitboards[us_idx] ^= rmask;
    piece_on[rook_from] = EMPTY_SQ;
    piece_on[rook_to] = rook_pt;
    color_on[rook_to] = us_idx;
  }

  // Update castling availability.
  if (castling_rights) {
    if (moving_pt == PieceType::KING) {
      // Moving king forfeits both rights for the active side.
      castling_rights &= static_cast<uint8_t>(us_idx == 0u ? ~0x03u : ~0x0Cu);
    } else {
      // A rook leaving its home corner, or a capture of a corner rook,
      // revokes the corresponding right.
      const auto revoke = [&](uint8_t sq, uint8_t mask) noexcept {
        if (from == sq || (is_capt && capt_sq == sq))
          castling_rights &= static_cast<uint8_t>(~mask);
      };
      revoke(0u, 0x02u);  // a1 — white queenside
      revoke(7u, 0x01u);  // h1 — white kingside
      revoke(56u, 0x08u); // a8 — black queenside
      revoke(63u, 0x04u); // h8 — black kingside
    }
  }

  side_to_move = !side_to_move;
  attacked_squares_valid_[0] = false;
  attacked_squares_valid_[1] = false;
}

// ---------------------------------------------------------------------------
// Board — undo_move  (restore bitboards, mailbox, and game state)
// ---------------------------------------------------------------------------

void Board::undo_move(const StateInfo &st) noexcept {
  // Restore all scalar game state unconditionally.
  castling_rights = st.previous_castling_rights;
  en_passant_square = st.previous_en_passant_square;
  halfmove_clock = st.previous_halfmove_clock;
  fullmove_number = st.previous_fullmove_number;
  side_to_move = st.previous_side_to_move;

  attacked_squares_valid_[0] = false;
  attacked_squares_valid_[1] = false;

  if (st.is_null_move)
    return;

  const uint8_t from = st.move.from;
  const uint8_t to = st.move.to;
  const uint8_t us_idx = static_cast<uint8_t>(st.mover);
  const uint8_t them_idx = us_idx ^ 1u;

  const Bitboard from_bb = 1ULL << from;
  const Bitboard to_bb = 1ULL << to;

  // Remove the piece at the destination (could be the promoted piece).
  const uint8_t dest_pt_idx = st.was_promotion
                                  ? st.move.promotion
                                  : static_cast<uint8_t>(st.moving_piece);
  piece_bitboards[dest_pt_idx] &= ~to_bb;
  color_bitboards[us_idx] &= ~to_bb;
  piece_on[to] = EMPTY_SQ;

  // Restore the moving piece at the source square.
  const uint8_t moving_pt_idx = static_cast<uint8_t>(st.moving_piece);
  piece_bitboards[moving_pt_idx] |= from_bb;
  color_bitboards[us_idx] |= from_bb;
  piece_on[from] = moving_pt_idx;
  color_on[from] = us_idx;

  // Restore the captured piece.  For en passant the captured square differs
  // from the move destination; StateInfo always records the exact square.
  if (st.was_capture) {
    const uint8_t capt_sq = st.captured_square;
    const uint8_t capt_pt_idx = static_cast<uint8_t>(st.captured_piece);
    const Bitboard capt_bb = 1ULL << capt_sq;
    piece_bitboards[capt_pt_idx] |= capt_bb;
    color_bitboards[them_idx] |= capt_bb;
    piece_on[capt_sq] = capt_pt_idx;
    color_on[capt_sq] = them_idx;
  }

  // Undo rook relocation from castling.
  if (st.was_castling) {
    const bool kingside = st.was_kingside_castle;
    const uint8_t rook_from =
        kingside ? (us_idx == 0u ? 7u : 63u) : (us_idx == 0u ? 0u : 56u);
    const uint8_t rook_to = kingside ? static_cast<uint8_t>(from + 1u)
                                     : static_cast<uint8_t>(from - 1u);
    const Bitboard rmask = (1ULL << rook_from) | (1ULL << rook_to);
    const uint8_t rook_pt = static_cast<uint8_t>(PieceType::ROOK);
    piece_bitboards[rook_pt] ^= rmask;
    color_bitboards[us_idx] ^= rmask;
    piece_on[rook_to] = EMPTY_SQ;
    piece_on[rook_from] = rook_pt;
    color_on[rook_from] = us_idx;
  }
}

// ---------------------------------------------------------------------------
// Board — parse_san  (converts Standard Algebraic Notation to a Move)
// ---------------------------------------------------------------------------

Move Board::parse_san(const std::string &san) const {
  // Strip whitespace and annotation characters.
  std::string work;
  work.reserve(san.size());
  for (char c : san)
    if (!std::isspace(static_cast<unsigned char>(c)))
      work.push_back(c);
  if (work.empty())
    throw std::runtime_error("Empty SAN string");

  while (!work.empty() && (work.back() == '+' || work.back() == '#' ||
                           work.back() == '!' || work.back() == '?'))
    work.pop_back();

  const auto legal_moves = generate_legal_moves();

  auto match_castle = [&](bool kingside) -> Move {
    for (const auto &mv : legal_moves)
      if (is_castling(mv) && (mv.to > mv.from) == kingside)
        return mv;
    throw std::runtime_error("No legal castling move for SAN: " + san);
  };

  if (work == "O-O" || work == "0-0")
    return match_castle(true);
  if (work == "O-O-O" || work == "0-0-0")
    return match_castle(false);

  // Extract promotion suffix.
  uint8_t promotion = 0;
  const auto eq_pos = work.find('=');
  if (eq_pos != std::string::npos) {
    if (eq_pos + 1 >= work.size())
      throw std::runtime_error("Invalid promotion SAN: " + san);
    promotion = static_cast<uint8_t>(piece_type_from_char(work[eq_pos + 1]));
    work.erase(eq_pos);
  }

  if (work.size() < 2)
    throw std::runtime_error("Invalid SAN: " + san);

  // Extract target square (last two characters).
  const char tgt_file = work[work.size() - 2];
  const char tgt_rank = work[work.size() - 1];
  if (!is_file_char(tgt_file) || !is_rank_char(tgt_rank))
    throw std::runtime_error("Invalid target square in SAN: " + san);
  const uint8_t tgt_sq =
      static_cast<uint8_t>((tgt_rank - '1') * 8 + (tgt_file - 'a'));
  work.erase(work.size() - 2);

  const bool capture_flag = (work.find('x') != std::string::npos);
  work.erase(std::remove(work.begin(), work.end(), 'x'), work.end());

  // Piece type (uppercase prefix); absent implies pawn.
  PieceType desired = PieceType::PAWN;
  if (!work.empty() && std::isupper(static_cast<unsigned char>(work.front()))) {
    desired = piece_type_from_char(work.front());
    work.erase(work.begin());
  }

  // Optional disambiguation.
  std::optional<char> disamb_file;
  std::optional<char> disamb_rank;
  for (char c : work) {
    if (is_file_char(c))
      disamb_file = c;
    if (is_rank_char(c))
      disamb_rank = c;
  }

  std::optional<Move> candidate;
  for (const auto &mv : legal_moves) {
    if (piece_on[mv.from] != static_cast<uint8_t>(desired))
      continue;
    if (mv.to != tgt_sq)
      continue;
    if (promotion != 0 ? mv.promotion != promotion : mv.promotion != 0)
      continue;
    if (capture_flag != is_capture(mv))
      continue;
    if (disamb_file &&
        static_cast<char>('a' + square_file(mv.from)) != *disamb_file)
      continue;
    if (disamb_rank &&
        static_cast<char>('1' + square_rank(mv.from)) != *disamb_rank)
      continue;

    if (candidate)
      throw std::runtime_error("Ambiguous SAN: " + san);
    candidate = mv;
  }

  if (!candidate)
    throw std::runtime_error("Illegal SAN: " + san);
  return *candidate;
}

// ---------------------------------------------------------------------------
// move_from_uci / move_to_uci  (global helpers exposed through bindings)
// ---------------------------------------------------------------------------

Move move_from_uci(const std::string &uci) {
  if (uci.size() < 4)
    throw std::runtime_error("Invalid UCI string: " + uci);

  const auto parse_sq = [&](char file, char rank) -> uint8_t {
    file = static_cast<char>(file | 32); // fold to lowercase
    if (!is_file_char(file) || !is_rank_char(rank))
      throw std::runtime_error("Invalid UCI square in: " + uci);
    return static_cast<uint8_t>((rank - '1') * 8 + (file - 'a'));
  };

  const uint8_t from = parse_sq(uci[0], uci[1]);
  const uint8_t to = parse_sq(uci[2], uci[3]);
  uint8_t promotion = 0;
  if (uci.size() >= 5)
    promotion = static_cast<uint8_t>(piece_type_from_char(uci[4]));
  return Move{from, to, promotion};
}

std::string move_to_uci(const Move &move) {
  std::string uci;
  uci.reserve(5);
  uci.push_back(static_cast<char>('a' + square_file(move.from)));
  uci.push_back(static_cast<char>('1' + square_rank(move.from)));
  uci.push_back(static_cast<char>('a' + square_file(move.to)));
  uci.push_back(static_cast<char>('1' + square_rank(move.to)));
  if (move.promotion != 0) {
    static const char promo_map[] = {' ', 'n', 'b', 'r', 'q', 'k'};
    uci.push_back(promo_map[move.promotion < 6u ? move.promotion : 0u]);
  }
  return uci;
}

// ---------------------------------------------------------------------------
// Board — perft  (bulk-counting node enumerator for move-generation testing)
// ---------------------------------------------------------------------------

uint64_t Board::perft(int depth) noexcept {
  if (depth == 0)
    return 1ULL;

  const std::vector<Move> moves = generate_legal_moves();
  if (depth == 1)
    return static_cast<uint64_t>(moves.size());

  uint64_t nodes = 0;
  for (const auto &mv : moves) {
    push(mv);
    nodes += perft(depth - 1);
    (void)pop();
  }
  return nodes;
}
