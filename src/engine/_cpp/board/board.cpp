#include "board.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <sstream>
#include <vector>

#include "../move_generation/move_generation.hpp"

// Helper function to get the piece type at a square
[[nodiscard]] inline PieceType get_piece_at(
    const std::array<Bitboard, NUM_PIECE_TYPES>& piece_bbs,
    Bitboard square_bb) noexcept {
  // Use direct bit tests instead of loop for common pieces
  if (piece_bbs[0] & square_bb) return PieceType::PAWN;
  if (piece_bbs[1] & square_bb) return PieceType::KNIGHT;
  if (piece_bbs[2] & square_bb) return PieceType::BISHOP;
  if (piece_bbs[3] & square_bb) return PieceType::ROOK;
  if (piece_bbs[4] & square_bb) return PieceType::QUEEN;
  if (piece_bbs[5] & square_bb) return PieceType::KING;

  return static_cast<PieceType>(NUM_PIECE_TYPES);  // Empty square
}

// Helper to get color at a square (assumes occupied)
[[nodiscard]] inline Color get_color_at(
    const std::array<Bitboard, NUM_COLORS>& color_bbs,
    Bitboard square_bb) noexcept {
  return (color_bbs[0] & square_bb) ? Color::WHITE : Color::BLACK;
}

[[nodiscard]] inline PieceType piece_type_from_char(char c) {
  switch (std::tolower(static_cast<unsigned char>(c))) {
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

[[nodiscard]] inline bool is_file_char(char c) {
  return c >= 'a' && c <= 'h';
}

[[nodiscard]] inline bool is_rank_char(char c) {
  return c >= '1' && c <= '8';
}

char Piece::symbol() const noexcept {
  if (!valid) return '.';
  static const char symbols[] = {'p', 'n', 'b', 'r', 'q', 'k'};
  char c = symbols[static_cast<uint8_t>(type)];
  if (color == Color::WHITE) c = static_cast<char>(std::toupper(c));
  return c;
}

std::optional<Piece> Board::piece_at(uint8_t square) const noexcept {
  Bitboard bb = square_bitboard(square);
  PieceType pt = get_piece_at(piece_bitboards, bb);
  if (pt == static_cast<PieceType>(NUM_PIECE_TYPES)) {
    return std::nullopt;
  }
  Color color = get_color_at(color_bitboards, bb);
  return Piece{pt, color, true};
}

std::vector<uint8_t> Board::pieces(PieceType pt,
                                   Color color) const noexcept {
  Bitboard bb = get_piece_bb(pt, color);
  std::vector<uint8_t> squares;
  squares.reserve(popcount(bb));

  while (bb) {
    squares.push_back(pop_lsb(bb));
  }
  return squares;
}

std::optional<uint8_t> Board::king(Color color) const noexcept {
  Bitboard king_bb = get_piece_bb(PieceType::KING, color);
  if (!king_bb) return std::nullopt;
  return static_cast<uint8_t>(__builtin_ctzll(king_bb));
}

bool Board::has_kingside_castling_rights(Color color) const noexcept {
  return color == Color::WHITE ? (castling_rights & 0x01) != 0
                               : (castling_rights & 0x04) != 0;
}

bool Board::has_queenside_castling_rights(Color color) const noexcept {
  return color == Color::WHITE ? (castling_rights & 0x02) != 0
                               : (castling_rights & 0x08) != 0;
}

std::optional<uint8_t> Board::ep_square() const noexcept {
  if (en_passant_square == -1) return std::nullopt;
  return static_cast<uint8_t>(en_passant_square);
}

bool Board::is_en_passant(const Move& move) const noexcept {
  if (en_passant_square == -1) return false;
  if (move.to != static_cast<uint8_t>(en_passant_square)) return false;
  auto piece = piece_at(move.from);
  return piece && piece->type == PieceType::PAWN &&
         !piece_at(move.to).has_value();
}

bool Board::is_capture(const Move& move) const noexcept {
  Bitboard to_bb = square_bitboard(move.to);
  Color us =
      (color_bitboards[0] & square_bitboard(move.from)) ? Color::WHITE : Color::BLACK;
  uint8_t them_idx = static_cast<uint8_t>(us == Color::WHITE ? Color::BLACK
                                                             : Color::WHITE);
  if (color_bitboards[them_idx] & to_bb) return true;
  return is_en_passant(move);
}

bool Board::is_castling(const Move& move) const noexcept {
  auto piece = piece_at(move.from);
  if (!piece || piece->type != PieceType::KING) return false;
  return std::abs(static_cast<int>(move.to) - static_cast<int>(move.from)) == 2;
}

bool Board::is_kingside_castling(const Move& move) const noexcept {
  return is_castling(move) && move.to > move.from;
}

bool Board::is_queenside_castling(const Move& move) const noexcept {
  return is_castling(move) && move.to < move.from;
}

bool Board::is_check() const noexcept {
  const Color us = side_to_move ? Color::WHITE : Color::BLACK;
  return is_in_check(*this, us);
}

std::string Board::print_move(const Move& move) const {
  return move_to_string(move, *this);
}

// Load FEN string - optimized but still handles all cases
void Board::load_fen(const std::string& fen) {
  clear();

  // Fast parsing using C-style approach
  const char* ptr = fen.c_str();
  int8_t rank = 7, file = 0;

  // Parse piece placement
  while (*ptr && *ptr != ' ') {
    char c = *ptr++;

    if (c == '/') {
      rank--;
      file = 0;
    } else if (c >= '1' && c <= '8') {
      file += c - '0';
    } else {
      uint8_t square = rank * 8 + file;
      Color color = (c >= 'A' && c <= 'Z') ? Color::WHITE : Color::BLACK;
      PieceType pt;

      // Use fast switch with lowercase comparison
      switch (c | 32)  // Convert to lowercase
      {
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
          continue;  // Skip invalid characters
      }

      piece_bitboards[static_cast<uint8_t>(pt)] |= (1ULL << square);
      color_bitboards[static_cast<uint8_t>(color)] |= (1ULL << square);
      file++;
    }
  }

  // Skip space
  if (*ptr == ' ') ptr++;

  // Active color
  side_to_move = (*ptr == 'w');
  ptr += 2;  // Skip color and space

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
        break;  // Skip '-' or other chars
    }
  }
  ptr++;  // Skip space

  // En passant
  if (*ptr == '-') {
    en_passant_square = -1;
    ptr += 2;  // Skip '-' and space
  } else {
    int8_t ep_file = ptr[0] - 'a';
    int8_t ep_rank = ptr[1] - '1';
    en_passant_square = ep_rank * 8 + ep_file;
    ptr += 3;  // Skip file, rank, space
  }

  // Halfmove clock
  halfmove_clock = 0;
  while (*ptr && *ptr != ' ') {
    halfmove_clock = halfmove_clock * 10 + (*ptr++ - '0');
  }
  ptr++;  // Skip space

  // Fullmove number
  fullmove_number = 0;
  while (*ptr && *ptr >= '0' && *ptr <= '9') {
    fullmove_number = fullmove_number * 10 + (*ptr++ - '0');
  }

  // Ensure minimum valid value
  if (fullmove_number == 0) fullmove_number = 1;
}

// Convert to FEN string - optimized
std::string Board::to_fen() const {
  std::string result;
  result.reserve(90);  // Preallocate for typical FEN length

  // Piece placement
  for (int8_t rank = 7; rank >= 0; --rank) {
    uint8_t empty_squares = 0;
    for (int8_t file = 0; file < 8; ++file) {
      uint8_t square = rank * 8 + file;
      Bitboard square_bb = 1ULL << square;

      // Check if square is occupied by any piece
      PieceType pt = get_piece_at(piece_bitboards, square_bb);

      if (pt != static_cast<PieceType>(NUM_PIECE_TYPES)) {
        // Output empty square count if any
        if (empty_squares > 0) {
          result += ('0' + empty_squares);
          empty_squares = 0;
        }

        // Output piece character
        bool is_white = (color_bitboards[0] & square_bb);
        char piece_char = 0;

        switch (pt) {
          case PieceType::PAWN:
            piece_char = 'p';
            break;
          case PieceType::KNIGHT:
            piece_char = 'n';
            break;
          case PieceType::BISHOP:
            piece_char = 'b';
            break;
          case PieceType::ROOK:
            piece_char = 'r';
            break;
          case PieceType::QUEEN:
            piece_char = 'q';
            break;
          case PieceType::KING:
            piece_char = 'k';
            break;
        }

        // Convert to uppercase for white pieces
        if (is_white) piece_char -= 32;  // ASCII uppercase conversion
        result += piece_char;
      } else {
        empty_squares++;
      }
    }

    // Output any remaining empty squares
    if (empty_squares > 0) {
      result += ('0' + empty_squares);
    }

    // Add rank separator (except for last rank)
    if (rank > 0) {
      result += '/';
    }
  }

  // Active color
  result += side_to_move ? " w " : " b ";

  // Castling
  bool has_castling = false;
  if (castling_rights & 1) {
    result += 'K';
    has_castling = true;
  }
  if (castling_rights & 2) {
    result += 'Q';
    has_castling = true;
  }
  if (castling_rights & 4) {
    result += 'k';
    has_castling = true;
  }
  if (castling_rights & 8) {
    result += 'q';
    has_castling = true;
  }
  if (!has_castling) result += '-';

  // En passant
  result += ' ';
  if (en_passant_square == -1) {
    result += '-';
  } else {
    result += ('a' + (en_passant_square % 8));
    result += ('1' + (en_passant_square / 8));
  }

  // Halfmove and fullmove counters
  result += ' ' + std::to_string(halfmove_clock);
  result += ' ' + std::to_string(fullmove_number);

  return result;
}

// Pretty print board - optimized
std::string Board::pretty() const {
  std::string result;
  result.reserve(200);  // Preallocate for typical board size

  for (int8_t rank = 7; rank >= 0; --rank) {
    result += static_cast<char>('1' + rank);
    result += "  ";

    for (int8_t file = 0; file < 8; ++file) {
      uint8_t square = rank * 8 + file;
      Bitboard square_bb = 1ULL << square;

      // Find piece at square
      PieceType pt = get_piece_at(piece_bitboards, square_bb);

      if (pt == static_cast<PieceType>(NUM_PIECE_TYPES)) {
        result += ". ";
      } else {
        bool is_white = (color_bitboards[0] & square_bb);
        char piece_char;

        switch (pt) {
          case PieceType::PAWN:
            piece_char = is_white ? 'P' : 'p';
            break;
          case PieceType::KNIGHT:
            piece_char = is_white ? 'N' : 'n';
            break;
          case PieceType::BISHOP:
            piece_char = is_white ? 'B' : 'b';
            break;
          case PieceType::ROOK:
            piece_char = is_white ? 'R' : 'r';
            break;
          case PieceType::QUEEN:
            piece_char = is_white ? 'Q' : 'q';
            break;
          case PieceType::KING:
            piece_char = is_white ? 'K' : 'k';
            break;
        }

        result += piece_char;
        result += ' ';
      }
    }
    result += '\n';
  }

  result += "\n   a b c d e f g h\n";
  return result;
}

void Board::push(const Move& move) {
  StateInfo state{};
  apply_move(move, &state);
  state_history.push_back(state);
}

Move Board::pop() {
  if (state_history.empty()) {
    throw std::runtime_error("Cannot pop from an empty move stack");
  }
  StateInfo state = state_history.back();
  state_history.pop_back();
  undo_move(state);
  return state.move;
}

Move Board::push_san(const std::string& san) {
  Move move = parse_san(san);
  push(move);
  return move;
}

// Optimized helper method to check for insufficient material
bool Board::has_insufficient_material() const noexcept {
  // Direct popcount for material counting - leverages hardware instructions
  const int white_knights =
      popcount(get_piece_bb(PieceType::KNIGHT, Color::WHITE));
  const int white_bishops =
      popcount(get_piece_bb(PieceType::BISHOP, Color::WHITE));
  const int white_rooks = popcount(get_piece_bb(PieceType::ROOK, Color::WHITE));
  const int white_queens =
      popcount(get_piece_bb(PieceType::QUEEN, Color::WHITE));
  const int white_pawns = popcount(get_piece_bb(PieceType::PAWN, Color::WHITE));

  const int black_knights =
      popcount(get_piece_bb(PieceType::KNIGHT, Color::BLACK));
  const int black_bishops =
      popcount(get_piece_bb(PieceType::BISHOP, Color::BLACK));
  const int black_rooks = popcount(get_piece_bb(PieceType::ROOK, Color::BLACK));
  const int black_queens =
      popcount(get_piece_bb(PieceType::QUEEN, Color::BLACK));
  const int black_pawns = popcount(get_piece_bb(PieceType::PAWN, Color::BLACK));

  // Early exit for common cases (most efficient branch ordering)
  if (white_pawns || black_pawns || white_rooks || black_rooks ||
      white_queens || black_queens) {
    return false;
  }

  // Pre-compute sums to avoid redundant calculations
  const int white_minors = white_knights + white_bishops;
  const int black_minors = black_knights + black_bishops;

  // King vs King - most common insufficient material case
  if (white_minors == 0 && black_minors == 0) return true;

  // King + minor vs King
  if ((white_minors == 1 && black_minors == 0) ||
      (white_minors == 0 && black_minors == 1))
    return true;

  // King + Knight vs King + Knight
  if (white_minors == 1 && black_minors == 1 && white_knights == 1 &&
      black_knights == 1)
    return true;

  return false;
}

// Efficient game state detection - checks in order of computational cost
GameState Board::is_game_over() const noexcept {
  // First generate legal moves (needed for both checkmate and stalemate)
  const std::vector<Move> legal_moves = generate_legal_moves();

  // If there are no legal moves, the game ends in either checkmate or stalemate
  // This takes precedence over other conditions
  if (legal_moves.empty()) {
    // Direct computation of current player using side_to_move bit
    const Color us = side_to_move ? Color::WHITE : Color::BLACK;

    // Use is_in_check to determine checkmate vs stalemate
    return is_in_check(*this, us) ? GameState::CHECKMATE : GameState::STALEMATE;
  }

  // Check for 50-move rule next
  if (halfmove_clock >= 100) {  // 50 moves = 100 half-moves
    return GameState::DRAW_BY_FIFTY_MOVE;
  }

  // Finally, check for insufficient material
  if (has_insufficient_material()) {
    return GameState::DRAW_BY_INSUFFICIENT_MATERIAL;
  }

  // If none of the above conditions are met, the game is ongoing
  return GameState::ONGOING;
}

void Board::apply_move(const Move& move, StateInfo* state) noexcept {
  const uint8_t from = move.from;
  const uint8_t to = move.to;
  const uint8_t promotion = move.promotion;

  const Bitboard from_bb = square_bitboard(from);
  const Bitboard to_bb = square_bitboard(to);

  PieceType moving_pt = PieceType::PAWN;
  for (uint8_t pt = 0; pt < NUM_PIECE_TYPES; ++pt) {
    if (piece_bitboards[pt] & from_bb) {
      moving_pt = static_cast<PieceType>(pt);
      break;
    }
  }

  const Color us =
      (color_bitboards[0] & from_bb) ? Color::WHITE : Color::BLACK;
  const Color them = (us == Color::WHITE) ? Color::BLACK : Color::WHITE;
  const uint8_t us_idx = static_cast<uint8_t>(us);
  const uint8_t them_idx = static_cast<uint8_t>(them);

  const int8_t old_ep = en_passant_square;

  bool capture = (color_bitboards[them_idx] & to_bb) != 0;
  bool en_passant_capture = false;
  uint8_t capture_square = to;
  PieceType captured_pt = PieceType::PAWN;

  if (moving_pt == PieceType::PAWN && old_ep != -1 &&
      to == static_cast<uint8_t>(old_ep) && !capture) {
    capture = true;
    en_passant_capture = true;
    capture_square = static_cast<uint8_t>(old_ep +
                                          (us == Color::WHITE ? -8 : 8));
  }

  if (capture) {
    if (en_passant_capture) {
      captured_pt = PieceType::PAWN;
    } else {
      Bitboard capture_bb = square_bitboard(capture_square);
      captured_pt = get_piece_at(piece_bitboards, capture_bb);
    }
  }

  if (state) {
    state->move = move;
    state->moving_piece = moving_pt;
    state->mover = us;
    state->was_capture = capture;
    state->captured_piece = captured_pt;
    state->captured_color = them;
    state->captured_square = capture ? capture_square : 64;
    state->was_en_passant = en_passant_capture;
    state->was_promotion = promotion != 0;
    state->was_castling =
        moving_pt == PieceType::KING &&
        std::abs(static_cast<int>(to) - static_cast<int>(from)) == 2;
    state->was_kingside_castle = state->was_castling && (to > from);
    state->previous_castling_rights = castling_rights;
    state->previous_en_passant_square = en_passant_square;
    state->previous_halfmove_clock = halfmove_clock;
    state->previous_fullmove_number = fullmove_number;
    state->previous_side_to_move = side_to_move;
  }

  // Reset halfmove clock if capture or pawn move
  halfmove_clock =
      (capture || moving_pt == PieceType::PAWN) ? 0 : static_cast<uint8_t>(halfmove_clock + 1);

  // Update fullmove number if Black moved
  if (us == Color::BLACK) ++fullmove_number;

  en_passant_square = -1;

  // Remove captured piece
  if (capture) {
    Bitboard capture_bb = square_bitboard(capture_square);
    piece_bitboards[static_cast<uint8_t>(captured_pt)] &= ~capture_bb;
    color_bitboards[them_idx] &= ~capture_bb;
  }

  // Move the piece
  piece_bitboards[static_cast<uint8_t>(moving_pt)] &= ~from_bb;
  color_bitboards[us_idx] &= ~from_bb;

  if (promotion != 0) {
    piece_bitboards[promotion] |= to_bb;
  } else {
    piece_bitboards[static_cast<uint8_t>(moving_pt)] |= to_bb;
  }
  color_bitboards[us_idx] |= to_bb;

  // Handle pawn specific logic
  if (moving_pt == PieceType::PAWN) {
    if (std::abs(static_cast<int>(to) - static_cast<int>(from)) == 16) {
      en_passant_square = from + (us == Color::WHITE ? 8 : -8);
    }
  }

  // Handle castling rook movement
  const bool castling =
      moving_pt == PieceType::KING &&
      std::abs(static_cast<int>(to) - static_cast<int>(from)) == 2;
  if (castling) {
    const bool kingside = to > from;
    const int8_t rook_from = kingside ? (us == Color::WHITE ? 7 : 63)
                                      : (us == Color::WHITE ? 0 : 56);
    const int8_t rook_to = kingside ? (from + 1) : (from - 1);
    const Bitboard rook_from_bb = square_bitboard(rook_from);
    const Bitboard rook_to_bb = square_bitboard(rook_to);
    const Bitboard rook_mask = rook_from_bb | rook_to_bb;
    piece_bitboards[static_cast<uint8_t>(PieceType::ROOK)] ^= rook_mask;
    color_bitboards[us_idx] ^= rook_mask;
  }

  // Update castling rights
  if (castling_rights) {
    if (moving_pt == PieceType::KING) {
      castling_rights &= (us == Color::WHITE ? ~0x03 : ~0x0C);
    } else if ((moving_pt == PieceType::ROOK &&
                (from == 0 || from == 7 || from == 56 || from == 63)) ||
               (capture && (capture_square == 0 || capture_square == 7 ||
                            capture_square == 56 || capture_square == 63))) {
      if (from == 0 || capture_square == 0)
        castling_rights &= ~0x02;
      if (from == 7 || capture_square == 7)
        castling_rights &= ~0x01;
      if (from == 56 || capture_square == 56)
        castling_rights &= ~0x08;
      if (from == 63 || capture_square == 63)
        castling_rights &= ~0x04;
    }
  }

  side_to_move = !side_to_move;
}

void Board::undo_move(const StateInfo& state) noexcept {
  const Move move = state.move;
  const uint8_t from = move.from;
  const uint8_t to = move.to;

  const Bitboard from_bb = square_bitboard(from);
  const Bitboard to_bb = square_bitboard(to);

  const Color us = state.mover;
  const Color them = (us == Color::WHITE) ? Color::BLACK : Color::WHITE;
  const uint8_t us_idx = static_cast<uint8_t>(us);
  const uint8_t them_idx = static_cast<uint8_t>(them);

  castling_rights = state.previous_castling_rights;
  en_passant_square = state.previous_en_passant_square;
  halfmove_clock = state.previous_halfmove_clock;
  fullmove_number = state.previous_fullmove_number;
  side_to_move = state.previous_side_to_move;

  // Remove moving piece from destination
  if (state.was_promotion) {
    piece_bitboards[move.promotion] &= ~to_bb;
  } else {
    piece_bitboards[static_cast<uint8_t>(state.moving_piece)] &= ~to_bb;
  }
  color_bitboards[us_idx] &= ~to_bb;

  // Restore moving piece on source square
  piece_bitboards[static_cast<uint8_t>(state.moving_piece)] |= from_bb;
  color_bitboards[us_idx] |= from_bb;

  // Restore captured piece
  if (state.was_capture && state.captured_square != 64) {
    const Bitboard capture_bb = square_bitboard(state.captured_square);
    piece_bitboards[static_cast<uint8_t>(state.captured_piece)] |= capture_bb;
    color_bitboards[them_idx] |= capture_bb;
  }

  // Undo castling rook move
  if (state.was_castling) {
    const bool kingside = state.was_kingside_castle;
    const int8_t rook_from = kingside ? (us == Color::WHITE ? 7 : 63)
                                      : (us == Color::WHITE ? 0 : 56);
    const int8_t rook_to = kingside ? (from + 1) : (from - 1);
    const Bitboard rook_mask =
        square_bitboard(rook_from) | square_bitboard(rook_to);
    piece_bitboards[static_cast<uint8_t>(PieceType::ROOK)] ^= rook_mask;
    color_bitboards[us_idx] ^= rook_mask;
  }
}

Move Board::parse_san(const std::string& san) const {
  std::string work;
  work.reserve(san.size());
  for (char c : san) {
    if (!std::isspace(static_cast<unsigned char>(c))) {
      work.push_back(c);
    }
  }
  if (work.empty()) {
    throw std::runtime_error("Empty SAN string");
  }

  while (!work.empty() &&
         (work.back() == '+' || work.back() == '#' || work.back() == '!' ||
          work.back() == '?')) {
    work.pop_back();
  }

  const auto legal_moves = generate_legal_moves();
  auto match_castle = [&](bool kingside) -> Move {
    for (const auto& mv : legal_moves) {
      if (is_castling(mv) && (mv.to > mv.from) == kingside) {
        return mv;
      }
    }
    throw std::runtime_error("No legal castling move for SAN: " + san);
  };

  if (work == "O-O" || work == "0-0") {
    return match_castle(true);
  }
  if (work == "O-O-O" || work == "0-0-0") {
    return match_castle(false);
  }

  uint8_t promotion = 0;
  auto eq_pos = work.find('=');
  if (eq_pos != std::string::npos) {
    if (eq_pos + 1 >= work.size()) {
      throw std::runtime_error("Invalid promotion SAN: " + san);
    }
    promotion =
        static_cast<uint8_t>(piece_type_from_char(work[eq_pos + 1]));
    work.erase(eq_pos);
  }

  if (work.size() < 2) {
    throw std::runtime_error("Invalid SAN: " + san);
  }

  const char target_file = work[work.size() - 2];
  const char target_rank = work[work.size() - 1];
  if (!is_file_char(target_file) || !is_rank_char(target_rank)) {
    throw std::runtime_error("Invalid target square in SAN: " + san);
  }
  const uint8_t target_square =
      static_cast<uint8_t>((target_rank - '1') * 8 + (target_file - 'a'));
  work.erase(work.size() - 2);

  const auto capture_pos = work.find('x');
  const bool capture = capture_pos != std::string::npos;
  if (capture) {
    work.erase(capture_pos, 1);
  }

  PieceType desired_piece = PieceType::PAWN;
  if (!work.empty() &&
      std::isupper(static_cast<unsigned char>(work.front()))) {
    desired_piece = piece_type_from_char(work.front());
    work.erase(work.begin());
  }

  std::optional<char> disamb_file;
  std::optional<char> disamb_rank;
  for (char c : work) {
    if (is_file_char(c)) disamb_file = c;
    if (is_rank_char(c)) disamb_rank = c;
  }

  std::optional<Move> candidate;
  for (const auto& mv : legal_moves) {
    auto piece = piece_at(mv.from);
    if (!piece || piece->type != desired_piece) continue;
    if (mv.to != target_square) continue;

    if (promotion != 0) {
      if (mv.promotion != promotion) continue;
    } else if (mv.promotion != 0) {
      continue;
    }

    if (capture && !is_capture(mv)) continue;
    if (!capture && is_capture(mv)) continue;

    const char from_file_char =
        static_cast<char>('a' + square_file(mv.from));
    const char from_rank_char =
        static_cast<char>('1' + square_rank(mv.from));
    if (disamb_file && from_file_char != *disamb_file) continue;
    if (disamb_rank && from_rank_char != *disamb_rank) continue;

    if (candidate) {
      throw std::runtime_error("Ambiguous SAN: " + san);
    }
    candidate = mv;
  }

  if (!candidate) {
    throw std::runtime_error("Illegal SAN: " + san);
  }
  return *candidate;
}

Move move_from_uci(const std::string& uci) {
  if (uci.size() < 4) {
    throw std::runtime_error("Invalid UCI string: " + uci);
  }
  auto to_square = [](char file, char rank) -> uint8_t {
    file = static_cast<char>(std::tolower(static_cast<unsigned char>(file)));
    if (!is_file_char(file) || !is_rank_char(rank)) {
      throw std::runtime_error("Invalid UCI square");
    }
    return static_cast<uint8_t>((rank - '1') * 8 + (file - 'a'));
  };
  uint8_t from = to_square(uci[0], uci[1]);
  uint8_t to = to_square(uci[2], uci[3]);

  uint8_t promotion = 0;
  if (uci.size() >= 5) {
    promotion = static_cast<uint8_t>(piece_type_from_char(uci[4]));
  }

  return Move{from, to, promotion};
}

std::string move_to_uci(const Move& move) {
  std::string uci;
  uci.reserve(5);
  uci.push_back('a' + square_file(move.from));
  uci.push_back('1' + square_rank(move.from));
  uci.push_back('a' + square_file(move.to));
  uci.push_back('1' + square_rank(move.to));

  if (move.promotion != 0) {
    static const char promo_map[] = {' ', 'n', 'b', 'r', 'q', 'k'};
    char promo_char =
        promo_map[std::min<size_t>(move.promotion, std::size(promo_map) - 1)];
    uci.push_back(promo_char);
  }
  return uci;
}
