#include "board.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <sstream>
#include <vector>

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