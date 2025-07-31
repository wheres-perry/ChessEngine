#include "board.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "movegen.hpp"  // Include this to access move generation functions

// Helper function to get the piece type at a square
[[nodiscard]] inline PieceType get_piece_at(
    const std::array<Bitboard, NUM_PIECE_TYPES> &piece_bbs,
    Bitboard square_bb) noexcept {
  for (uint8_t pt = 0; pt < NUM_PIECE_TYPES; ++pt) {
    if (piece_bbs[pt] & square_bb) {
      return static_cast<PieceType>(pt);
    }
  }
  return static_cast<PieceType>(NUM_PIECE_TYPES);  // Sentinel for empty
}

// Helper to get color at a square (assumes occupied)
[[nodiscard]] inline Color get_color_at(
    const std::array<Bitboard, NUM_COLORS> &color_bbs,
    Bitboard square_bb) noexcept {
  return (color_bbs[static_cast<uint8_t>(Color::WHITE)] & square_bb)
             ? Color::WHITE
             : Color::BLACK;
}

// Constructor
Board::Board() noexcept { clear(); }

// Clear board
void Board::clear() noexcept {
  for (auto &bb : piece_bitboards) bb = 0ULL;
  for (auto &bb : color_bitboards) bb = 0ULL;
  side_to_move = true;  // WHITE
  castling_rights = 0;
  en_passant_square = -1;
  halfmove_clock = 0;
  fullmove_number = 1;
}

// Load FEN string
void Board::load_fen(const std::string &fen) {
  clear();
  std::stringstream ss(fen);
  std::string piece_placement, active_color, castling, en_passant, halfmove,
      fullmove;
  ss >> piece_placement >> active_color >> castling >> en_passant >> halfmove >>
      fullmove;

  int8_t rank = 7, file = 0;
  for (const char c : piece_placement) {
    if (c == '/') {
      rank--;
      file = 0;
    } else if (isdigit(c)) {
      file += c - '0';
    } else {
      uint8_t square = rank * 8 + file;
      Color color = isupper(c) ? Color::WHITE : Color::BLACK;
      PieceType pt;
      switch (tolower(c)) {
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
          throw std::runtime_error("Invalid piece in FEN string");
      }
      piece_bitboards[static_cast<uint8_t>(pt)] |= (1ULL << square);
      color_bitboards[static_cast<uint8_t>(color)] |= (1ULL << square);
      file++;
    }
  }

  side_to_move = (active_color == "w");

  castling_rights = 0;
  for (const char c : castling) {
    if (c == 'K')
      castling_rights |= 1;
    else if (c == 'Q')
      castling_rights |= 2;
    else if (c == 'k')
      castling_rights |= 4;
    else if (c == 'q')
      castling_rights |= 8;
  }

  if (en_passant != "-") {
    int8_t ep_file = en_passant[0] - 'a';
    int8_t ep_rank = en_passant[1] - '1';
    en_passant_square = ep_rank * 8 + ep_file;
  } else {
    en_passant_square = -1;
  }

  try {
    halfmove_clock = std::stoi(halfmove);
    fullmove_number = std::stoi(fullmove);
  } catch (const std::exception &e) {
    throw std::runtime_error("Invalid clock value in FEN string");
  }
}

// Create board from FEN
Board Board::from_fen(const std::string &fen) {
  Board board;
  board.load_fen(fen);
  return board;
}

// Convert to FEN string
std::string Board::to_fen() const {
  std::stringstream ss;

  // Piece placement
  for (int8_t rank = 7; rank >= 0; --rank) {
    uint8_t empty_squares = 0;
    for (int8_t file = 0; file < 8; ++file) {
      uint8_t square = rank * 8 + file;
      Bitboard square_bb = 1ULL << square;
      char piece_char = 0;

      for (uint8_t pt_idx = 0; pt_idx < NUM_PIECE_TYPES; ++pt_idx) {
        if (piece_bitboards[pt_idx] & square_bb) {
          bool is_white = (color_bitboards[static_cast<uint8_t>(Color::WHITE)] &
                           square_bb) != 0;
          switch (static_cast<PieceType>(pt_idx)) {
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
          break;
        }
      }

      if (piece_char != 0) {
        if (empty_squares > 0) {
          ss << static_cast<char>('0' + empty_squares);
          empty_squares = 0;
        }
        ss << piece_char;
      } else {
        empty_squares++;
      }
    }
    if (empty_squares > 0) {
      ss << static_cast<char>('0' + empty_squares);
    }
    if (rank > 0) {
      ss << '/';
    }
  }

  // Active color
  ss << ' ' << (side_to_move ? 'w' : 'b');

  // Castling availability
  ss << ' ';
  std::string castling_str;
  if (castling_rights & 1) castling_str += 'K';
  if (castling_rights & 2) castling_str += 'Q';
  if (castling_rights & 4) castling_str += 'k';
  if (castling_rights & 8) castling_str += 'q';
  ss << (castling_str.empty() ? "-" : castling_str);

  // En passant target square
  ss << ' ';
  if (en_passant_square == -1) {
    ss << '-';
  } else {
    char file = 'a' + (en_passant_square % 8);
    char rank = '1' + (en_passant_square / 8);
    ss << file << rank;
  }

  // Halfmove clock and fullmove number
  ss << ' ' << static_cast<int>(halfmove_clock);
  ss << ' ' << fullmove_number;

  return ss.str();
}

// Pretty print board
std::string Board::pretty() const {
  std::stringstream ss;

  for (int8_t rank = 7; rank >= 0; --rank) {
    ss << (rank + 1) << "  ";
    for (int8_t file = 0; file < 8; ++file) {
      uint8_t square = rank * 8 + file;
      Bitboard square_bb = 1ULL << square;
      char piece_char = '.';

      for (uint8_t pt_idx = 0; pt_idx < NUM_PIECE_TYPES; ++pt_idx) {
        if (piece_bitboards[pt_idx] & square_bb) {
          bool is_white = (color_bitboards[static_cast<uint8_t>(Color::WHITE)] &
                           square_bb) != 0;
          switch (static_cast<PieceType>(pt_idx)) {
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
          break;
        }
      }
      ss << piece_char << ' ';
    }
    ss << '\n';
  }
  ss << "\n   a b c d e f g h\n";
  return ss.str();
}

// Make move
void Board::make_move(const Move &move) noexcept {
  uint8_t from = move.from;
  uint8_t to = move.to;
  uint8_t promotion = move.promotion;

  Bitboard from_bb = 1ULL << from;
  Bitboard to_bb = 1ULL << to;

  // Determine moving piece and colors
  PieceType moving_pt = get_piece_at(piece_bitboards, from_bb);
  Color us = get_color_at(color_bitboards, from_bb);
  Color them = (us == Color::WHITE) ? Color::BLACK : Color::WHITE;

  // Handle capture
  bool is_capture = (get_all_pieces_bb() & to_bb) != 0;
  PieceType captured_pt = is_capture ? get_piece_at(piece_bitboards, to_bb)
                                     : static_cast<PieceType>(NUM_PIECE_TYPES);

  // Reset halfmove clock if capture or pawn move
  halfmove_clock =
      (is_capture || moving_pt == PieceType::PAWN) ? 0 : halfmove_clock + 1;

  // Update fullmove number if Black moved
  if (us == Color::BLACK) ++fullmove_number;

  // Clear en passant
  en_passant_square = -1;

  // Move the piece
  piece_bitboards[static_cast<uint8_t>(moving_pt)] &= ~from_bb;
  color_bitboards[static_cast<uint8_t>(us)] &= ~from_bb;

  PieceType target_pt =
      (promotion != 0) ? static_cast<PieceType>(promotion) : moving_pt;
  piece_bitboards[static_cast<uint8_t>(target_pt)] |= to_bb;
  color_bitboards[static_cast<uint8_t>(us)] |= to_bb;

  // Remove captured piece
  if (is_capture) {
    piece_bitboards[static_cast<uint8_t>(captured_pt)] &= ~to_bb;
    color_bitboards[static_cast<uint8_t>(them)] &= ~to_bb;
  }

  // Special cases
  // En passant capture
  if (moving_pt == PieceType::PAWN && to == en_passant_square) {
    int8_t capture_rank_offset = (us == Color::WHITE) ? -8 : 8;
    Bitboard ep_capture_bb = 1ULL << (to + capture_rank_offset);
    piece_bitboards[static_cast<uint8_t>(PieceType::PAWN)] &= ~ep_capture_bb;
    color_bitboards[static_cast<uint8_t>(them)] &= ~ep_capture_bb;
  } else if (moving_pt == PieceType::PAWN &&
             std::abs(static_cast<int>(to) - static_cast<int>(from)) == 16) {
    // Set en passant square for double pawn push
    int8_t ep_offset = (us == Color::WHITE) ? -8 : 8;
    en_passant_square = from + ep_offset;
  }

  // Castling
  if (moving_pt == PieceType::KING &&
      std::abs(static_cast<int>(to) - static_cast<int>(from)) == 2) {
    bool kingside = (to > from);
    int8_t rook_from = kingside ? (us == Color::WHITE ? 7 : 63)
                                : (us == Color::WHITE ? 0 : 56);
    int8_t rook_to = kingside ? (from + 1) : (from - 1);

    Bitboard rook_from_bb = 1ULL << rook_from;
    Bitboard rook_to_bb = 1ULL << rook_to;

    piece_bitboards[static_cast<uint8_t>(PieceType::ROOK)] &= ~rook_from_bb;
    color_bitboards[static_cast<uint8_t>(us)] &= ~rook_from_bb;

    piece_bitboards[static_cast<uint8_t>(PieceType::ROOK)] |= rook_to_bb;
    color_bitboards[static_cast<uint8_t>(us)] |= rook_to_bb;
  }

  // Update castling rights
  if (castling_rights != 0) {
    if (moving_pt == PieceType::KING) {
      castling_rights &= (us == Color::WHITE) ? ~3 : ~12;
    } else if (moving_pt == PieceType::ROOK ||
               (is_capture && captured_pt == PieceType::ROOK)) {
      if (from == 0 || to == 0) castling_rights &= ~2;    // White queenside
      if (from == 7 || to == 7) castling_rights &= ~1;    // White kingside
      if (from == 56 || to == 56) castling_rights &= ~8;  // Black queenside
      if (from == 63 || to == 63) castling_rights &= ~4;  // Black kingside
    }
  }

  // Switch side to move
  side_to_move = !side_to_move;
}

// Placeholder for feature extraction
std::vector<float> Board::to_half_kp_features() const { return {}; }

// Move formatting functions
std::string move_to_string(const Move &move, const Board &board) {
  auto square_to_algebraic = [](uint8_t square) -> std::string {
    if (square >= 64) return "??";
    char file = 'a' + (square % 8);
    char rank = '1' + (square / 8);
    return std::string(1, file) + std::string(1, rank);
  };

  std::stringstream ss;
  ss << square_to_algebraic(move.from) << square_to_algebraic(move.to);
  if (move.promotion != 0) {
    switch (static_cast<PieceType>(move.promotion)) {
      case PieceType::QUEEN:
        ss << "q";
        break;
      case PieceType::ROOK:
        ss << "r";
        break;
      case PieceType::BISHOP:
        ss << "b";
        break;
      case PieceType::KNIGHT:
        ss << "n";
        break;
      default:
        break;
    }
  }
  return ss.str();
}

std::string move_debug_string(const Move &move, const Board &board) {
  return move_to_string(move, board);
}

std::string moves_to_string(const std::vector<Move> &moves,
                            const Board &board) {
  std::stringstream ss;
  ss << "Moves [" << moves.size() << "]:\n";
  for (size_t i = 0; i < moves.size(); ++i) {
    ss << "  " << i << ": " << move_to_string(moves[i], board) << "\n";
  }
  return ss.str();
}
