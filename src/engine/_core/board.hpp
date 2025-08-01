#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

// Type aliases for cleaner code
using Bitboard = uint64_t;

// Constants
static constexpr uint8_t NUM_PIECE_TYPES = 6;
static constexpr uint8_t NUM_COLORS = 2;

// Enums
enum class Color : uint8_t { WHITE = 0, BLACK = 1 };

enum class PieceType : uint8_t {
  PAWN = 0,
  KNIGHT = 1,
  BISHOP = 2,
  ROOK = 3,
  QUEEN = 4,
  KING = 5
};

// Move structure - packed for memory efficiency
struct Move {
  uint8_t from;
  uint8_t to;
  uint8_t promotion;  // 0 if no promotion, otherwise PieceType value
};

// Board class
class Board {
 public:
  // Constructors and basic operations
  inline Board() noexcept;
  inline void clear() noexcept;
  void load_fen(const std::string& fen);
  static inline Board from_fen(const std::string& fen) noexcept;

  // Accessors (all const and noexcept for performance)
  [[nodiscard]] constexpr Bitboard get_piece_bb(PieceType pt,
                                                Color color) const noexcept {
    return piece_bitboards[static_cast<uint8_t>(pt)] &
           color_bitboards[static_cast<uint8_t>(color)];
  }

  [[nodiscard]] constexpr Bitboard get_piece_bb(PieceType pt) const noexcept {
    return piece_bitboards[static_cast<uint8_t>(pt)];
  }

  [[nodiscard]] constexpr Bitboard get_color_bb(Color color) const noexcept {
    return color_bitboards[static_cast<uint8_t>(color)];
  }

  [[nodiscard]] constexpr Bitboard get_all_pieces_bb() const noexcept {
    return color_bitboards[0] |
           color_bitboards[1];  // Direct indexing for speed
  }

  [[nodiscard]] constexpr Color side_to_move_color() const noexcept {
    return side_to_move ? Color::WHITE : Color::BLACK;
  }

  [[nodiscard]] constexpr bool get_side_to_move() const noexcept {
    return side_to_move;
  }

  [[nodiscard]] constexpr uint8_t get_castling_rights() const noexcept {
    return castling_rights;
  }

  [[nodiscard]] constexpr int8_t get_en_passant_square() const noexcept {
    return en_passant_square;
  }

  [[nodiscard]] constexpr uint8_t get_halfmove_clock() const noexcept {
    return halfmove_clock;
  }

  [[nodiscard]] constexpr uint16_t get_fullmove_number() const noexcept {
    return fullmove_number;
  }

  // String representations
  [[nodiscard]] std::string to_fen() const;
  [[nodiscard]] std::string pretty() const;

  // Move generation and game logic
  [[nodiscard]] std::vector<Move> generate_legal_moves() const noexcept;

  // Moved make_move implementation from inline declaration to inline definition
  inline void make_move(const Move& move) noexcept;

  // Feature extraction (placeholder)
  [[nodiscard]] std::vector<float> to_half_kp_features() const;

 private:
  // Bitboard representation - aligned for better cache performance
  alignas(16) std::array<Bitboard, NUM_PIECE_TYPES> piece_bitboards{};
  alignas(16) std::array<Bitboard, NUM_COLORS> color_bitboards{};

  // Game state
  bool side_to_move = true;       // true = WHITE, false = BLACK
  uint8_t castling_rights = 0;    // Bitmask: 1=K, 2=Q, 4=k, 8=q
  int8_t en_passant_square = -1;  // -1 if no en passant
  uint8_t halfmove_clock = 0;     // For 50-move rule
  uint16_t fullmove_number = 1;   // Increments after Black's move
};

// Helper functions for move formatting (made inline with definitions)
inline std::string move_to_string(const Move& move, const Board& board) {
  static const char files[] = "abcdefgh";
  static const char ranks[] = "12345678";
  static const char promo[] =
      " nbrq";  // Index 0 is empty, 1-4 maps to Knight-Queen

  std::string result;
  result.reserve(5);  // "e2e4q" worst case

  result += files[move.from % 8];
  result += ranks[move.from / 8];
  result += files[move.to % 8];
  result += ranks[move.to / 8];

  if (move.promotion != 0) {
    result += promo[move.promotion];
  }

  return result;
}

inline std::string move_debug_string(const Move& move, const Board& board) {
  return move_to_string(move, board);
}

inline std::string moves_to_string(const std::vector<Move>& moves,
                                   const Board& board) {
  std::string result;
  result.reserve(moves.size() * 10);  // Estimate size to avoid reallocations

  result = "Moves [" + std::to_string(moves.size()) + "]:\n";

  for (size_t i = 0; i < moves.size(); ++i) {
    result += "  " + std::to_string(i) + ": " +
              move_to_string(moves[i], board) + "\n";
  }

  return result;
}

// Inline implementation of simple methods
inline Board::Board() noexcept { clear(); }

inline void Board::clear() noexcept {
  for (auto& bb : piece_bitboards) bb = 0ULL;
  for (auto& bb : color_bitboards) bb = 0ULL;
  side_to_move = true;  // WHITE
  castling_rights = 0;
  en_passant_square = -1;
  halfmove_clock = 0;
  fullmove_number = 1;
}

inline Board Board::from_fen(const std::string& fen) noexcept {
  Board board;
  board.load_fen(fen);
  return board;
}

// Make move - inline implementation
inline void Board::make_move(const Move& move) noexcept {
  uint8_t from = move.from;
  uint8_t to = move.to;
  uint8_t promotion = move.promotion;

  Bitboard from_bb = 1ULL << from;
  Bitboard to_bb = 1ULL << to;

  // Initialize moving_pt with a proper default value
  PieceType moving_pt = PieceType::PAWN;  // Safe default
  // Find the actual piece type
  for (uint8_t pt = 0; pt < NUM_PIECE_TYPES; ++pt) {
    if (piece_bitboards[pt] & from_bb) {
      moving_pt = static_cast<PieceType>(pt);
      break;
    }
  }

  Color us = (color_bitboards[0] & from_bb) ? Color::WHITE : Color::BLACK;
  uint8_t us_idx = static_cast<uint8_t>(us);
  uint8_t them_idx = us_idx ^ 1;

  // Handle capture - direct bitboard operations
  bool is_capture = (to_bb & color_bitboards[them_idx]);

  // Reset halfmove clock if capture or pawn move
  halfmove_clock =
      (is_capture || moving_pt == PieceType::PAWN) ? 0 : halfmove_clock + 1;

  // Update fullmove number if Black moved
  if (us == Color::BLACK) ++fullmove_number;

  // Save old en passant for restoration if needed
  int8_t old_ep = en_passant_square;
  en_passant_square = -1;

  // For each piece type, remove captured piece (if any)
  if (is_capture) {
    for (uint8_t pt = 0; pt < NUM_PIECE_TYPES; ++pt) {
      piece_bitboards[pt] &= ~(to_bb & color_bitboards[them_idx]);
    }
    color_bitboards[them_idx] &= ~to_bb;
  }

  // Move the piece - update from square
  piece_bitboards[static_cast<uint8_t>(moving_pt)] &= ~from_bb;
  color_bitboards[us_idx] &= ~from_bb;

  // Handle promotion
  if (promotion != 0) {
    piece_bitboards[promotion] |= to_bb;
  } else {
    piece_bitboards[static_cast<uint8_t>(moving_pt)] |= to_bb;
  }

  // Update destination square
  color_bitboards[us_idx] |= to_bb;

  // Special cases - en passant capture
  if (moving_pt == PieceType::PAWN) {
    if (old_ep != -1 && to == old_ep) {
      // En passant capture - remove captured pawn
      int8_t cap_square = old_ep + (us == Color::WHITE ? -8 : 8);
      Bitboard cap_bb = 1ULL << cap_square;
      piece_bitboards[static_cast<uint8_t>(PieceType::PAWN)] &= ~cap_bb;
      color_bitboards[them_idx] &= ~cap_bb;
    } else if (std::abs(static_cast<int>(to) - static_cast<int>(from)) == 16) {
      // Double pawn push - set en passant square
      en_passant_square = from + (us == Color::WHITE ? 8 : -8);
    }
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
    Bitboard rook_from_to_bb = rook_from_bb | rook_to_bb;

    // Move rook with minimal operations
    piece_bitboards[static_cast<uint8_t>(PieceType::ROOK)] ^= rook_from_to_bb;
    color_bitboards[us_idx] ^= rook_from_to_bb;
  }

  // Update castling rights - optimized bitwise operations
  if (castling_rights) {
    // King moved - lose all castling rights for that side
    if (moving_pt == PieceType::KING) {
      castling_rights &= (us == Color::WHITE ? ~0x03 : ~0x0C);
    }
    // Rook moved or captured
    else if ((moving_pt == PieceType::ROOK &&
              (from == 0 || from == 7 || from == 56 || from == 63)) ||
             (is_capture && (to == 0 || to == 7 || to == 56 || to == 63))) {
      // Update specific castling right bits based on square
      if (from == 0 || to == 0)
        castling_rights &= ~0x02;  // a1 - White queenside
      else if (from == 7 || to == 7)
        castling_rights &= ~0x01;  // h1 - White kingside
      else if (from == 56 || to == 56)
        castling_rights &= ~0x08;  // a8 - Black queenside
      else if (from == 63 || to == 63)
        castling_rights &= ~0x04;  // h8 - Black kingside
    }
  }

  // Switch side to move
  side_to_move = !side_to_move;
}