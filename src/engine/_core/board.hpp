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

// Move structure
struct Move {
  uint8_t from;
  uint8_t to;
  uint8_t promotion;  // 0 if no promotion, otherwise PieceType value
};

// Board class
class Board {
 public:
  // Constructors and basic operations
  Board() noexcept;
  void clear() noexcept;
  void load_fen(const std::string& fen);
  static Board from_fen(const std::string& fen);

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
    return color_bitboards[static_cast<uint8_t>(Color::WHITE)] |
           color_bitboards[static_cast<uint8_t>(Color::BLACK)];
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

  // Move generation and game logic (declared here, defined in movegen.cpp)
  [[nodiscard]] std::vector<Move> generate_legal_moves() const noexcept;
  void make_move(const Move& move) noexcept;

  // Feature extraction (placeholder)
  [[nodiscard]] std::vector<float> to_half_kp_features() const;

 private:
  // Bitboard representation
  std::array<Bitboard, NUM_PIECE_TYPES> piece_bitboards{};
  std::array<Bitboard, NUM_COLORS> color_bitboards{};

  // Game state
  bool side_to_move = true;       // true = WHITE, false = BLACK
  uint8_t castling_rights = 0;    // Bitmask: 1=K, 2=Q, 4=k, 8=q
  int8_t en_passant_square = -1;  // -1 if no en passant
  uint8_t halfmove_clock = 0;     // For 50-move rule
  uint16_t fullmove_number = 1;   // Increments after Black's move
};

// Helper functions for move formatting
std::string move_to_string(const Move& move, const Board& board);
std::string move_debug_string(const Move& move, const Board& board);
std::string moves_to_string(const std::vector<Move>& moves, const Board& board);
