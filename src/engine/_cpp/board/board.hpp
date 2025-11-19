#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

// Type aliases for cleaner code
using Bitboard = uint64_t;

// Constants
static constexpr uint8_t NUM_PIECE_TYPES = 6;
static constexpr uint8_t NUM_COLORS = 2;
static constexpr const char* STARTING_FEN =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

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

// Game state enum for is_game_over()
enum class GameState : uint8_t {
  ONGOING = 0,
  CHECKMATE = 1,
  STALEMATE = 2,
  DRAW_BY_FIFTY_MOVE = 3,
  DRAW_BY_INSUFFICIENT_MATERIAL = 4,
  DRAW_BY_REPETITION = 5
};

// Move structure - packed for memory efficiency
struct Move {
  uint8_t from;
  uint8_t to;
  uint8_t promotion;  // 0 if no promotion, otherwise PieceType value
};

struct Piece {
  PieceType type{PieceType::PAWN};
  Color color{Color::WHITE};
  bool valid{false};

  [[nodiscard]] char symbol() const noexcept;
};

struct StateInfo {
  Move move{};
  PieceType moving_piece{PieceType::PAWN};
  Color mover{Color::WHITE};
  bool was_capture{false};
  PieceType captured_piece{PieceType::PAWN};
  Color captured_color{Color::BLACK};
  uint8_t captured_square{64};
  bool was_en_passant{false};
  bool was_castling{false};
  bool was_kingside_castle{false};
  bool was_promotion{false};
  uint8_t previous_castling_rights{0};
  int8_t previous_en_passant_square{-1};
  uint8_t previous_halfmove_clock{0};
  uint16_t previous_fullmove_number{1};
  bool previous_side_to_move{true};
};

// Helpers for square metadata
constexpr uint8_t square_file(uint8_t square) noexcept { return square % 8; }
constexpr uint8_t square_rank(uint8_t square) noexcept { return square / 8; }
constexpr Bitboard square_bitboard(uint8_t square) noexcept {
  return 1ULL << square;
}

constexpr std::array<uint8_t, 64> SQUARES = []() constexpr {
  std::array<uint8_t, 64> squares{};
  for (uint8_t i = 0; i < 64; ++i) squares[i] = i;
  return squares;
}();

constexpr std::array<PieceType, NUM_PIECE_TYPES> PIECE_TYPES_ARRAY = {
    PieceType::PAWN, PieceType::KNIGHT, PieceType::BISHOP,
    PieceType::ROOK, PieceType::QUEEN,  PieceType::KING};

constexpr Bitboard BB_A1 = 1ULL << 0;
constexpr Bitboard BB_H1 = 1ULL << 7;
constexpr Bitboard BB_A8 = 1ULL << 56;
constexpr Bitboard BB_H8 = 1ULL << 63;

// Board class
class Board {
 public:
  // Constructors and basic operations
  inline Board();
  inline void clear() noexcept;
  void load_fen(const std::string& fen);
  static inline Board from_fen(const std::string& fen) noexcept;

  // Copy method
  [[nodiscard]] inline Board copy() const noexcept;

  // Game state checking
  [[nodiscard]] GameState is_game_over() const noexcept;

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
  void push(const Move& move);
  Move pop();
  Move push_san(const std::string& san);

  [[nodiscard]] bool is_capture(const Move& move) const noexcept;
  [[nodiscard]] bool is_castling(const Move& move) const noexcept;
  [[nodiscard]] bool is_kingside_castling(const Move& move) const noexcept;
  [[nodiscard]] bool is_queenside_castling(const Move& move) const noexcept;
  [[nodiscard]] bool is_en_passant(const Move& move) const noexcept;
  [[nodiscard]] bool is_check() const noexcept;

  [[nodiscard]] std::optional<Piece> piece_at(uint8_t square) const noexcept;
  [[nodiscard]] std::vector<uint8_t> pieces(PieceType pt,
                                            Color color) const noexcept;
  [[nodiscard]] std::optional<uint8_t> king(Color color) const noexcept;

  [[nodiscard]] bool has_kingside_castling_rights(Color color) const noexcept;
  [[nodiscard]] bool has_queenside_castling_rights(Color color) const noexcept;

  [[nodiscard]] bool turn() const noexcept { return side_to_move; }
  [[nodiscard]] std::optional<uint8_t> ep_square() const noexcept;

  void set_fen(const std::string& fen) { load_fen(fen); }
  [[nodiscard]] std::string fen() const { return to_fen(); }

  // Feature extraction (placeholder)
  [[nodiscard]] std::vector<float> to_half_kp_features() const;

  [[nodiscard]] std::string print_move(const Move& move) const;

 private:
  // Helper method for insufficient material detection
  [[nodiscard]] bool has_insufficient_material() const noexcept;
  void apply_move(const Move& move, StateInfo* state) noexcept;
  void undo_move(const StateInfo& state) noexcept;
  [[nodiscard]] Move parse_san(const std::string& san) const;

  // Bitboard representation - aligned for better cache performance
  alignas(16) std::array<Bitboard, NUM_PIECE_TYPES> piece_bitboards{};
  alignas(16) std::array<Bitboard, NUM_COLORS> color_bitboards{};

  // Game state
  bool side_to_move = true;       // true = WHITE, false = BLACK
  uint8_t castling_rights = 0;    // Bitmask: 1=K, 2=Q, 4=k, 8=q
  int8_t en_passant_square = -1;  // -1 if no en passant
  uint8_t halfmove_clock = 0;     // For 50-move rule
  uint16_t fullmove_number = 1;   // Increments after Black's move
  std::vector<StateInfo> state_history;
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

Move move_from_uci(const std::string& uci);
std::string move_to_uci(const Move& move);

// Inline implementation of simple methods
inline Board::Board() { load_fen(STARTING_FEN); }

inline void Board::clear() noexcept {
  for (auto& bb : piece_bitboards) bb = 0ULL;
  for (auto& bb : color_bitboards) bb = 0ULL;
  side_to_move = true;  // WHITE
  castling_rights = 0;
  en_passant_square = -1;
  halfmove_clock = 0;
  fullmove_number = 1;
  state_history.clear();
}

inline Board Board::from_fen(const std::string& fen) noexcept {
  Board board;
  board.load_fen(fen);
  return board;
}

// Copy method implementation
inline Board Board::copy() const noexcept {
  return *this;  // Uses default copy constructor
}

// Make move - inline implementation
inline void Board::make_move(const Move& move) noexcept {
  apply_move(move, nullptr);
}