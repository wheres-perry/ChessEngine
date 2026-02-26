#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

using Bitboard = uint64_t;

#if defined(_MSC_VER)
#include <intrin.h>
#endif

// ---------------------------------------------------------------------------
// Low-level bit manipulation — all functions are branchless hardware intrinsics
// ---------------------------------------------------------------------------

inline uint8_t ctz64(Bitboard bb) noexcept {
#if defined(_MSC_VER)
  unsigned long idx{};
  _BitScanForward64(&idx, bb);
  return static_cast<uint8_t>(idx);
#else
  return static_cast<uint8_t>(__builtin_ctzll(bb));
#endif
}

inline int popcount64(Bitboard bb) noexcept {
#if defined(_MSC_VER)
  return static_cast<int>(__popcnt64(bb));
#else
  return __builtin_popcountll(bb);
#endif
}

inline uint8_t clz64(Bitboard bb) noexcept {
#if defined(_MSC_VER)
  unsigned long idx{};
  _BitScanReverse64(&idx, bb);
  return static_cast<uint8_t>(63 - idx);
#else
  return static_cast<uint8_t>(__builtin_clzll(bb));
#endif
}

// Pops and returns the index of the least-significant set bit.
inline uint8_t pop_lsb(Bitboard &bb) noexcept {
  const uint8_t sq = ctz64(bb);
  bb &= bb - 1;
  return sq;
}

inline int popcount(Bitboard bb) noexcept { return popcount64(bb); }

// ---------------------------------------------------------------------------
// Fundamental constants
// ---------------------------------------------------------------------------

static constexpr uint8_t NUM_PIECE_TYPES = 6;
static constexpr uint8_t NUM_COLORS = 2;

// Sentinel value stored in the mailbox arrays for an unoccupied square.
static constexpr uint8_t EMPTY_SQ = 0xFF;

static constexpr const char *STARTING_FEN =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

// ---------------------------------------------------------------------------
// Enumerations
// ---------------------------------------------------------------------------

enum class Color : uint8_t { WHITE = 0, BLACK = 1 };

enum class PieceType : uint8_t {
  PAWN = 0,
  KNIGHT = 1,
  BISHOP = 2,
  ROOK = 3,
  QUEEN = 4,
  KING = 5
};

enum class GameState : uint8_t {
  ONGOING = 0,
  CHECKMATE = 1,
  STALEMATE = 2,
  DRAW_BY_FIFTY_MOVE = 3,
  DRAW_BY_INSUFFICIENT_MATERIAL = 4,
  DRAW_BY_REPETITION = 5
};

// ---------------------------------------------------------------------------
// Core data structures
// ---------------------------------------------------------------------------

// Packed move: 3 bytes.  promotion == 0 means no promotion; values 1-4 map to
// KNIGHT-QUEEN following the PieceType enum.
struct Move {
  uint8_t from;
  uint8_t to;
  uint8_t promotion;

  constexpr bool operator==(const Move &o) const noexcept {
    return from == o.from && to == o.to && promotion == o.promotion;
  }
  constexpr bool operator!=(const Move &o) const noexcept {
    return !(*this == o);
  }
};

struct Piece {
  PieceType type{PieceType::PAWN};
  Color color{Color::WHITE};
  bool valid{false};

  [[nodiscard]] char symbol() const noexcept;
};

// All information needed to reversibly undo a move.
struct StateInfo {
  Move move{};
  PieceType moving_piece{PieceType::PAWN};
  Color mover{Color::WHITE};
  PieceType captured_piece{PieceType::PAWN};
  Color captured_color{Color::BLACK};
  uint8_t captured_square{64};
  uint8_t previous_castling_rights{0};
  int8_t previous_en_passant_square{-1};
  uint8_t previous_halfmove_clock{0};
  uint16_t previous_fullmove_number{1};

  // Flags packed into a single byte via bitfields.
  bool was_capture : 1 {false};
  bool was_en_passant : 1 {false};
  bool was_castling : 1 {false};
  bool was_kingside_castle : 1 {false};
  bool was_promotion : 1 {false};
  bool previous_side_to_move : 1 {true};
  bool is_null_move : 1 {false};
};

// ---------------------------------------------------------------------------
// Square helper functions
// ---------------------------------------------------------------------------

constexpr uint8_t square_file(uint8_t square) noexcept { return square % 8; }
constexpr uint8_t square_rank(uint8_t square) noexcept { return square / 8; }
constexpr Bitboard square_bitboard(uint8_t square) noexcept {
  return 1ULL << square;
}

constexpr std::array<uint8_t, 64> SQUARES = []() constexpr {
  std::array<uint8_t, 64> s{};
  for (uint8_t i = 0; i < 64; ++i)
    s[i] = i;
  return s;
}();

constexpr std::array<PieceType, NUM_PIECE_TYPES> PIECE_TYPES_ARRAY = {
    PieceType::PAWN, PieceType::KNIGHT, PieceType::BISHOP,
    PieceType::ROOK, PieceType::QUEEN,  PieceType::KING};

constexpr Bitboard BB_A1 = 1ULL << 0;
constexpr Bitboard BB_H1 = 1ULL << 7;
constexpr Bitboard BB_A8 = 1ULL << 56;
constexpr Bitboard BB_H8 = 1ULL << 63;

// ---------------------------------------------------------------------------
// Board
// ---------------------------------------------------------------------------

class Board {
public:
  // Construction / copy
  inline Board();
  inline void clear() noexcept;
  void load_fen(const std::string &fen);
  static inline Board from_fen(const std::string &fen) noexcept;
  [[nodiscard]] inline Board copy() const noexcept;

  // Accessors — all constexpr/noexcept for zero-overhead callers
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
    return color_bitboards[0] | color_bitboards[1];
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
  [[nodiscard]] inline std::string fen() const { return to_fen(); }
  inline void set_fen(const std::string &f) { load_fen(f); }

  // Move execution
  inline void make_move(const Move &move) noexcept;
  void push(const Move &move);
  void push_null();
  [[nodiscard]] Move pop();
  [[nodiscard]] Move push_san(const std::string &san);

  // Move classification
  [[nodiscard]] bool is_capture(const Move &move) const noexcept;
  [[nodiscard]] bool is_castling(const Move &move) const noexcept;
  [[nodiscard]] bool is_kingside_castling(const Move &move) const noexcept;
  [[nodiscard]] bool is_queenside_castling(const Move &move) const noexcept;
  [[nodiscard]] bool is_en_passant(const Move &move) const noexcept;
  [[nodiscard]] bool is_check() const noexcept;

  // Piece queries — O(1) via mailbox
  [[nodiscard]] std::optional<Piece> piece_at(uint8_t square) const noexcept;
  [[nodiscard]] std::vector<uint8_t> pieces(PieceType pt,
                                            Color color) const noexcept;
  [[nodiscard]] std::optional<uint8_t> king(Color color) const noexcept;

  // Castling rights
  [[nodiscard]] bool has_kingside_castling_rights(Color color) const noexcept;
  [[nodiscard]] bool has_queenside_castling_rights(Color color) const noexcept;

  // Move generation
  [[nodiscard]] std::vector<Move> generate_legal_moves() const noexcept;
  [[nodiscard]] uint64_t perft(int depth) noexcept;

  // Game-over detection
  [[nodiscard]] GameState is_game_over() const noexcept;

  // Attacked-squares cache (lazy, invalidated on every move)
  [[nodiscard]] Bitboard get_attacked_squares(Color color) const noexcept;

  [[nodiscard]] std::string print_move(const Move &move) const;

  [[nodiscard]] inline size_t move_stack_size() const noexcept {
    return state_history.size();
  }

private:
  [[nodiscard]] bool has_insufficient_material() const noexcept;
  void apply_move(const Move &move, StateInfo *state) noexcept;
  void undo_move(const StateInfo &state) noexcept;
  [[nodiscard]] Move parse_san(const std::string &san) const;
  void update_attacked_squares(Color color) const noexcept;

  // ---------------------------------------------------------------------------
  // Board representation
  // ---------------------------------------------------------------------------

  // Bitboard layers — 16-byte aligned for SIMD-friendly access.
  alignas(16) std::array<Bitboard, NUM_PIECE_TYPES> piece_bitboards{};
  alignas(16) std::array<Bitboard, NUM_COLORS> color_bitboards{};

  // Mailbox arrays for O(1) piece-at-square lookup.
  // piece_on[sq] holds the raw PieceType index (0-5) or EMPTY_SQ (0xFF).
  // color_on[sq] holds the raw Color index (0-1); only valid when piece_on[sq]
  // != EMPTY_SQ.
  alignas(64) uint8_t piece_on[64];
  alignas(64) uint8_t color_on[64];

  // ---------------------------------------------------------------------------
  // Game state
  // ---------------------------------------------------------------------------

  bool side_to_move = true;      // true == WHITE
  uint8_t castling_rights = 0;   // bit 0=K, 1=Q, 2=k, 3=q
  int8_t en_passant_square = -1; // -1 when absent
  uint8_t halfmove_clock = 0;
  uint16_t fullmove_number = 1;
  std::vector<StateInfo> state_history;

  // Lazily computed attacked-square bitboards; invalidated after each move.
  mutable std::array<Bitboard, 2> cached_attacked_by_{0, 0};
  mutable std::array<bool, 2> attacked_squares_valid_{false, false};
};

// ---------------------------------------------------------------------------
// Move-formatting helpers (inlined — no separate .cpp symbol required)
// ---------------------------------------------------------------------------

[[nodiscard]] inline std::string move_to_string(const Move &move,
                                                const Board & /*board*/) {
  static constexpr char files[] = "abcdefgh";
  static constexpr char ranks[] = "12345678";
  static constexpr char promo[] = " nbrq"; // index 0 unused; 1-4 = N B R Q

  std::string result;
  result.reserve(5);
  result += files[move.from % 8];
  result += ranks[move.from / 8];
  result += files[move.to % 8];
  result += ranks[move.to / 8];
  if (move.promotion != 0)
    result += promo[move.promotion];
  return result;
}

[[nodiscard]] inline std::string move_debug_string(const Move &move,
                                                   const Board &board) {
  return move_to_string(move, board);
}

[[nodiscard]] inline std::string moves_to_string(const std::vector<Move> &moves,
                                                 const Board &board) {
  std::string result;
  result.reserve(moves.size() * 10);
  result = "Moves [" + std::to_string(moves.size()) + "]:\n";
  for (size_t i = 0; i < moves.size(); ++i)
    result += "  " + std::to_string(i) + ": " +
              move_to_string(moves[i], board) + "\n";
  return result;
}

[[nodiscard]] Move move_from_uci(const std::string &uci);
[[nodiscard]] std::string move_to_uci(const Move &move);

// ---------------------------------------------------------------------------
// Inline method definitions
// ---------------------------------------------------------------------------

inline Board::Board() { load_fen(STARTING_FEN); }

inline void Board::clear() noexcept {
  for (auto &bb : piece_bitboards)
    bb = 0ULL;
  for (auto &bb : color_bitboards)
    bb = 0ULL;
  for (int i = 0; i < 64; ++i) {
    piece_on[i] = EMPTY_SQ;
    color_on[i] = 0;
  }
  side_to_move = true;
  castling_rights = 0;
  en_passant_square = -1;
  halfmove_clock = 0;
  fullmove_number = 1;
  state_history.clear();
  attacked_squares_valid_[0] = false;
  attacked_squares_valid_[1] = false;
}

[[nodiscard]] inline Board Board::from_fen(const std::string &fen) noexcept {
  Board b;
  b.load_fen(fen);
  return b;
}

inline Board Board::copy() const noexcept { return *this; }

inline void Board::make_move(const Move &move) noexcept {
  apply_move(move, nullptr);
}
