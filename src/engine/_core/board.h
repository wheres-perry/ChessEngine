#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <array>

// A 64-bit unsigned integer for bitboards.
using Bitboard = uint64_t;

// Use constexpr for compile-time constants.
constexpr uint8_t NUM_SQUARES = 64;
constexpr uint8_t NUM_PIECE_TYPES = 6;
constexpr uint8_t NUM_COLORS = 2;

// Use enum class for type safety and scoping.
enum class PieceType : uint8_t
{
    PAWN,
    KNIGHT,
    BISHOP,
    ROOK,
    QUEEN,
    KING
};
enum class Color : uint8_t
{
    WHITE,
    BLACK
};

// Represents a single move using minimal types.
struct Move
{
    uint8_t from;      // Square index 0-63
    uint8_t to;        // Square index 0-63
    uint8_t promotion; // PieceType for promotion, or 0 for none.
};

class Board
{
public:
    // Constructors and Factories
    Board() noexcept;
    // from_fen can throw exceptions during parsing, so no noexcept.
    static Board from_fen(const std::string &fen);

    // Core functions for search
    void make_move(const Move &move) noexcept;
    std::vector<Move> generate_legal_moves() const noexcept;

    // Conversion and utility functions
    std::string to_fen() const; // String/vector ops can allocate, may throw std::bad_alloc
    std::string pretty() const;
    std::vector<float> to_half_kp_features() const;

    // `constexpr` allows compile-time evaluation.
    // `[[nodiscard]]` warns if the return value is unused.

    [[nodiscard]] constexpr Bitboard get_piece_bb(PieceType pt, Color c) const noexcept
    {
        return piece_bitboards[static_cast<uint8_t>(pt)] & color_bitboards[static_cast<uint8_t>(c)];
    }

    [[nodiscard]] constexpr Bitboard get_color_bb(Color c) const noexcept
    {
        return color_bitboards[static_cast<uint8_t>(c)];
    }

    [[nodiscard]] constexpr Bitboard get_all_pieces_bb() const noexcept
    {
        return color_bitboards[0] | color_bitboards[1];
    }

    [[nodiscard]] constexpr bool get_side_to_move() const noexcept
    {
        return side_to_move;
    }

    [[nodiscard]] constexpr uint8_t get_castling_rights() const noexcept
    {
        return castling_rights;
    }

    [[nodiscard]] constexpr int8_t get_en_passant_square() const noexcept
    {
        return en_passant_square;
    }

private:
    void load_fen(const std::string &fen);
    void clear() noexcept;

    std::array<Bitboard, NUM_PIECE_TYPES> piece_bitboards;
    std::array<Bitboard, NUM_COLORS> color_bitboards;

    bool side_to_move;
    uint8_t castling_rights;
    int8_t en_passant_square;
    uint8_t halfmove_clock;
    uint16_t fullmove_number;
};