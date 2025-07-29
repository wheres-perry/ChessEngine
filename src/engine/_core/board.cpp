#include "board.h"
#include <sstream>
#include <vector>
#include <stdexcept>

// Constructor is guaranteed not to throw.
Board::Board() noexcept
{
    clear();
}

// clear() is guaranteed not to throw.
void Board::clear() noexcept
{
    for (auto &bb : piece_bitboards)
        bb = 0ULL;
    for (auto &bb : color_bitboards)
        bb = 0ULL;
    side_to_move = true; // WHITE
    castling_rights = 0;
    en_passant_square = -1;
    halfmove_clock = 0;
    fullmove_number = 1;
}

// load_fen can throw exceptions, so it is NOT marked noexcept.
void Board::load_fen(const std::string &fen)
{
    clear();
    std::stringstream ss(fen);
    std::string piece_placement, active_color, castling, en_passant, halfmove, fullmove;
    ss >> piece_placement >> active_color >> castling >> en_passant >> halfmove >> fullmove;

    int8_t rank = 7, file = 0;
    for (const char c : piece_placement)
    {
        if (c == '/')
        {
            rank--;
            file = 0;
        }
        else if (isdigit(c))
        {
            file += c - '0';
        }
        else
        {
            uint8_t square = rank * 8 + file;
            Color color = isupper(c) ? Color::WHITE : Color::BLACK;
            PieceType pt;
            switch (tolower(c))
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
                throw std::runtime_error("Invalid piece in FEN string");
            }
            piece_bitboards[static_cast<uint8_t>(pt)] |= (1ULL << square);
            color_bitboards[static_cast<uint8_t>(color)] |= (1ULL << square);
            file++;
        }
    }

    side_to_move = (active_color == "w");

    castling_rights = 0; // Reset before setting
    for (const char c : castling)
    {
        if (c == 'K')
            castling_rights |= 1;
        else if (c == 'Q')
            castling_rights |= 2;
        else if (c == 'k')
            castling_rights |= 4;
        else if (c == 'q')
            castling_rights |= 8;
    }

    if (en_passant != "-")
    {
        int8_t ep_file = en_passant[0] - 'a';
        int8_t ep_rank = en_passant[1] - '1';
        en_passant_square = ep_rank * 8 + ep_file;
    }
    else
    {
        en_passant_square = -1;
    }

    try
    {
        halfmove_clock = std::stoi(halfmove);
        fullmove_number = std::stoi(fullmove);
    }
    catch (const std::exception &e)
    {
        throw std::runtime_error("Invalid clock value in FEN string");
    }
}

Board Board::from_fen(const std::string &fen)
{
    Board board;
    board.load_fen(fen);
    return board;
}

// --- Placeholder implementations ---

void Board::make_move(const Move &move) noexcept
{
    // Placeholder
    side_to_move = !side_to_move;
}

std::vector<Move> Board::generate_legal_moves() const noexcept
{
    // Placeholder
    return {};
}

std::vector<float> Board::to_half_kp_features() const
{
    // Placeholder
    return {};
}

std::string Board::to_fen() const
{
    // Placeholder
    return "";
}

std::string Board::pretty() const
{
    // Placeholder
    return "";
}