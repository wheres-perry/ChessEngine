#include "extractors.hpp"
#include "../move_generation/move_generation.hpp"

namespace extractors {

std::vector<float> extract_cnn(const Board& board) {
    // 17 channels, 8x8 squares.
    std::vector<float> tensor(17 * 64, 0.0f);
    
    // Channels 0-11: Pieces
    for (int sq = 0; sq < 64; ++sq) {
        if (auto p = board.piece_at(sq)) {
            int c = static_cast<int>(p->color);
            int pt = static_cast<int>(p->type);
            int channel = c * 6 + pt;
            tensor[channel * 64 + sq] = 1.0f;
        }
    }
    
    // Channel 12: Side to move
    float stm_val = board.get_side_to_move() ? 1.0f : 0.0f;
    for (int sq = 0; sq < 64; ++sq) {
        tensor[12 * 64 + sq] = stm_val;
    }
    
    // Channels 13-16: Castling rights
    float wk = board.has_kingside_castling_rights(Color::WHITE) ? 1.0f : 0.0f;
    float wq = board.has_queenside_castling_rights(Color::WHITE) ? 1.0f : 0.0f;
    float bk = board.has_kingside_castling_rights(Color::BLACK) ? 1.0f : 0.0f;
    float bq = board.has_queenside_castling_rights(Color::BLACK) ? 1.0f : 0.0f;
    
    for (int sq = 0; sq < 64; ++sq) {
        tensor[13 * 64 + sq] = wk;
        tensor[14 * 64 + sq] = wq;
        tensor[15 * 64 + sq] = bk;
        tensor[16 * 64 + sq] = bq;
    }
    
    return tensor;
}

std::vector<int> extract_halfkp(const Board& board) {
    std::vector<int> active_indices;
    active_indices.reserve(32);
    
    Color stm = board.side_to_move_color();
    auto king_sq_opt = board.king(stm);
    if (!king_sq_opt) return active_indices; // Safe fallback
    uint8_t k_sq = *king_sq_opt;
    
    Bitboard all_pieces = board.get_all_pieces_bb();
    while (all_pieces) {
        uint8_t sq = pop_lsb(all_pieces);
        if (auto p = board.piece_at(sq)) {
            if (p->type == PieceType::KING) continue; // HalfKP excludes kings from piece features
            
            // Piece mapping: 0-4 for STM pieces, 5-9 for enemy pieces
            int is_enemy = (p->color == stm) ? 0 : 5;
            int pt_idx = static_cast<int>(p->type); // PAWN=0, KNIGHT=1, BISHOP=2, ROOK=3, QUEEN=4
            int piece_idx = pt_idx + is_enemy;
            
            int feature_idx = k_sq * (10 * 64) + piece_idx * 64 + sq;
            active_indices.push_back(feature_idx);
        }
    }
    
    return active_indices;
}

GraphData extract_gnn(const Board& board) {
    GraphData data;
    data.nodes.reserve(32 * 3);
    data.edges.reserve(32 * 8 * 2); // Heuristic allocation
    
    int sq_to_node[64];
    for (int i = 0; i < 64; ++i) sq_to_node[i] = -1;
    
    int node_idx = 0;
    Bitboard all_pieces = board.get_all_pieces_bb();
    Bitboard temp = all_pieces;
    
    // 1. Create nodes (Only squares with pieces)
    while (temp) {
        uint8_t sq = pop_lsb(temp);
        if (auto p = board.piece_at(sq)) {
            data.nodes.push_back(sq);
            data.nodes.push_back(static_cast<int>(p->type));
            data.nodes.push_back(static_cast<int>(p->color));
            sq_to_node[sq] = node_idx++;
        }
    }
    
    // 2. Create edges (directed attacks between pieces)
    temp = all_pieces;
    while (temp) {
        uint8_t sq = pop_lsb(temp);
        auto p = board.piece_at(sq);
        if (!p) continue;
        
        Bitboard att = 0;
        switch (p->type) {
            case PieceType::PAWN:
                att = PAWN_ATTACKS[static_cast<int>(p->color)][sq];
                // Note: Pawns attack diagonally. Straight pushes are not "edges" in standard GNN.
                break;
            case PieceType::KNIGHT:
                att = KNIGHT_ATTACKS[sq];
                break;
            case PieceType::BISHOP:
                att = get_ray_attacks(sq, BISHOP_DIRECTIONS, 4, all_pieces);
                break;
            case PieceType::ROOK:
                att = get_ray_attacks(sq, ROOK_DIRECTIONS, 4, all_pieces);
                break;
            case PieceType::QUEEN:
                att = get_ray_attacks(sq, QUEEN_DIRECTIONS, 8, all_pieces);
                break;
            case PieceType::KING:
                att = KING_ATTACKS[sq];
                break;
        }
        
        att &= all_pieces; // Edges only point to existing piece nodes
        while (att) {
            uint8_t tgt_sq = pop_lsb(att);
            int src_idx = sq_to_node[sq];
            int dst_idx = sq_to_node[tgt_sq];
            if (src_idx != -1 && dst_idx != -1) {
                data.edges.push_back(src_idx);
                data.edges.push_back(dst_idx);
            }
        }
    }
    
    return data;
}

} // namespace extractors