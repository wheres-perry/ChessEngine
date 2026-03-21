#pragma once

#include <vector>
#include "../board/board.hpp"

namespace extractors {

// A. CNN Spatial Extractor
// Returns a flat vector representing [17, 8, 8] float tensor
// Channels 0-11: Pieces (White P-K, Black P-K)
// Channel 12: Side to Move (1 if White, 0 if Black)
// Channels 13-16: Castling rights (WK, WQ, BK, BQ)
std::vector<float> extract_cnn(const Board& board);

// B. NNUE Sparse Categorical Extractor (HalfKP)
// Returns a vector of active feature indices for the Side to Move.
std::vector<int> extract_halfkp(const Board& board);

// D. GNN Graph Extractor
struct GraphData {
    // Flat array of [num_pieces * 3], each chunk is (square, piece_type, color)
    std::vector<int> nodes; 
    // Flat array of [2 * num_edges], each chunk is (src_node_idx, dst_node_idx)
    std::vector<int> edges; 
};
GraphData extract_gnn(const Board& board);

} // namespace extractors