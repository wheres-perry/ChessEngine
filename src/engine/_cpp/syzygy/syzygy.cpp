#include "syzygy.hpp"
#include <utility>

namespace syzygy {

SyzygyProber::SyzygyProber(std::string path, bool use_50_move_rule,
                           ProbeFunc wdl_func, ProbeFunc dtz_func) noexcept
    : path_(std::move(path)), use_50_move_rule_(use_50_move_rule),
      wdl_func_(std::move(wdl_func)), dtz_func_(std::move(dtz_func)) {}

int SyzygyProber::piece_count(const Board &board) noexcept {
  return popcount64(board.get_all_pieces_bb());
}

std::optional<int> SyzygyProber::probe_wdl(const Board &board) const {
  if (piece_count(board) > MAX_PIECES) {
    return std::nullopt;
  }
  std::string fen = board.to_fen();
  return wdl_func_(fen, use_50_move_rule_);
}

std::optional<int> SyzygyProber::probe_dtz(const Board &board) const {
  if (piece_count(board) > MAX_PIECES) {
    return std::nullopt;
  }
  std::string fen = board.to_fen();
  return dtz_func_(fen, use_50_move_rule_);
}

} // namespace syzygy
