#pragma once

#include "../board/board.hpp"
#include <cstdint>
#include <functional>
#include <optional>
#include <string>

namespace syzygy {

static constexpr int MAX_PIECES = 5;

class SyzygyProber {
public:
  /// Callback type: takes a FEN string and use_50_move_rule flag,
  /// returns an optional int result from the Python tablebase.
  using ProbeFunc =
      std::function<std::optional<int>(const std::string &, bool)>;

  SyzygyProber(std::string path, bool use_50_move_rule, ProbeFunc wdl_func,
               ProbeFunc dtz_func) noexcept;

  /// Count total pieces on the board using hardware popcount.
  [[nodiscard]] static int piece_count(const Board &board) noexcept;

  /// Probe WDL table.  Returns 2=win, 1=cursed win, 0=draw,
  /// -1=blessed loss, -2=loss, or nullopt if not in tablebase.
  [[nodiscard]] std::optional<int> probe_wdl(const Board &board) const;

  /// Probe DTZ table.  Returns distance-to-zeroing (positive = win,
  /// negative = loss, zero = draw), or nullopt if not in tablebase.
  [[nodiscard]] std::optional<int> probe_dtz(const Board &board) const;

private:
  std::string path_;
  bool use_50_move_rule_;
  ProbeFunc wdl_func_;
  ProbeFunc dtz_func_;
};

} // namespace syzygy
