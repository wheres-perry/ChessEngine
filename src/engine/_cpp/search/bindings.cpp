#include <optional>
#include <tuple>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "move_ordering.hpp"
#include "search.hpp"
#include "transposition_table.hpp"
#include "zobrist.hpp"

namespace py = pybind11;
using namespace search;

void init_search_bindings(py::module_ &m) {
  // ── BoundType enum ───────────────────────────────────────────────
  py::enum_<BoundType>(m, "BoundType")
      .value("EXACT", BoundType::EXACT)
      .value("LOWER", BoundType::LOWER)
      .value("UPPER", BoundType::UPPER);

  // ── TTEntry ──────────────────────────────────────────────────────
  py::class_<TTEntry>(m, "TTEntry")
      .def(py::init<>())
      .def_readonly("key", &TTEntry::key)
      .def_readonly("score", &TTEntry::score)
      .def_readonly("best_move", &TTEntry::best_move)
      .def_readonly("depth", &TTEntry::depth)
      .def_readonly("bound", &TTEntry::bound)
      .def_readonly("age", &TTEntry::age)
      .def("__repr__", [](const TTEntry &e) {
        return "<TTEntry key=" + std::to_string(e.key) +
               " depth=" + std::to_string(e.depth) +
               " score=" + std::to_string(e.score) + ">";
      });

  // ── TranspositionTable ───────────────────────────────────────────
  py::class_<TranspositionTable>(m, "TranspositionTable")
      .def(py::init<size_t>(), py::arg("size_mb") = 64)
      .def("resize", &TranspositionTable::resize, py::arg("size_mb"))
      .def("clear", &TranspositionTable::clear)
      .def("increment_age", &TranspositionTable::increment_age)
      .def("probe", &TranspositionTable::probe, py::arg("key"),
           py::return_value_policy::reference_internal)
      .def("try_get_score", &TranspositionTable::try_get_score,
           py::arg("entry"), py::arg("depth"), py::arg("alpha"),
           py::arg("beta"))
      .def(
          "store",
          [](TranspositionTable &self, uint64_t key, int depth, int32_t score,
             std::optional<Move> best_move, BoundType bound) {
            Move mv{};
            if (best_move.has_value())
              mv = best_move.value();
            self.store(key, depth, score, mv, bound);
          },
          py::arg("key"), py::arg("depth"), py::arg("score"),
          py::arg("best_move"), py::arg("bound"))
      .def("size", &TranspositionTable::size)
      .def("capacity", &TranspositionTable::capacity)
      .def("hashfull", &TranspositionTable::hashfull)
      .def("current_age", &TranspositionTable::current_age);

  py::class_<Zobrist>(m, "Zobrist")
      .def(py::init<std::optional<uint64_t>>(), py::arg("seed") = py::none())
      .def("hash_board", &Zobrist::hash_board, "Compute full hash from scratch",
           py::arg("board"))
      .def("make_move_hash", &Zobrist::make_move_hash,
           "Fast incremental hash update", py::arg("board"), py::arg("move"))
      .def("make_null_move_hash", &Zobrist::make_null_move_hash,
           "O(1) incremental hash for null move", py::arg("board"))
      .def("get_current_hash", &Zobrist::get_current_hash, "Get current hash")
      .def("set_current_hash", &Zobrist::set_current_hash, "Set current hash",
           py::arg("hash_val"))
      .def("invalidate_hash", &Zobrist::invalidate_hash, "Invalidate hash");

  // ── SearchConfig ─────────────────────────────────────────────────
  py::class_<SearchConfig>(m, "SearchConfig")
      .def(py::init<>())
      // Move ordering
      .def_readwrite("use_move_ordering", &SearchConfig::use_move_ordering)
      .def_readwrite("use_mvv_lva", &SearchConfig::use_mvv_lva)
      .def_readwrite("use_history_heuristic",
                     &SearchConfig::use_history_heuristic)
      .def_readwrite("use_countermove_heuristic",
                     &SearchConfig::use_countermove_heuristic)
      .def_readwrite("use_see_ordering", &SearchConfig::use_see_ordering)
      .def_readwrite("use_killer_moves", &SearchConfig::use_killer_moves)
      .def_readwrite("use_hash_move_ordering",
                     &SearchConfig::use_hash_move_ordering)
      .def_readwrite("history_max_score", &SearchConfig::history_max_score)
      .def_readwrite("killer_slots_per_ply",
                     &SearchConfig::killer_slots_per_ply)
      .def_readwrite("see_capture_threshold",
                     &SearchConfig::see_capture_threshold)
      // Search algorithms
      .def_readwrite("use_alpha_beta", &SearchConfig::use_alpha_beta)
      .def_readwrite("use_pvs", &SearchConfig::use_pvs)
      .def_readwrite("use_quiescence_search",
                     &SearchConfig::use_quiescence_search)
      .def_readwrite("qs_max_depth", &SearchConfig::qs_max_depth)
      .def_readwrite("use_iid", &SearchConfig::use_iid)
      .def_readwrite("iid_min_depth", &SearchConfig::iid_min_depth)
      .def_readwrite("iid_depth_reduction", &SearchConfig::iid_depth_reduction)
      // Pruning
      .def_readwrite("use_null_move_pruning",
                     &SearchConfig::use_null_move_pruning)
      .def_readwrite("nmp_reduction_r", &SearchConfig::nmp_reduction_r)
      .def_readwrite("nmp_min_depth", &SearchConfig::nmp_min_depth)
      .def_readwrite("use_lmr", &SearchConfig::use_lmr)
      .def_readwrite("lmr_min_depth", &SearchConfig::lmr_min_depth)
      .def_readwrite("lmr_min_move_number", &SearchConfig::lmr_min_move_number)
      .def_readwrite("use_futility_pruning",
                     &SearchConfig::use_futility_pruning)
      .def_readwrite("futility_margin_standard",
                     &SearchConfig::futility_margin_standard)
      .def_readwrite("use_extended_futility_pruning",
                     &SearchConfig::use_extended_futility_pruning)
      .def_readwrite("futility_margin_extended",
                     &SearchConfig::futility_margin_extended)
      .def_readwrite("use_reverse_futility_pruning",
                     &SearchConfig::use_reverse_futility_pruning)
      .def_readwrite("rfp_margin_multiplier",
                     &SearchConfig::rfp_margin_multiplier)
      .def_readwrite("rfp_max_depth", &SearchConfig::rfp_max_depth)
      .def_readwrite("use_delta_pruning", &SearchConfig::use_delta_pruning)
      .def_readwrite("delta_margin", &SearchConfig::delta_margin)
      .def_readwrite("use_see_pruning_in_qs",
                     &SearchConfig::use_see_pruning_in_qs)
      // State & hashing
      .def_readwrite("use_aspiration_windows",
                     &SearchConfig::use_aspiration_windows)
      .def_readwrite("aspiration_window_margin",
                     &SearchConfig::aspiration_window_margin)
      .def_readwrite("use_check_extensions",
                     &SearchConfig::use_check_extensions)
      .def_readwrite("max_check_extensions",
                     &SearchConfig::max_check_extensions)
      .def_readwrite("use_transposition_table",
                     &SearchConfig::use_transposition_table)
      .def_readwrite("tt_size_mb", &SearchConfig::tt_size_mb)
      .def_readwrite("use_tt_aging", &SearchConfig::use_tt_aging)
      // Syzygy
      .def_readwrite("use_syzygy", &SearchConfig::use_syzygy)
      .def_readwrite("use_50_move_rule", &SearchConfig::use_50_move_rule)
      // Lazy SMP
      .def_readwrite("use_lazy_smp", &SearchConfig::use_lazy_smp)
      .def_readwrite("smp_num_threads", &SearchConfig::smp_num_threads)
      // Time
      .def_readwrite("max_time", &SearchConfig::max_time)
      .def_readwrite("has_max_time", &SearchConfig::has_max_time);

  // ── SearchStats (new, with all fields) ───────────────────────────
  py::class_<SearchStats>(m, "SearchStats")
      .def_property_readonly("nodes",
                             [](const SearchStats &s) {
                               return s.nodes.load(std::memory_order_relaxed);
                             })
      .def_readonly("depth", &SearchStats::depth)
      .def_readonly("seldepth", &SearchStats::seldepth)
      .def_property_readonly("tt_hits",
                             [](const SearchStats &s) {
                               return s.tt_hits.load(std::memory_order_relaxed);
                             })
      .def_readonly("hashfull", &SearchStats::hashfull)
      .def_property_readonly("beta_cutoffs",
                             [](const SearchStats &s) {
                               return s.beta_cutoffs.load(
                                   std::memory_order_relaxed);
                             })
      .def_property_readonly("first_move_cuts",
                             [](const SearchStats &s) {
                               return s.first_move_cuts.load(
                                   std::memory_order_relaxed);
                             })
      .def_property_readonly("killer_cuts",
                             [](const SearchStats &s) {
                               return s.killer_cuts.load(
                                   std::memory_order_relaxed);
                             })
      .def_property_readonly("history_cuts",
                             [](const SearchStats &s) {
                               return s.history_cuts.load(
                                   std::memory_order_relaxed);
                             })
      .def_property_readonly("qsearch_nodes",
                             [](const SearchStats &s) {
                               return s.qsearch_nodes.load(
                                   std::memory_order_relaxed);
                             })
      .def_property_readonly("null_move_cuts",
                             [](const SearchStats &s) {
                               return s.null_move_cuts.load(
                                   std::memory_order_relaxed);
                             })
      .def_property_readonly("pvs_researches",
                             [](const SearchStats &s) {
                               return s.pvs_researches.load(
                                   std::memory_order_relaxed);
                             })
      .def_property_readonly("lmr_researches",
                             [](const SearchStats &s) {
                               return s.lmr_researches.load(
                                   std::memory_order_relaxed);
                             })
      .def_property_readonly("qs_see_pruning",
                             [](const SearchStats &s) {
                               return s.qs_see_pruning.load(
                                   std::memory_order_relaxed);
                             })
      .def_property_readonly("qs_delta_pruning",
                             [](const SearchStats &s) {
                               return s.qs_delta_pruning.load(
                                   std::memory_order_relaxed);
                             })
      .def_property_readonly("check_extensions",
                             [](const SearchStats &s) {
                               return s.check_extensions.load(
                                   std::memory_order_relaxed);
                             })
      .def_property_readonly("iid_searches",
                             [](const SearchStats &s) {
                               return s.iid_searches.load(
                                   std::memory_order_relaxed);
                             })
      .def_property_readonly("root_move_changes",
                             [](const SearchStats &s) {
                               return s.root_move_changes.load(
                                   std::memory_order_relaxed);
                             })
      .def_readonly("history_saturation", &SearchStats::history_saturation)
      .def_readonly("score", &SearchStats::score)
      .def_readonly("best_move", &SearchStats::best_move)
      .def("__repr__", [](const SearchStats &s) {
        return "<SearchStats nodes=" +
               std::to_string(s.nodes.load(std::memory_order_relaxed)) +
               " depth=" + std::to_string(s.depth) + ">";
      });

  // ── Search (new full-featured search) ────────────────────────────
  py::class_<Search>(m, "Search")
      .def(py::init([](Board &board, const SearchConfig &config,
                       eval::Evaluator *evaluator,
                       syzygy::SyzygyProber *syzygy) {
             return new Search(board, config, evaluator, syzygy);
           }),
           py::arg("board"), py::arg("config"), py::arg("evaluator"),
           py::arg("syzygy") = nullptr, py::keep_alive<1, 2>(),
           py::keep_alive<1, 4>(), py::keep_alive<1, 5>())
      .def(
          "find_best_move",
          [](Search &self, int depth) {
            py::gil_scoped_release release;
            return self.find_best_move(depth);
          },
          py::arg("depth"))
      .def("get_stats", &Search::get_stats, "Get search statistics",
           py::return_value_policy::reference_internal)
      .def("reset_state", &Search::reset_state,
           "Reset search state (TT, history, killers)",
           py::arg("clear_tt") = true, py::arg("clear_history") = true,
           py::arg("clear_killers") = true);

  // ── MoveSorterConfig ──────────────────────────────────────────────
  using MsCfg = move_ordering::MoveSorterConfig;
  py::class_<MsCfg>(m, "MoveSorterConfig")
      .def(py::init<>())
      .def_readwrite("use_move_ordering", &MsCfg::use_move_ordering)
      .def_readwrite("use_mvv_lva", &MsCfg::use_mvv_lva)
      .def_readwrite("use_history_heuristic", &MsCfg::use_history_heuristic)
      .def_readwrite("use_countermove_heuristic",
                     &MsCfg::use_countermove_heuristic)
      .def_readwrite("use_see_ordering", &MsCfg::use_see_ordering)
      .def_readwrite("use_killer_moves", &MsCfg::use_killer_moves)
      .def_readwrite("use_hash_move_ordering", &MsCfg::use_hash_move_ordering)
      .def_readwrite("history_max_score", &MsCfg::history_max_score)
      .def_readwrite("killer_slots_per_ply", &MsCfg::killer_slots_per_ply)
      .def_readwrite("see_capture_threshold", &MsCfg::see_capture_threshold);

  // ── MoveSorter ────────────────────────────────────────────────────
  using Ms = move_ordering::MoveSorter;
  py::class_<Ms>(m, "MoveSorter")
      .def(py::init<const MsCfg &>(), py::arg("config"))
      .def(
          "sort_moves",
          [](const Ms &self, Board &board, const std::vector<Move> &moves,
             int ply, std::optional<Move> hash_move,
             std::optional<Move> previous_move) {
            Move hm = hash_move.value_or(move_ordering::NO_MOVE);
            Move pm = previous_move.value_or(move_ordering::NO_MOVE);
            return self.sort_moves(board, moves, ply, hm, pm);
          },
          py::arg("board"), py::arg("moves"), py::arg("ply"),
          py::arg("hash_move"), py::arg("previous_move"))
      .def("sort_tactical", &Ms::sort_tactical, py::arg("board"),
           py::arg("moves"))
      .def(
          "see",
          [](const Ms &self, Board &board, const Move &move) {
            return self.see(board, move);
          },
          py::arg("board"), py::arg("move"))
      .def(
          "on_beta_cutoff",
          [](Ms &self, const Move &move, int ply, int depth,
             std::optional<Move> previous_move, bool is_tactical) {
            Move pm = previous_move.value_or(move_ordering::NO_MOVE);
            self.on_beta_cutoff(move, ply, depth, pm, is_tactical);
          },
          py::arg("move"), py::arg("ply"), py::arg("depth"),
          py::arg("previous_move"), py::arg("is_tactical"))
      .def("history_saturation", &Ms::history_saturation)
      .def("reset", &Ms::reset, py::arg("clear_history") = true,
           py::arg("clear_killers") = true)
      .def("get_killers", &Ms::get_killers, py::arg("ply"),
           py::return_value_policy::reference_internal)
      .def("get_history", &Ms::get_history, py::arg("from_sq"),
           py::arg("to_sq"), py::arg("promo"))
      .def("get_history_table",
           [](const Ms &self) {
             // Return as dict of (from, to, promo) -> score from flat arrays
             py::dict result;
             for (const auto &[key, val] : self.get_history_entries()) {
               auto py_key = py::make_tuple(std::get<0>(key), std::get<1>(key),
                                            std::get<2>(key));
               result[py_key] = val;
             }
             return result;
           })
      .def("get_killer_moves_dict", [](const Ms &self) {
        // Return as dict of ply -> list[Move]
        py::dict result;
        for (int ply = 0; ply < move_ordering::MAX_PLY; ++ply) {
          const auto &killers = self.get_killers(ply);
          if (!killers.empty()) {
            result[py::int_(ply)] = killers;
          }
        }
        return result;
      });
}
