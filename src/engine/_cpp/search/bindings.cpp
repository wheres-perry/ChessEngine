#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "search.hpp"
#include "zobrist.hpp"

namespace py = pybind11;
using namespace search;

void init_search_bindings(py::module_ &m) {
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

  py::class_<SearchStats>(m, "SearchStats")
      .def_readonly("nodes", &SearchStats::nodes)
      .def_readonly("depth", &SearchStats::depth)
      .def_readonly("seldepth", &SearchStats::seldepth)
      .def_readonly("tt_hits", &SearchStats::tt_hits)
      .def_readonly("hashfull", &SearchStats::hashfull)
      .def_readonly("beta_cutoffs", &SearchStats::beta_cutoffs)
      .def_readonly("first_move_cuts", &SearchStats::first_move_cuts)
      .def_readonly("killer_cuts", &SearchStats::killer_cuts)
      .def_readonly("history_cuts", &SearchStats::history_cuts)
      .def_readonly("qsearch_nodes", &SearchStats::qsearch_nodes)
      .def_readonly("null_move_cuts", &SearchStats::null_move_cuts)
      .def_readonly("pvs_researches", &SearchStats::pvs_researches)
      .def_readonly("root_move_changes", &SearchStats::root_move_changes)
      .def_readonly("score", &SearchStats::score)
      .def_readonly("best_move", &SearchStats::best_move)
      .def("__repr__", [](const SearchStats &s) {
        return "<SearchStats nodes=" + std::to_string(s.nodes) +
               " depth=" + std::to_string(s.depth) + ">";
      });

  py::class_<Search>(m, "Search")
      .def(py::init<Board &>(), py::keep_alive<1, 2>())
      .def("search", &Search::search, "Run a fixed depth search",
           py::arg("depth"))
      .def("get_stats", &Search::get_stats,
           "Get statistics from the last search")
      .def("reset_stats", &Search::reset_stats, "Reset search statistics");
}
