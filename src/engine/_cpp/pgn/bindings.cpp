#include "pgn.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void init_pgn_bindings(py::module_ &m) {
  py::module_ pgn_m = m.def_submodule("pgn", "Fast C++ PGN parsing module");

  py::class_<pgn::Game>(pgn_m, "Game")
      .def_readonly("headers", &pgn::Game::headers, "Dictionary of PGN headers")
      .def_readonly("moves", &pgn::Game::moves, "List of clean SAN moves")
      .def_readonly("result", &pgn::Game::result,
                    "Game result (e.g., '1-0', '1/2-1/2')")
      .def("__repr__", [](const pgn::Game &g) {
        std::string res = "<Game ";
        auto white_it = g.headers.find("White");
        auto black_it = g.headers.find("Black");

        if (white_it != g.headers.end())
          res += white_it->second;
        else
          res += "?";

        res += " vs ";

        if (black_it != g.headers.end())
          res += black_it->second;
        else
          res += "?";

        res += " (" + std::to_string(g.moves.size()) + " moves)>";
        return res;
      });

  py::class_<pgn::PGNStream>(pgn_m, "PGNStream")
      .def(py::init<std::string>(), py::arg("filepath"),
           "Open a PGN file for streaming")
      .def("__iter__", [](pgn::PGNStream &s) -> pgn::PGNStream & { return s; })
      .def("__next__", [](pgn::PGNStream &s) {
        auto game = s.next();
        if (!game)
          throw py::stop_iteration();
        return *game;
      });
}