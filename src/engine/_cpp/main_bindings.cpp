#include <pybind11/pybind11.h>

namespace py = pybind11;

// Forward declarations for each component's bindings
void init_board_bindings(py::module_ &m);
void init_movegen_bindings(py::module_ &m);
void init_search_bindings(py::module_ &m);
void init_extractors_bindings(py::module_ &m);
void init_pgn_bindings(py::module_ &m);
void init_eval_bindings(py::module_ &m);
void init_syzygy_bindings(py::module_ &m);

PYBIND11_MODULE(chess_engine_core, m) {
  m.doc() = "High-performance C++ chess engine core components";

  // Initialize each component's bindings
  init_board_bindings(m);
  init_movegen_bindings(m);
  init_search_bindings(m);
  init_extractors_bindings(m);
  init_pgn_bindings(m);
  init_eval_bindings(m);
  init_syzygy_bindings(m);
}
