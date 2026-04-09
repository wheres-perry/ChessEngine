#include <pybind11/pybind11.h>

#include "eval.hpp"

namespace py = pybind11;

void init_eval_bindings(py::module_ &m) {
  using namespace eval;

  py::class_<EvalConfig>(m, "EvalConfig")
      .def(py::init<>())
      .def_readwrite("use_pst", &EvalConfig::use_pst)
      .def_readwrite("use_pawn_structure", &EvalConfig::use_pawn_structure)
      .def_readwrite("use_mobility", &EvalConfig::use_mobility)
      .def_readwrite("use_king_safety", &EvalConfig::use_king_safety)
      .def_readwrite("game_stage_conscious", &EvalConfig::game_stage_conscious);

  py::class_<Evaluator>(m, "CppEvaluator")
      .def(py::init<const EvalConfig &>(), py::arg("config"))
      .def("go", &Evaluator::go, py::arg("board"));
}
