#include <optional>
#include <string>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "syzygy.hpp"

namespace py = pybind11;

void init_syzygy_bindings(py::module_ &m) {
  using namespace syzygy;

  py::class_<SyzygyProber>(m, "CppSyzygyProber")
      .def(py::init([](std::string path, bool use_50_move_rule,
                       py::function wdl_func, py::function dtz_func) {
             // Wrap Python callables into std::function, acquiring the
             // GIL each time we call back into Python.
             auto wdl = [wdl_func = std::move(wdl_func)](
                            const std::string &fen,
                            bool use_50mr) -> std::optional<int> {
               py::gil_scoped_acquire acquire;
               py::object result = wdl_func(fen, use_50mr);
               if (result.is_none()) {
                 return std::nullopt;
               }
               return result.cast<int>();
             };
             auto dtz = [dtz_func = std::move(dtz_func)](
                            const std::string &fen,
                            bool use_50mr) -> std::optional<int> {
               py::gil_scoped_acquire acquire;
               py::object result = dtz_func(fen, use_50mr);
               if (result.is_none()) {
                 return std::nullopt;
               }
               return result.cast<int>();
             };
             return new SyzygyProber(std::move(path), use_50_move_rule,
                                     std::move(wdl), std::move(dtz));
           }),
           py::arg("path"), py::arg("use_50_move_rule"), py::arg("wdl_func"),
           py::arg("dtz_func"))
      .def_static("piece_count", &SyzygyProber::piece_count, py::arg("board"))
      .def("probe_wdl", &SyzygyProber::probe_wdl, py::arg("board"))
      .def("probe_dtz", &SyzygyProber::probe_dtz, py::arg("board"));
}
