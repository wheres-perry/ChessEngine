#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "evaluators.hpp"

namespace py = pybind11;
using namespace evaluators;

namespace {

// Trampoline so Python subclasses (e.g. MockEvaluator) can override go().
class PyEvaluator : public IEvaluator {
public:
  using IEvaluator::IEvaluator;
  [[nodiscard]] double go(const Board &board) const override {
    PYBIND11_OVERRIDE_PURE(double, IEvaluator, go, board);
  }
};

} // namespace

void init_evaluator_bindings(py::module_ &m) {
  py::module_ ev_m = m.def_submodule("evaluators", "Position evaluators");

  // Base evaluator interface with Python trampoline.
  py::class_<IEvaluator, PyEvaluator, std::shared_ptr<IEvaluator>>(ev_m,
                                                                   "IEvaluator")
      .def(py::init<>())
      .def("go", &IEvaluator::go, py::arg("board"));

  py::class_<MaterialComponent, IEvaluator, std::shared_ptr<MaterialComponent>>(
      ev_m, "MaterialComponent")
      .def(py::init<>())
      .def("score", &MaterialComponent::score, py::arg("board"),
           py::arg("phase"))
      .def("go", &MaterialComponent::go, py::arg("board"));

  py::class_<PSTComponent, IEvaluator, std::shared_ptr<PSTComponent>>(
      ev_m, "PSTComponent")
      .def(py::init<bool>(), py::arg("gsc") = false)
      .def("score", &PSTComponent::score, py::arg("board"), py::arg("phase"))
      .def("go", &PSTComponent::go, py::arg("board"));

  py::class_<PawnStructureComponent, IEvaluator,
             std::shared_ptr<PawnStructureComponent>>(ev_m,
                                                      "PawnStructureComponent")
      .def(py::init<bool>(), py::arg("gsc") = false)
      .def("score", &PawnStructureComponent::score, py::arg("board"),
           py::arg("phase"))
      .def("go", &PawnStructureComponent::go, py::arg("board"));

  py::class_<MobilityComponent, IEvaluator, std::shared_ptr<MobilityComponent>>(
      ev_m, "MobilityComponent")
      .def(py::init<bool>(), py::arg("gsc") = false)
      .def("score", &MobilityComponent::score, py::arg("board"),
           py::arg("phase"))
      .def("go", &MobilityComponent::go, py::arg("board"));

  py::class_<KingSafetyComponent, IEvaluator,
             std::shared_ptr<KingSafetyComponent>>(ev_m, "KingSafetyComponent")
      .def(py::init<bool>(), py::arg("gsc") = false)
      .def("score", &KingSafetyComponent::score, py::arg("board"),
           py::arg("phase"))
      .def("go", &KingSafetyComponent::go, py::arg("board"));

  py::class_<CompositeEvaluator, IEvaluator,
             std::shared_ptr<CompositeEvaluator>>(ev_m, "CompositeEvaluator")
      .def(py::init<>())
      .def("add_component", &CompositeEvaluator::add_component,
           py::arg("component"))
      .def("go", &CompositeEvaluator::go, py::arg("board"))
      .def("components", &CompositeEvaluator::components,
           py::return_value_policy::reference_internal);

  ev_m.def("compute_game_phase", &compute_game_phase, py::arg("board"));
}
