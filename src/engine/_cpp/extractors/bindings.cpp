#include "extractors.hpp"
#include <algorithm>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void init_extractors_bindings(py::module_ &m) {
  py::module_ ext_m =
      m.def_submodule("extractors", "Machine learning feature extractors");

  ext_m.def(
      "extract_cnn",
      [](const Board &board) -> py::array_t<float> {
        std::vector<float> tensor = extractors::extract_cnn(board);
        py::array_t<float> result({17, 8, 8});
        auto buf = result.request();
        float *ptr = static_cast<float *>(buf.ptr);
        std::copy(tensor.begin(), tensor.end(), ptr);
        return result;
      },
      py::arg("board"), "Extract a [17, 8, 8] spatial CNN representation");

  ext_m.def(
      "extract_halfkp",
      [](const Board &board) -> py::array_t<int> {
        std::vector<int> indices = extractors::extract_halfkp(board);
        return py::array_t<int>(indices.size(), indices.data());
      },
      py::arg("board"), "Extract active HalfKP indices for sparse NNUE models");

  ext_m.def(
      "extract_gnn",
      [](const Board &board) -> py::dict {
        extractors::GraphData data = extractors::extract_gnn(board);

        int num_nodes = data.nodes.size() / 3;
        py::array_t<int> py_nodes({num_nodes, 3});
        auto buf_nodes = py_nodes.request();
        std::copy(data.nodes.begin(), data.nodes.end(),
                  static_cast<int *>(buf_nodes.ptr));

        int num_edges = data.edges.size() / 2;
        py::array_t<int> py_edges({2, num_edges});
        auto buf_edges = py_edges.request();

        // PyTorch Geometric edge_index is shape [2, num_edges].
        // PyBind11 arrays are C-contiguous (row-major) by default.
        int *ptr_edges = static_cast<int *>(buf_edges.ptr);
        for (int i = 0; i < num_edges; ++i) {
          ptr_edges[i] = data.edges[i * 2];                 // row 0: src
          ptr_edges[num_edges + i] = data.edges[i * 2 + 1]; // row 1: dst
        }

        py::dict result;
        result["nodes"] = py_nodes;
        result["edge_index"] = py_edges;
        return result;
      },
      py::arg("board"),
      "Extract a PyTorch Geometric compatible Graph representation");
}
