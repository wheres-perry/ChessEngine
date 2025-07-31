py::class_<Board>(m, "Board")
    .def(py::init<>())
    .def_static("from_fen", &Board::from_fen, py::arg("fen"))
    .def("make_move", &Board::make_move, py::arg("move"))
    .def("generate_legal_moves", &Board::generate_legal_moves)
    .def("get_castling_rights", &Board::get_castling_rights)  // Add this line
    .def("to_fen", &Board::to_fen)
    .def("pretty", &Board::pretty);