Massive project overhaul and clean up.

This is a major commit antithetical to git commit tracking. However, this is essentially a new project at this point. Interoperatability with python-chess has been completely removed, only using it to test the custom c++ core. A big thanks to python-chess and it's contributors. Any hotswapping of python vs c++ components has been removed. There is no point in comparing C++ to Python, unnecesary complexity.

Major CI/CD changes have been made as well, rehauling how commits and features will be added, and how tests/linters/sanitizers/etc will be ran.

Config assumes Minimax and IDDFS, there is essentially no reason to not have those in any chess engine (ignoring MCTS, currently out of scope for this engine). Assuming both are active greatly decreases the config resolution complexity. Also Transposition tables and zobrist hashing were combined to be the same flag, as there is no reason to separate them.

Adds drawio diagram to detail the config dependencies

Move to using sci kit build as it is more modern and fits well with pybind, also avoids setup.py shim completely.

Cleans up tons of areas I blindy extended, narrows the scope of the project to be more manageable for development.
