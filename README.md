

This is a modular chess engine written primarily in Python. As performance is a huge concern in chess engines, the internal chess core for generating legal moves, managing board states, and hashing is written in C++, bound to the engine with Pybind11.

The central idea of this project is to provide a chess engine that can introduce modular optimizations and configurations. This makes it especially good at comparisons and having a repeatable and comparable benchmark between different search and evaluation configurations. For example, we can see the performance gain of using Null window pruning vs simple minimax.
