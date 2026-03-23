# ChessEngine

A modular, high-performance chess engine written in Python with a heavily optimized C++ core. 

The central idea of this project is to provide a highly configurable chess engine that supports modular optimizations. This makes it especially useful for educational purposes, algorithmic comparisons, and repeatable benchmarking between different search and evaluation heuristics (e.g., comparing the performance gain of Null Move Pruning vs. pure Alpha-Beta Minimax).

## Architecture

- **C++ Core**: Move generation, board state management, and Zobrist hashing are implemented in C++ and exposed to Python via Pybind11 for maximum performance.
- **Search**: A flexible Negamax searcher supporting Alpha-Beta pruning, Principal Variation Search (PVS), Quiescence Search, Null Move Pruning, Late Move Reductions (LMR), and more.
- **Evaluation**: A composable evaluation pipeline including Material, Piece-Square Tables (PST), Pawn Structure, Mobility, and King Safety.
- **UCI Protocol**: Full support for the Universal Chess Interface (UCI) protocol, allowing it to be used with standard chess GUIs.

## Requirements

The project uses Docker to provide a consistent development and execution environment.

- Docker
- `uv` (for local dependency management, optional)

## Building the Engine

Build the development Docker image:

```bash
docker build --target development -t chess_engine_dev .
```

## Running the Engine

The engine communicates using the standard UCI protocol. You can run the engine directly inside the Docker container.

### Command Line (UCI)

You can pipe UCI commands directly to the engine. For example, using PowerShell:

```powershell
@"
uci
setoption name use_null_move_pruning value false
setoption name qs_max_depth value 4
setoption name use_mobility value true
setoption name Hash value 32
isready
position startpos moves e2e4 e7e5
go depth 4
quit
"@ | docker run --rm -i chess_engine_dev python -m engine
```

Or run it interactively:

```bash
docker run --rm -i chess_engine_dev python -m engine
```
*(Then type UCI commands like `uci`, `isready`, `position startpos`, `go depth 4`)*

### Connecting to a GUI

You can connect this engine to chess GUIs like Arena, Cute Chess, or Lucas Chess.
Simply point the GUI to a batch script or shell script that executes:
```bash
docker run -i chess_engine_dev python -m engine
```
The GUI will automatically parse the `uci` output to dynamically generate a settings menu with all available configuration flags.

## Configuration Flags

The engine exposes all internal `SearchConfig` and `EvaluationConfig` parameters dynamically via UCI `setoption`. You can toggle features or tune integer parameters on the fly. 

Some key toggles include:
- `use_alpha_beta`: Enable/disable Alpha-Beta pruning.
- `use_pvs`: Enable/disable Principal Variation Search.
- `use_null_move_pruning`: Enable/disable Null Move Pruning.
- `use_quiescence_search`: Enable/disable Quiescence Search.
- `use_transposition_table`: Enable/disable Zobrist Transposition Tables.
- `use_pst`, `use_pawn_structure`, `use_mobility`, `use_king_safety`: Toggle specific evaluation heuristics.
- `Hash`: Set the Transposition Table size in Megabytes.

## Development and Testing

The project uses `nox` for automation within the Docker container.

Run code linters and formatters:
```bash
docker run --rm chess_engine_dev nox -s lint
```

Run type checking:
```bash
docker run --rm chess_engine_dev nox -s types
```

Run the fast test suite:
```bash
docker run --rm chess_engine_dev nox -s tests_fast
```

Run the full test suite (including slow benchmarks/parity tests):
```bash
docker run --rm chess_engine_dev nox -s tests_full
```