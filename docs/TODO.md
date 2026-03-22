# ChessEngine Development Roadmap

## Core Testing & Stability
- [ ] Setup Elo tests with configs and games (simulated and real)
- [ ] Set up Mate in N tests (M1 through M12+)
- [ ] Set up draw tests (stalemate, threefold repetition, fifty-move rule, insufficient material)
- [ ] Set up "Best Move" tactical tests:
    - [ ] En passant is the only/best move
    - [ ] Castling is the only/best move
    - [ ] Promotion (specifically under-promotion) is the only/best move
- [x] Fix broken benchmark in nox: `pytest --benchmark-only --benchmark-json=output.json --benchmark-autosave tests/benchmarks`
- [x] Fix ASAN/UBSAN sanitizer session in nox (`exit code -6` in `tests_all`)

## Engine & Search Improvements
- [ ] Implement and evaluate different evaluators:
    - [ ] Simple Hand-Coded Evaluation (current)
    - [ ] PeSTO-style tapered evaluation
    - [ ] NNUE (Efficiently Updatable Neural Network) integration
- [ ] Static Exchange Evaluation (SEE) for better move ordering and pruning
- [ ] Syzygy Tablebase Integration in the search loop
- [ ] Refine Null Move Pruning (NMP) with adaptive reduction
- [ ] Implement Aspiration Windows and Principal Variation Search (PVS) optimizations
- [ ] Internal Iterative Deepening (IID) for nodes without hash moves

## Configuration & Tooling
- [x] Dependency Resolution with z3 solver for config validation
- [ ] Expand UCI options support (e.g., MultiPV, Hash size, Threads)
- [ ] Automated Elo regression testing in CI

## Proposed Future Ideas
- [ ] **Multi-threading (Lazy SMP)**: Parallelize search across multiple CPU cores.
- [ ] **Search Visualization Tool**: A CLI or web-based tool to visualize the search tree and pruning decisions. (Planned for later)
- [ ] **Game Database Analysis**: Script to analyze PGN databases to identify engine weaknesses.
