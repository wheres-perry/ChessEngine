# Migration Plan: python-chess → Custom C++ Chess Engine

## ✅ Completed

1. **C++ Core Implementation**
   - ✅ Added `Piece` struct with `piece_type`, `color`, `symbol()`
   - ✅ Added `StateInfo` struct for move undo stack
   - ✅ Implemented `push()` and `pop()` methods with history stack
   - ✅ Added move predicates: `is_capture()`, `is_castling()`, `is_en_passant()`, `is_check()`
   - ✅ Added board queries: `piece_at()`, `pieces()`, `king()`
   - ✅ Added square utilities: `square_file()`, `square_rank()`
   - ✅ Added constants: `SQUARES`, `PIECE_TYPES_ARRAY`, `BB_A1`, `BB_H1`, etc.
   - ✅ Added UCI/SAN parsing: `move_from_uci()`, `move_to_uci()`, `push_san()`
   - ✅ Added properties: `turn`, `ep_square`, `castling_rights`, `legal_moves`
   - ✅ Added `set_fen()` and `fen` property alias

2. **PyBind Bindings**
   - ✅ Bound all new C++ classes and methods
   - ✅ Added Python-friendly Move class with `from_uci()`, `uci()`, `__hash__`, `__eq__`
   - ✅ Added Piece class with proper Python integration
   - ✅ Exported all constants and helper functions

3. **Compilation**
   - ✅ Fixed compilation errors
   - ✅ Successfully builds and installs

## 🔄 Next Steps

### Phase 1: Direct Import Migration (COMPLETED)

- [x] All `import chess` statements in `src/` now use `from engine._core import chess_engine_core as chess`.
- [x] Only PGN utilities (`src/io_utils/load_games.py`) import python-chess (`chess.pgn`) for parsing, then convert to the C++ board via FEN.
- [x] Test suite updated to instantiate the C++ boards directly.

### Phase 2: Python-Chess Ground Truth Tests (COMPLETED/ONGOING)

- [x] Added `tests/core/test_python_chess_parity.py` which cross-checks move generation and move application against python-chess.
- [ ] Expand parity coverage to additional scenarios (captures, promotions, castling edge cases).

### Phase 3: PGN Handling (IN PROGRESS)

- ✅ Current approach: parse PGNs with python-chess, convert to C++ boards.
- [ ] Evaluate moving PGN conversion into a dedicated adapter module with better logging and error handling.

### Phase 4: Testing & Validation (ONGOING)

- [ ] Run `pytest tests/` after every major change.
- [ ] Add nox session to mirror CI workflow.
- [ ] Benchmark C++ board vs python-chess for representative workloads.

### Phase 5: Documentation & Cleanup (ONGOING)

- [x] Documented migration status in this file.
- [ ] Update README to describe accelerated board usage and PGN strategy.
- [ ] Note python-chess is now an optional dependency used only for PGN import and parity tests.

## 🐛 Known Issues

1. **PGN Sampling** – currently logs warnings but does not raise custom exceptions. Introduce a custom `PgnLoadError`.
2. **HalfKP Representation** – module still unimplemented; ensure future work uses new board API.

## 📊 Progress Tracking

- [x] C++ Core Implementation
- [x] PyBind Bindings
- [x] Compilation
- [x] Direct Import Migration
- [ ] Expanded Parity Tests
- [ ] PGN Adapter Enhancements
- [ ] Full Testing & Benchmarking
- [ ] README/Docs Update

## 🎯 Immediate Next Actions

1. Increase parity coverage by comparing en-passant/castling cases against python-chess.
2. Enhance PGN loader logging with custom exceptions.
3. Run the full pytest suite (via nox) once the above changes are in place.

