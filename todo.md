# Chess Engine @config.py Implementation Roadmap

## Overview
This document tracks the implementation of all 49 features referenced in the `@config.py` configuration system. Currently **16 features are implemented** (~33%) and **33 features are missing** (~67%).

## Implementation Status Summary

### Search Features (Tree 1 - Move Exploration)
**✅ IMPLEMENTED (10/38):**
- `use_minimax` - Basic minimax search
- `use_alpha_beta` - Alpha-beta pruning
- `use_iddfs` - Iterative deepening
- `use_move_ordering` - Move ordering system
- `use_transposition_table` - TT implementation
- `use_zobrist` - Zobrist hashing
- `use_tt_aging` - TT aging strategy
- `use_pvs` - Principal variation search
- `use_lmr` - Late move reductions
- Move ordering heuristics (6/6): `use_killer_moves`, `use_history_heuristic`, `use_countermove_heuristic`, `use_hash_move_ordering`, `use_mvv_lva`, `use_see_ordering`

**❌ MISSING (28/38):**
- `use_check_extensions` - Check extensions
- `use_quiescence_search` - Quiescence search
- `use_futility_pruning` - Futility pruning
- `use_extended_futility_pruning` - Extended futility pruning
- `use_reverse_futility_pruning` - Reverse futility pruning
- `use_null_move_pruning` - Null move pruning
- `use_aspiration_windows` - Aspiration windows
- `use_delta_pruning` - Delta pruning (requires quiescence)
- `use_see_pruning_in_qs` - SEE pruning in quiescence
- `use_recapture_extensions` - Recapture extensions
- `use_singular_extensions` - Singular extensions
- `use_iid` - Internal iterative deepening
- `use_probcut` - ProbCut pruning
- `use_multicut_pruning` - Multi-cut pruning
- `use_razoring` - Razoring
- Parallel search (5 variants): `use_parallel_search`, `use_naive_parallel`, `use_lazy_smp`, `use_ybwc`, `use_dts`
- `use_mtdf` - MTD(f) search
- `use_opening_book` - Opening book system
- `use_endgame_tablebases` - Endgame tablebases

### Evaluation Features (Tree 2 - State Evaluation)
**✅ IMPLEMENTED (6/11):**
- `use_material` - Material counting
- `use_pst` - Piece-square tables
- `use_tapered_eval` - Midgame/endgame interpolation
- `use_pawn_structure` - Pawn structure analysis
- `use_mobility` - Piece mobility evaluation
- `use_king_safety` - King safety evaluation

**❌ MISSING (5/11):**
- `use_bitboards` - Bitboard-specific optimizations
- `use_see` - Static exchange evaluation for evaluation
- `use_endgame_tables` - Endgame table knowledge
- `use_eval_tuning` - Evaluation parameter tuning
- `use_eval_caching` - Evaluation result caching

## Detailed Implementation Tasks

### Search Engine Updates (26 todos)
- [ ] **search_check_extensions**: Implement check extensions in search recursive functions (minimax.py, explorer.py)
- [ ] **search_quiescence_search**: Implement quiescence search function (_quiescence_search) in minimax.py and explorer.py
- [ ] **search_delta_pruning**: Add delta pruning logic to quiescence search (requires quiescence search)
- [ ] **search_see_pruning_qs**: Add SEE pruning logic to quiescence search (requires quiescence search + SEE)
- [ ] **search_futility_pruning**: Implement futility pruning logic in search recursive functions
- [ ] **search_extended_futility_pruning**: Implement extended futility pruning logic
- [ ] **search_reverse_futility_pruning**: Implement reverse futility pruning (static null move pruning)
- [ ] **search_null_move_pruning**: Implement null move pruning logic in search recursive functions
- [ ] **search_aspiration_windows**: Implement aspiration windows around alpha-beta search
- [ ] **search_recapture_extensions**: Implement recapture extensions in search extensions
- [ ] **search_singular_extensions**: Implement singular extensions logic
- [ ] **search_iid**: Implement internal iterative deepening (requires IDDFS + TT)
- [ ] **search_probcut**: Implement ProbCut pruning algorithm
- [ ] **search_multicut_pruning**: Implement Multi-cut pruning algorithm
- [ ] **search_razoring**: Implement razoring logic in search
- [ ] **parallel_search_base**: Implement parallel search master switch and thread management
- [ ] **parallel_lazy_smp**: Implement Lazy SMP parallel search algorithm
- [ ] **parallel_ybwc**: Implement Young Brothers Wait Concept (YBWC) parallel search
- [ ] **parallel_dts**: Implement Dynamic Tree Splitting (DTS) parallel search
- [ ] **parallel_naive**: Implement naive parallel search (root split)
- [ ] **search_mtdf**: Implement MTD(f) search algorithm
- [ ] **opening_book_system**: Implement opening book system with book loading and probing
- [ ] **endgame_tablebases**: Implement endgame tablebase system with probing

### Class/Function Updates (22 todos)
- [ ] **update_minimax_class_extensions**: Update minimax.py Minimax class: add _check_extension() method
- [ ] **update_minimax_class_futility**: Update minimax.py Minimax class: add _is_futile() method for futility pruning
- [ ] **update_minimax_class_null_move**: Update minimax.py Minimax class: add _null_move_search() method
- [ ] **update_minimax_class_aspiration**: Update minimax.py Minimax class: add _aspiration_search() wrapper method
- [ ] **update_explorer_class_extensions**: Update explorer.py Explorer class: add _check_extension() method
- [ ] **update_explorer_class_futility**: Update explorer.py Explorer class: add _is_futile() method for futility pruning
- [ ] **update_explorer_class_null_move**: Update explorer.py Explorer class: add _null_move_search() method
- [ ] **update_explorer_class_aspiration**: Update explorer.py Explorer class: add _aspiration_search() wrapper method
- [ ] **update_minimax_search_recursive_extensions**: Update minimax.py _search_recursive: integrate check extensions logic
- [ ] **update_explorer_search_recursive_extensions**: Update explorer.py _search_recursive: integrate check extensions logic
- [ ] **update_minimax_search_recursive_futility**: Update minimax.py _search_recursive: integrate futility pruning logic
- [ ] **update_explorer_search_recursive_futility**: Update explorer.py _search_recursive: integrate futility pruning logic
- [ ] **update_minimax_search_recursive_null_move**: Update minimax.py _search_recursive: integrate null move pruning logic
- [ ] **update_explorer_search_recursive_null_move**: Update explorer.py _search_recursive: integrate null move pruning logic
- [ ] **update_minimax_find_top_move_aspiration**: Update minimax.py find_top_move: wrap with aspiration window logic
- [ ] **update_explorer_search_aspiration**: Update explorer.py search: wrap with aspiration window logic
- [ ] **update_move_ordering_see**: Update move_ordering.py MoveOrderer class: enhance _calculate_see() for evaluation use
- [ ] **update_handcoded_eval_see**: Update handcoded_eval.py HandcodedEvaluator class: add _evaluate_see() method
- [ ] **update_handcoded_eval_endgame**: Update handcoded_eval.py HandcodedEvaluator class: add _evaluate_endgame_tables() method
- [ ] **update_handcoded_eval_caching**: Update handcoded_eval.py HandcodedEvaluator class: add evaluation caching logic
- [ ] **update_handcoded_eval_bitboards**: Update handcoded_eval.py HandcodedEvaluator class: add bitboard-specific optimizations
- [ ] **update_config_dependency_resolver**: Update config_dependency_resolver.py: add validation for new advanced features

### New Files to Create (7 todos)
- [ ] **new_quiescence_search_file**: Create new file: `src/engine/search/quiescence_search.py`
- [ ] **new_extensions_file**: Create new file: `src/engine/search/extensions.py` (for all extension types)
- [ ] **new_pruning_file**: Create new file: `src/engine/search/pruning.py` (for advanced pruning techniques)
- [ ] **new_parallel_search_file**: Create new file: `src/engine/search/parallel_search.py`
- [ ] **new_opening_book_file**: Create new file: `src/engine/opening_book.py`
- [ ] **new_endgame_tablebases_file**: Create new file: `src/engine/endgame_tablebases.py`
- [ ] **new_eval_tuning_file**: Create new file: `src/engine/evaluators/eval_tuning.py`

### Evaluation Feature Implementation (2 todos)
- [ ] **eval_bitboards_full**: Implement bitboard-specific evaluation optimizations in handcoded_eval.py (not just move ordering)
- [ ] **eval_see_full**: Implement SEE for evaluation beyond move ordering in handcoded_eval.py

### Tests to Create (8 todos)
- [ ] **update_dependency_resolver_tests**: Update `tests/search/config_dependency_resolver_test.py`: add tests for new features
- [ ] **create_quiescence_search_tests**: Create comprehensive tests for quiescence search functionality
- [ ] **create_pruning_tests**: Create comprehensive tests for all pruning techniques
- [ ] **create_extensions_tests**: Create tests for all extension types
- [ ] **create_parallel_search_tests**: Create tests for parallel search implementations
- [ ] **create_opening_book_tests**: Create tests for opening book system
- [ ] **create_endgame_tablebases_tests**: Create tests for endgame tablebases
- [ ] **create_eval_tuning_tests**: Create tests for evaluation tuning system

### Module Exports & Documentation (6 todos)
- [ ] **update_search_init_exports**: Update `src/engine/search/__init__.py` to export new search classes
- [ ] **update_evaluators_init_exports**: Update `src/engine/evaluators/__init__.py` to export new evaluator classes
- [ ] **update_engine_init_exports**: Update `src/engine/__init__.py` to export new engine modules
- [ ] **update_config_docstrings**: Update docstrings in config.py to reflect implemented vs configured status
- [ ] **update_migration_plan**: Update MIGRATION_PLAN.md to reflect current implementation gaps
- [ ] **update_readme_implemented_features**: Update README.md to accurately reflect implemented features

## Implementation Priority Recommendations

### Phase 1: Core Search Enhancements (High Impact)
1. Quiescence search - Foundation for many pruning techniques
2. Futility pruning - Simple but effective pruning
3. Null move pruning - High impact search optimization
4. Aspiration windows - Improves search efficiency

### Phase 2: Advanced Pruning (Medium-High Impact)
1. Delta pruning, SEE pruning in QS - Requires quiescence search
2. ProbCut, Multi-cut pruning - Advanced algorithms
3. Razoring - Reduces search tree

### Phase 3: Extensions & Refinements (Medium Impact)
1. Check extensions - Improves tactical strength
2. Recapture/singular extensions - Fine-tuning
3. Internal iterative deepening - Optimization

### Phase 4: External Knowledge & Parallelism (Variable Impact)
1. Opening book - User experience improvement
2. Endgame tablebases - Perfect play in endgames
3. Parallel search - Performance scaling

### Phase 5: Evaluation Enhancements (Low-Medium Impact)
1. SEE for evaluation - Tactical evaluation
2. Bitboard optimizations - Performance
3. Endgame tables - Specialized knowledge
4. Evaluation tuning/caching - Optimization

## Notes
- All features respect the dependency tree structure defined in config.py
- Implementation order should follow dependency requirements
- Each feature includes proper configuration validation
- Tests should cover both functionality and configuration integration
- Documentation updates ensure accurate feature status reporting
