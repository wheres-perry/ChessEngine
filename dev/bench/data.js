window.BENCHMARK_DATA = {
  "lastUpdate": 1774252944020,
  "repoUrl": "https://github.com/wheres-perry/ChessEngine",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "email": "86326912+wheres-perry@users.noreply.github.com",
            "name": "Ethan Perry",
            "username": "wheres-perry"
          },
          "committer": {
            "email": "86326912+wheres-perry@users.noreply.github.com",
            "name": "Ethan Perry",
            "username": "wheres-perry"
          },
          "distinct": true,
          "id": "d9447913d470949a66258d1c62e605cc1f43a2f4",
          "message": "Update git ignore to exclude output.json",
          "timestamp": "2026-03-21T00:21:15-06:00",
          "tree_id": "9934098fdf0439c15bc9f2a5c7a99130b1545cf3",
          "url": "https://github.com/wheres-perry/ChessEngine/commit/d9447913d470949a66258d1c62e605cc1f43a2f4"
        },
        "date": 1774074436940,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_performance.py::test_full_game_cycle_300_ply",
            "value": 2716.859498483975,
            "unit": "iter/sec",
            "range": "stddev: 0.000029973994677275183",
            "extra": "mean: 368.07203337456593 usec\nrounds: 2427"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_perft_traversal_depth_3",
            "value": 9.01772886035053,
            "unit": "iter/sec",
            "range": "stddev: 0.0005332528104010417",
            "extra": "mean: 110.89266659999453 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_perft_traversal_depth_5",
            "value": 0.17531715794579675,
            "unit": "iter/sec",
            "range": "stddev: 0.019725407596847825",
            "extra": "mean: 5.703948271333331 sec\nrounds: 3"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_search_node_expansion_loop",
            "value": 86.34306757735251,
            "unit": "iter/sec",
            "range": "stddev: 0.00010716437019657106",
            "extra": "mean: 11.581705724135015 msec\nrounds: 87"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_promotion_heavy_movegen",
            "value": 123310.03595914064,
            "unit": "iter/sec",
            "range": "stddev: 2.0916212511700464e-7",
            "extra": "mean: 8.109639999872796 usec\nrounds: 10"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_castling_and_ep_movegen",
            "value": 75176.69907191866,
            "unit": "iter/sec",
            "range": "stddev: 1.5064872214906174e-7",
            "extra": "mean: 13.301994000073591 usec\nrounds: 10"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_bulk_push_pop_precomputed",
            "value": 67639.34296378763,
            "unit": "iter/sec",
            "range": "stddev: 3.4085999456931274e-7",
            "extra": "mean: 14.784295000254133 usec\nrounds: 10"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_bulk_instantiation_1000",
            "value": 722.1540470935048,
            "unit": "iter/sec",
            "range": "stddev: 0.00009354896890275651",
            "extra": "mean: 1.3847460995680325 msec\nrounds: 693"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_single_move_toggle_50k",
            "value": 26.962884433062236,
            "unit": "iter/sec",
            "range": "stddev: 0.0008097301715944658",
            "extra": "mean: 37.08802010714355 msec\nrounds: 28"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_copy_chain_stress",
            "value": 1585.4369201722914,
            "unit": "iter/sec",
            "range": "stddev: 0.00010432742681683432",
            "extra": "mean: 630.7409568154429 usec\nrounds: 1482"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_batch_legal_generation",
            "value": 166.21141846028837,
            "unit": "iter/sec",
            "range": "stddev: 0.0002235891613160499",
            "extra": "mean: 6.016433824243683 msec\nrounds: 165"
          },
          {
            "name": "tests/benchmarks/test_search_metrics.py::test_search_metrics_full_suite",
            "value": 30.522583212307794,
            "unit": "iter/sec",
            "range": "stddev: 0.00021918342828630057",
            "extra": "mean: 32.76262670968047 msec\nrounds: 31"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "86326912+wheres-perry@users.noreply.github.com",
            "name": "Ethan Perry",
            "username": "wheres-perry"
          },
          "committer": {
            "email": "86326912+wheres-perry@users.noreply.github.com",
            "name": "Ethan Perry",
            "username": "wheres-perry"
          },
          "distinct": true,
          "id": "83025aadbddf74487774e243fcfa3cb889e64b1e",
          "message": "Unify dockerfiles and clean up build process",
          "timestamp": "2026-03-21T07:08:03Z",
          "tree_id": "1208f92041dfc77e1748732efaff78ec9a245748",
          "url": "https://github.com/wheres-perry/ChessEngine/commit/83025aadbddf74487774e243fcfa3cb889e64b1e"
        },
        "date": 1774077256984,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_performance.py::test_full_game_cycle_300_ply",
            "value": 2723.559897512401,
            "unit": "iter/sec",
            "range": "stddev: 0.00003862561566851534",
            "extra": "mean: 367.16651648210967 usec\nrounds: 2275"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_perft_traversal_depth_3",
            "value": 9.024067136519854,
            "unit": "iter/sec",
            "range": "stddev: 0.0005629969551564602",
            "extra": "mean: 110.81477839998115 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_perft_traversal_depth_5",
            "value": 0.1776463670385498,
            "unit": "iter/sec",
            "range": "stddev: 0.031965525618911424",
            "extra": "mean: 5.629160993666687 sec\nrounds: 3"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_search_node_expansion_loop",
            "value": 86.46238510331732,
            "unit": "iter/sec",
            "range": "stddev: 0.0003484029498601115",
            "extra": "mean: 11.56572304598191 msec\nrounds: 87"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_promotion_heavy_movegen",
            "value": 119243.60443582115,
            "unit": "iter/sec",
            "range": "stddev: 4.3407821814884584e-7",
            "extra": "mean: 8.386193999513125 usec\nrounds: 10"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_castling_and_ep_movegen",
            "value": 75549.44847378983,
            "unit": "iter/sec",
            "range": "stddev: 1.45389564621456e-7",
            "extra": "mean: 13.23636400002215 usec\nrounds: 10"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_bulk_push_pop_precomputed",
            "value": 67722.00118731389,
            "unit": "iter/sec",
            "range": "stddev: 3.441987379423949e-7",
            "extra": "mean: 14.766249999524916 usec\nrounds: 10"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_bulk_instantiation_1000",
            "value": 737.4863321786593,
            "unit": "iter/sec",
            "range": "stddev: 0.000041361441280168526",
            "extra": "mean: 1.3559573328577237 msec\nrounds: 703"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_single_move_toggle_50k",
            "value": 26.949429046053673,
            "unit": "iter/sec",
            "range": "stddev: 0.0002922706826438142",
            "extra": "mean: 37.1065375185169 msec\nrounds: 27"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_copy_chain_stress",
            "value": 1659.657413969289,
            "unit": "iter/sec",
            "range": "stddev: 0.00008408757462259755",
            "extra": "mean: 602.5339877874967 usec\nrounds: 1474"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_batch_legal_generation",
            "value": 166.9555802991953,
            "unit": "iter/sec",
            "range": "stddev: 0.00007551999208992794",
            "extra": "mean: 5.989617107783608 msec\nrounds: 167"
          },
          {
            "name": "tests/benchmarks/test_search_metrics.py::test_search_metrics_full_suite",
            "value": 29.845328138873132,
            "unit": "iter/sec",
            "range": "stddev: 0.00023688270375431857",
            "extra": "mean: 33.50608160000471 msec\nrounds: 30"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "86326912+wheres-perry@users.noreply.github.com",
            "name": "Ethan Perry",
            "username": "wheres-perry"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "b4824a339fe953c559de7e0103056d05fd1ff430",
          "message": "Feat/ml extractors and pgn (#6)\n\n* feat: add C++ ML feature extractors and high-performance PGN parser\n\n- Implement zero-copy C++ extractors for CNN, NNUE (HalfKP), and GNN formats\n\n- Add fast, memory-efficient PGN stream parser\n\n- Expose extractors and PGN parser to Python via PyBind11\n\n- Add comprehensive test suites for extractors and PGN parser\n\n- Include rigorous PGN parity tests against `python-chess`\n\n* test: refactor test suite structure, parameterize core tests, and expand uci coverage\n\n- Break monolithic test_core_engine.py into parameterized core tests\n\n- Move tests from unit dumping ground to domain-specific directories\n\n- Add stdin/stdout mock tests to UCI handler covering missing branches\n\n- Remove empty conftest.py files\n\n* style: fix ruff linting and formatting issues in test suite\n\n* Update src/engine/_cpp/extractors/bindings.cpp\n\nCo-authored-by: Copilot <175728472+Copilot@users.noreply.github.com>\n\n* style: fix C++ code formatting to pass clang-format CI check\n\n* style: fix include order in extractors/bindings.cpp\n\n* ci: fix noxfile test paths after directory restructure\n\n* docs: add comprehensive style and naming guides\n\n---------\n\nCo-authored-by: Copilot <175728472+Copilot@users.noreply.github.com>",
          "timestamp": "2026-03-21T03:39:50-06:00",
          "tree_id": "a8db626698439f439182c9caeb75f6b9077913c6",
          "url": "https://github.com/wheres-perry/ChessEngine/commit/b4824a339fe953c559de7e0103056d05fd1ff430"
        },
        "date": 1774086387285,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_performance.py::test_full_game_cycle_300_ply",
            "value": 2682.5266518288313,
            "unit": "iter/sec",
            "range": "stddev: 0.000097193988486751",
            "extra": "mean: 372.78287591970167 usec\nrounds: 2450"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_perft_traversal_depth_3",
            "value": 8.70194211928179,
            "unit": "iter/sec",
            "range": "stddev: 0.011855205375120371",
            "extra": "mean: 114.91687560001083 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_perft_traversal_depth_5",
            "value": 0.17872875774137625,
            "unit": "iter/sec",
            "range": "stddev: 0.013730558775151844",
            "extra": "mean: 5.59507050033335 sec\nrounds: 3"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_search_node_expansion_loop",
            "value": 82.83754287324493,
            "unit": "iter/sec",
            "range": "stddev: 0.0019450459325810613",
            "extra": "mean: 12.071821124996001 msec\nrounds: 88"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_promotion_heavy_movegen",
            "value": 120456.99456448622,
            "unit": "iter/sec",
            "range": "stddev: 6.447071908726987e-7",
            "extra": "mean: 8.301717999984248 usec\nrounds: 10"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_castling_and_ep_movegen",
            "value": 73063.01187936706,
            "unit": "iter/sec",
            "range": "stddev: 2.843844916416917e-7",
            "extra": "mean: 13.686816000017643 usec\nrounds: 10"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_bulk_push_pop_precomputed",
            "value": 68719.64309659223,
            "unit": "iter/sec",
            "range": "stddev: 3.728567235627073e-7",
            "extra": "mean: 14.551880000226447 usec\nrounds: 10"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_bulk_instantiation_1000",
            "value": 728.0288795383251,
            "unit": "iter/sec",
            "range": "stddev: 0.000047767763698514786",
            "extra": "mean: 1.373571884447968 msec\nrounds: 701"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_single_move_toggle_50k",
            "value": 27.474815602774065,
            "unit": "iter/sec",
            "range": "stddev: 0.000903166097002389",
            "extra": "mean: 36.396968571429916 msec\nrounds: 28"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_copy_chain_stress",
            "value": 1633.656724779455,
            "unit": "iter/sec",
            "range": "stddev: 0.0000615724321292324",
            "extra": "mean: 612.1237006721842 usec\nrounds: 1490"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_batch_legal_generation",
            "value": 164.11032797432833,
            "unit": "iter/sec",
            "range": "stddev: 0.00004900633405883705",
            "extra": "mean: 6.093461711662835 msec\nrounds: 163"
          },
          {
            "name": "tests/benchmarks/test_search_metrics.py::test_search_metrics_full_suite",
            "value": 30.680256577670928,
            "unit": "iter/sec",
            "range": "stddev: 0.00016689119370425127",
            "extra": "mean: 32.5942515333395 msec\nrounds: 30"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "86326912+wheres-perry@users.noreply.github.com",
            "name": "Ethan Perry",
            "username": "wheres-perry"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "0972801128b4287cfe3c26889741ecb39f1d0716",
          "message": "feat(elo): add standalone deterministic Elo estimation subsystem (#7)\n\n* feat(elo): add standalone deterministic Elo estimation subsystem\n\nIntroduce a self-contained elo_tests package with deterministic paired scheduling, simulated engine adapters, JSONL raw game logging, Elo point estimation, normal and paired-bootstrap confidence intervals, sequential stopping, and CLI/config/reporting support.\n\n* fix: resolve style guide issues, runtime errors, and docstring violations\n\n- Fix TypeError in PST tables caused by misplaced string literals.\n- Enable and satisfy strict docstring (pydocstyle) rules project-wide.\n- Move type-only imports into TYPE_CHECKING blocks to resolve TC001/TC003 failures.\n- Resolve undefined name 'Searcher' in Engine initialization.\n- Shorten long lines to comply with 88-character limit.\n- Update TODO.md with project roadmap and maintenance task completion.",
          "timestamp": "2026-03-22T14:51:03-06:00",
          "tree_id": "2216697e688b63ec1dfeecc34f7ce26bf783a457",
          "url": "https://github.com/wheres-perry/ChessEngine/commit/0972801128b4287cfe3c26889741ecb39f1d0716"
        },
        "date": 1774213077415,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_performance.py::test_full_game_cycle_300_ply",
            "value": 2626.736078987747,
            "unit": "iter/sec",
            "range": "stddev: 0.00006316081497294919",
            "extra": "mean: 380.70059950041315 usec\nrounds: 2402"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_perft_traversal_depth_3",
            "value": 8.867772507575618,
            "unit": "iter/sec",
            "range": "stddev: 0.000819593767951151",
            "extra": "mean: 112.76789060000283 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_perft_traversal_depth_5",
            "value": 0.17395474973961292,
            "unit": "iter/sec",
            "range": "stddev: 0.03986579664873935",
            "extra": "mean: 5.748621417333339 sec\nrounds: 3"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_search_node_expansion_loop",
            "value": 85.92101642025122,
            "unit": "iter/sec",
            "range": "stddev: 0.00014858735179540056",
            "extra": "mean: 11.638596022990066 msec\nrounds: 87"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_promotion_heavy_movegen",
            "value": 121311.02278142961,
            "unit": "iter/sec",
            "range": "stddev: 2.03903589211461e-7",
            "extra": "mean: 8.243273999937628 usec\nrounds: 10"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_castling_and_ep_movegen",
            "value": 73824.69970684049,
            "unit": "iter/sec",
            "range": "stddev: 1.7925088331789914e-7",
            "extra": "mean: 13.545602000021972 usec\nrounds: 10"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_bulk_push_pop_precomputed",
            "value": 68457.09348890783,
            "unit": "iter/sec",
            "range": "stddev: 2.1731073231234683e-7",
            "extra": "mean: 14.607690000190132 usec\nrounds: 10"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_bulk_instantiation_1000",
            "value": 722.8216800624461,
            "unit": "iter/sec",
            "range": "stddev: 0.000045140924822467304",
            "extra": "mean: 1.3834670812773737 msec\nrounds: 689"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_single_move_toggle_50k",
            "value": 26.606034680080267,
            "unit": "iter/sec",
            "range": "stddev: 0.00015950728406995251",
            "extra": "mean: 37.58545803703294 msec\nrounds: 27"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_copy_chain_stress",
            "value": 1633.3630033082109,
            "unit": "iter/sec",
            "range": "stddev: 0.00004662776623600035",
            "extra": "mean: 612.2337765546309 usec\nrounds: 1544"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_batch_legal_generation",
            "value": 163.81656587331102,
            "unit": "iter/sec",
            "range": "stddev: 0.00019745633312539835",
            "extra": "mean: 6.104388739130075 msec\nrounds: 161"
          },
          {
            "name": "tests/benchmarks/test_search_metrics.py::test_search_metrics_full_suite",
            "value": 30.25789328719916,
            "unit": "iter/sec",
            "range": "stddev: 0.0006007691400377753",
            "extra": "mean: 33.04922753571406 msec\nrounds: 28"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "86326912+wheres-perry@users.noreply.github.com",
            "name": "Ethan Perry",
            "username": "wheres-perry"
          },
          "committer": {
            "email": "86326912+wheres-perry@users.noreply.github.com",
            "name": "Ethan Perry",
            "username": "wheres-perry"
          },
          "distinct": true,
          "id": "1290f91168e2e7651e67efb745bb626817baad8e",
          "message": "fix: update README for test command and fix styling/linting errors",
          "timestamp": "2026-03-23T07:54:49Z",
          "tree_id": "d13c48dd478046d622937729efd160bfa6062505",
          "url": "https://github.com/wheres-perry/ChessEngine/commit/1290f91168e2e7651e67efb745bb626817baad8e"
        },
        "date": 1774252942864,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_performance.py::test_full_game_cycle_300_ply",
            "value": 2457.889190742309,
            "unit": "iter/sec",
            "range": "stddev: 0.000008499215564417949",
            "extra": "mean: 406.85316643505365 usec\nrounds: 2157"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_perft_traversal_depth_3",
            "value": 8.162270561879673,
            "unit": "iter/sec",
            "range": "stddev: 0.001555405789679196",
            "extra": "mean: 122.51492920000828 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_perft_traversal_depth_5",
            "value": 0.16153136500709606,
            "unit": "iter/sec",
            "range": "stddev: 0.01369415079029391",
            "extra": "mean: 6.190748155666673 sec\nrounds: 3"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_search_node_expansion_loop",
            "value": 78.32723376030238,
            "unit": "iter/sec",
            "range": "stddev: 0.00006442775215199348",
            "extra": "mean: 12.766951569618914 msec\nrounds: 79"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_promotion_heavy_movegen",
            "value": 118323.05934276477,
            "unit": "iter/sec",
            "range": "stddev: 8.020973590426798e-8",
            "extra": "mean: 8.45143799995185 usec\nrounds: 10"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_castling_and_ep_movegen",
            "value": 72730.70032687175,
            "unit": "iter/sec",
            "range": "stddev: 8.940058321740455e-8",
            "extra": "mean: 13.749351999990722 usec\nrounds: 10"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_bulk_push_pop_precomputed",
            "value": 58069.6993184382,
            "unit": "iter/sec",
            "range": "stddev: 2.207275741230612e-7",
            "extra": "mean: 17.220684999870173 usec\nrounds: 10"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_bulk_instantiation_1000",
            "value": 806.1682522203776,
            "unit": "iter/sec",
            "range": "stddev: 0.000037721131338660623",
            "extra": "mean: 1.2404358485288451 msec\nrounds: 713"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_single_move_toggle_50k",
            "value": 23.070689221409335,
            "unit": "iter/sec",
            "range": "stddev: 0.0009655895836670456",
            "extra": "mean: 43.34504229167161 msec\nrounds: 24"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_copy_chain_stress",
            "value": 1574.2695379191778,
            "unit": "iter/sec",
            "range": "stddev: 0.0000966389485688487",
            "extra": "mean: 635.2152385046909 usec\nrounds: 1283"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_batch_legal_generation",
            "value": 159.42373024508905,
            "unit": "iter/sec",
            "range": "stddev: 0.0002614180970434234",
            "extra": "mean: 6.272591906252956 msec\nrounds: 160"
          },
          {
            "name": "tests/benchmarks/test_search_metrics.py::test_search_metrics_full_suite",
            "value": 29.640987499809228,
            "unit": "iter/sec",
            "range": "stddev: 0.00022906807454930049",
            "extra": "mean: 33.73706763333833 msec\nrounds: 30"
          }
        ]
      }
    ]
  }
}