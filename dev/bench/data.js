window.BENCHMARK_DATA = {
  "lastUpdate": 1785167832921,
  "repoUrl": "https://github.com/wheres-perry/Moray",
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
          "id": "8f8fc6abdadf664e914d94753b2131fe9c549aa2",
          "message": "fix(tests): update regex for error message in test_move_ordering_dependencies",
          "timestamp": "2026-04-08T07:16:40Z",
          "tree_id": "eb7d91cb08b7319be49a2f58c310f3d4691e65a8",
          "url": "https://github.com/wheres-perry/ChessEngine/commit/8f8fc6abdadf664e914d94753b2131fe9c549aa2"
        },
        "date": 1775633158088,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_performance.py::test_full_game_cycle_300_ply",
            "value": 2385.880628094128,
            "unit": "iter/sec",
            "range": "stddev: 0.00003894056926129787",
            "extra": "mean: 419.13245290851484 usec\nrounds: 2166"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_perft_traversal_depth_3",
            "value": 8.544097083854734,
            "unit": "iter/sec",
            "range": "stddev: 0.0001607625379713249",
            "extra": "mean: 117.03986859999986 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_perft_traversal_depth_5",
            "value": 0.16580238695223307,
            "unit": "iter/sec",
            "range": "stddev: 0.03279978931138623",
            "extra": "mean: 6.031276258333335 sec\nrounds: 3"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_search_node_expansion_loop",
            "value": 79.91539053925258,
            "unit": "iter/sec",
            "range": "stddev: 0.0003906996176496478",
            "extra": "mean: 12.513234224999792 msec\nrounds: 80"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_promotion_heavy_movegen",
            "value": 125101.0503730913,
            "unit": "iter/sec",
            "range": "stddev: 9.612154451293603e-8",
            "extra": "mean: 7.993538000022228 usec\nrounds: 10"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_castling_and_ep_movegen",
            "value": 76392.91012690816,
            "unit": "iter/sec",
            "range": "stddev: 2.998098338952998e-7",
            "extra": "mean: 13.090219999980945 usec\nrounds: 10"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_bulk_push_pop_precomputed",
            "value": 59028.915314769714,
            "unit": "iter/sec",
            "range": "stddev: 3.872414571241044e-7",
            "extra": "mean: 16.940849999826924 usec\nrounds: 10"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_bulk_instantiation_1000",
            "value": 770.2617005432666,
            "unit": "iter/sec",
            "range": "stddev: 0.00002699479524568525",
            "extra": "mean: 1.2982600579708152 msec\nrounds: 690"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_single_move_toggle_50k",
            "value": 23.24557024785893,
            "unit": "iter/sec",
            "range": "stddev: 0.0004911934746427097",
            "extra": "mean: 43.018948958333546 msec\nrounds: 24"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_copy_chain_stress",
            "value": 1437.5934971636846,
            "unit": "iter/sec",
            "range": "stddev: 0.00005249278088632191",
            "extra": "mean: 695.6069305912698 usec\nrounds: 1167"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_batch_legal_generation",
            "value": 168.12142626322148,
            "unit": "iter/sec",
            "range": "stddev: 0.0005281448330065827",
            "extra": "mean: 5.948081825301298 msec\nrounds: 166"
          },
          {
            "name": "tests/benchmarks/test_search_metrics.py::test_search_metrics_full_suite",
            "value": 27.85349828657778,
            "unit": "iter/sec",
            "range": "stddev: 0.0002345338856687857",
            "extra": "mean: 35.90213300000045 msec\nrounds: 28"
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
          "id": "252ea0ca445a86ca696428f82304a2b8b95a5545",
          "message": "migrate: minimax, evaluators, transposition table and move ordering to C++",
          "timestamp": "2026-04-08T16:10:33Z",
          "tree_id": "d04d31020250bbc1f62d219067c5e7f4c5458adb",
          "url": "https://github.com/wheres-perry/ChessEngine/commit/252ea0ca445a86ca696428f82304a2b8b95a5545"
        },
        "date": 1775664989046,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_performance.py::test_full_game_cycle_300_ply",
            "value": 2599.8797824095127,
            "unit": "iter/sec",
            "range": "stddev: 0.000015322412289496797",
            "extra": "mean: 384.6331691049274 usec\nrounds: 2324"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_perft_traversal_depth_3",
            "value": 8.404122616800965,
            "unit": "iter/sec",
            "range": "stddev: 0.00045223488795473435",
            "extra": "mean: 118.98922060000245 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_perft_traversal_depth_5",
            "value": 0.16528796518550423,
            "unit": "iter/sec",
            "range": "stddev: 0.0368233568105724",
            "extra": "mean: 6.05004725466667 sec\nrounds: 3"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_search_node_expansion_loop",
            "value": 80.31562350434389,
            "unit": "iter/sec",
            "range": "stddev: 0.0005870004891433613",
            "extra": "mean: 12.450877629629742 msec\nrounds: 81"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_promotion_heavy_movegen",
            "value": 109076.5427184687,
            "unit": "iter/sec",
            "range": "stddev: 2.162898633305601e-7",
            "extra": "mean: 9.167874000013398 usec\nrounds: 10"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_castling_and_ep_movegen",
            "value": 66610.94439112312,
            "unit": "iter/sec",
            "range": "stddev: 1.852825630756331e-7",
            "extra": "mean: 15.012548000044035 usec\nrounds: 10"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_bulk_push_pop_precomputed",
            "value": 62926.108073685675,
            "unit": "iter/sec",
            "range": "stddev: 4.278343677807152e-7",
            "extra": "mean: 15.891655000004336 usec\nrounds: 10"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_bulk_instantiation_1000",
            "value": 734.3591741980081,
            "unit": "iter/sec",
            "range": "stddev: 0.00007509349661636216",
            "extra": "mean: 1.3617314730112788 msec\nrounds: 704"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_single_move_toggle_50k",
            "value": 24.680503307766905,
            "unit": "iter/sec",
            "range": "stddev: 0.0003289780513318445",
            "extra": "mean: 40.51781227999925 msec\nrounds: 25"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_copy_chain_stress",
            "value": 1615.518649747758,
            "unit": "iter/sec",
            "range": "stddev: 0.00005272684775549632",
            "extra": "mean: 618.9962586666127 usec\nrounds: 1500"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_batch_legal_generation",
            "value": 147.82658342822592,
            "unit": "iter/sec",
            "range": "stddev: 0.00007561897888763006",
            "extra": "mean: 6.764683163265618 msec\nrounds: 147"
          },
          {
            "name": "tests/benchmarks/test_search_metrics.py::test_search_metrics_full_suite",
            "value": 36.25294312847355,
            "unit": "iter/sec",
            "range": "stddev: 0.0007347941117441534",
            "extra": "mean: 27.58396736111024 msec\nrounds: 36"
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
          "id": "7414da3c368e7fb06a52843fdc10f769afee97f5",
          "message": "refactor: complete Moray rename, upgrade dependencies, and add elo-test CI pipeline",
          "timestamp": "2026-07-27T09:49:06-06:00",
          "tree_id": "c1712a0c604a9e6f6d06d7876135d9e055999666",
          "url": "https://github.com/wheres-perry/Moray/commit/7414da3c368e7fb06a52843fdc10f769afee97f5"
        },
        "date": 1785167831917,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_performance.py::test_full_game_cycle_300_ply",
            "value": 5325.481854752378,
            "unit": "iter/sec",
            "range": "stddev: 0.000004660050617972863",
            "extra": "mean: 187.7764354989991 usec\nrounds: 4248"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_perft_traversal_depth_3",
            "value": 16.5347124959059,
            "unit": "iter/sec",
            "range": "stddev: 0.0002824680879804997",
            "extra": "mean: 60.478826000004915 msec\nrounds: 5"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_perft_traversal_depth_5",
            "value": 0.3162020253957736,
            "unit": "iter/sec",
            "range": "stddev: 0.03933001371094776",
            "extra": "mean: 3.162535087333334 sec\nrounds: 3"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_search_node_expansion_loop",
            "value": 163.14031551713543,
            "unit": "iter/sec",
            "range": "stddev: 0.00011201368645912735",
            "extra": "mean: 6.129692693250707 msec\nrounds: 163"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_promotion_heavy_movegen",
            "value": 211586.29631949458,
            "unit": "iter/sec",
            "range": "stddev: 1.8182677660719378e-7",
            "extra": "mean: 4.72620399995094 usec\nrounds: 10"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_castling_and_ep_movegen",
            "value": 129752.03866384245,
            "unit": "iter/sec",
            "range": "stddev: 1.10559878249022e-7",
            "extra": "mean: 7.7070080000112275 usec\nrounds: 10"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_bulk_push_pop_precomputed",
            "value": 129615.94794028945,
            "unit": "iter/sec",
            "range": "stddev: 2.2513466075454422e-7",
            "extra": "mean: 7.715100000353914 usec\nrounds: 10"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_bulk_instantiation_1000",
            "value": 1738.0644225452534,
            "unit": "iter/sec",
            "range": "stddev: 0.00002284720618365208",
            "extra": "mean: 575.3526664653671 usec\nrounds: 1652"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_single_move_toggle_50k",
            "value": 52.51511556962826,
            "unit": "iter/sec",
            "range": "stddev: 0.0004989045824232616",
            "extra": "mean: 19.04213651922996 msec\nrounds: 52"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_copy_chain_stress",
            "value": 4559.427364289956,
            "unit": "iter/sec",
            "range": "stddev: 0.000019994422292489967",
            "extra": "mean: 219.3257881092993 usec\nrounds: 4205"
          },
          {
            "name": "tests/benchmarks/test_performance.py::test_batch_legal_generation",
            "value": 290.1019258713062,
            "unit": "iter/sec",
            "range": "stddev: 0.00006718131718172597",
            "extra": "mean: 3.4470643274654296 msec\nrounds: 284"
          },
          {
            "name": "tests/benchmarks/test_search_metrics.py::test_search_metrics_full_suite",
            "value": 60.08666407511269,
            "unit": "iter/sec",
            "range": "stddev: 0.0004830867605830001",
            "extra": "mean: 16.64262803389996 msec\nrounds: 59"
          }
        ]
      }
    ]
  }
}