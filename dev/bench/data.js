window.BENCHMARK_DATA = {
  "lastUpdate": 1774077258174,
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
      }
    ]
  }
}