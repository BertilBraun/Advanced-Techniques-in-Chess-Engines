# Inference batching benchmark

This run measures the enqueue-and-deadline batching change at a 500 microsecond
inference timeout. It uses the same four-GPU topology, model, workload, and
17,946,496 completed searches as the
[`20260720T071550Z`](../self-play-cpp-baseline-4x8x3x96-20260720T071550Z/summary.json)
baseline.

| Metric | Before | After | Change |
| --- | ---: | ---: | ---: |
| Searches per second | 164,106 | 172,672 | +5.2% |
| Completed plies per second | 280.9 | 295.6 | +5.2% |
| Makespan | 109.36 s | 103.93 s | -5.0% |
| Average inference batch size | 62.38 | 87.37 | +40.1% |
| Model inference calls | 133,370 | 95,221 | -28.6% |

The cache hit rate and number of evaluated positions were identical. This is one
before/after run on one node, so the throughput delta should be confirmed with
repeated measurements before treating it as a stable effect.

Validation on the benchmark revision: CTest passed 1/1. Pytest passed 125 tests
but reported two collection/setup errors because the speed-test functions require
unprovided `num_boards` and `num_iterations` fixtures.
