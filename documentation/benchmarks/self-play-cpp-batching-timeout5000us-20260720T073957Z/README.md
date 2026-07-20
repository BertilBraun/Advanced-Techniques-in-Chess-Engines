# Exploratory 5 ms batching timeout

This run compares a 5,000 microsecond inference timeout with the post-fix
[`500 microsecond run`](../self-play-cpp-baseline-4x8x3x96-20260720T073130Z/summary.json)
using the same model, four-GPU topology, workload, and completed searches.

| Metric | 500 microseconds | 5,000 microseconds | Change |
| --- | ---: | ---: | ---: |
| Completed plies per second | 295.571 | 287.263 | -2.81% |
| Average inference batch size | 87.369 | 104.087 | +19.14% |
| Model inference calls | 95,221 | 79,927 | -16.06% |

The longer timeout produced larger batches but reduced throughput relative to
the post-fix 500 microsecond run. It remained 2.26% faster than the
[`original baseline`](../self-play-cpp-baseline-4x8x3x96-20260720T071550Z/summary.json)
in completed plies per second.

This is a single exploratory run. The 500 microsecond timeout remains
provisional, and final timeout tuning is explicitly deferred until the
prepare/submit/resolve pipeline and MCTS optimizations are complete.
