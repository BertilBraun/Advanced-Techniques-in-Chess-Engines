# Monte-Carlo graph search screening

## Decision

Reject the current graph-search implementation for production use. Exact chess state identity produced too little
reuse to offset the implementation's graph maintenance overhead, and the disadvantage increased with search budget.
No strength benchmark was run.

## Method

- Source branch: `codex/chess-monte-carlo-graph-search`
- Benchmark harness commit: `3fac88e8`
- Model: generation 624 from `vast-chess-8gpu-1d-r4`
- Hardware: RTX 3060 device 0 on Vast instance 47400225
- Workload: 16 games, four parallel searches, maximum inference batch 256
- Measurement: at least 120 measured seconds per arm, completed only at ply boundaries
- Pair ordering alternated between tree-first and graph-first
- Budgets 100 through 3,200 ran alongside the confirmatory Stockfish evaluation

The confirmatory evaluation exited on its own at approximately 21:34 UTC. The matrix stopped when the service-status
probe returned nonzero. The unpaired 10,000-search tree result was produced after that load transition and is retained
as raw evidence but excluded from the comparison below. The user then stopped the remaining matrix because the current
implementation was already clearly unsuitable.

## Results

| Searches | Tree searches/s | Graph searches/s | Graph slowdown | TT hit rate | Evaluations avoided | Graph pruning time |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 4,121.5 | 3,407.1 | 17.3% | 0.47% | 0.25% | 0.03 s |
| 400 | 4,011.1 | 2,930.4 | 26.9% | 0.84% | 0.52% | 0.38 s |
| 1,000 | 3,552.2 | 2,447.9 | 31.1% | 1.59% | 1.08% | 2.66 s |
| 3,200 | 3,324.1 | 1,869.4 | 43.8% | 2.82% | 1.72% | 25.95 s |

Evaluation avoidance is `corrections / (corrections + neural evaluations)`. TT hit rate is verified semantic-identity
hits divided by probes.

## Diagnosed overhead

The implementation contains several avoidable scaling problems:

- peak-edge instrumentation scans every live node on every node creation and expansion;
- capacity pruning rescans every node and edge once per reclaimed leaf;
- rerooting performs mark/sweep and completely rebuilds the transposition table every ply;
- graph selection creates path vectors, attempted-edge vectors, and trajectory hash sets in the hot path;
- the transposition table stores a separately allocated vector for nearly every unique hash;
- benchmark graph counters are sampled before rerooting, so reroot time is included in wall time but omitted from the
  reported graph counters.

These problems explain substantial avoidable overhead, especially the measured pruning growth. They do not remove the
fundamental result: exact repetition-aware chess identity yielded only 0.25% to 1.72% avoided evaluations across the
completed budgets. Even a much cleaner graph implementation has little available throughput gain at this reuse rate.

