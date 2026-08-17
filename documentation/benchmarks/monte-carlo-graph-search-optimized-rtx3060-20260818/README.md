# Optimized Monte-Carlo graph-search screening

## Decision

The optimized implementation is suitable for strength evaluation. It is not yet accepted for production or
training. The sustained throughput cost was 0.17% at 1,000 searches and 1.67% at 3,200 searches, replacing the
31.1% and 43.8% slowdowns measured before the graph hot-path and pruning cleanup. No strength benchmark was run.

## State equivalence

Chess graph identity remains rule-exact while no longer requiring the reversible history to have the same order.
Two states merge only when they have the same current pieces, side to move, castling rights, effective en-passant
state, exact halfmove clock, and the same multiset of exact repetition positions. The multiset preserves every
position's occurrence count. Order is not a chess rule input: future repetition adjudication depends on how many
times a future current position already occurred, while irreversible moves reset the retained context.

This is deliberately stricter than board-only or Zobrist-only merging. A different repetition-position multiset,
halfmove clock, castling state, or en-passant state still produces a distinct graph node. The transposition table
uses a hash only to locate candidates and verifies semantic equality before linking.

## Method

- Source revision: `0bbc8cf7`
- Model: generation 624, SHA-256 `c95a67f478c0dc69ac0c5e12661d98c2ae524932c499e070f6321cfe5824c864`
- Hardware: RTX 3060 device 1 on Vast instance 47400225
- Workload: 16 games, four parallel searches, maximum inference batch 256
- Starting positions: the final FEN column of the first 16 entries in
  `py/reference/chess-stockfish-8moves-v3-openings-50.tsv`
- Root history: each FEN starts with a fresh pre-root repetition history; two warm-up plies precede measurement
- Measurement: at least 120 measured seconds per arm, completed at whole-ply boundaries
- Arm order: 1,000 tree, 1,000 graph, 3,200 graph, 3,200 tree
- Graph correction threshold: 0.01

The prior confirmatory evaluator was not running. An unrelated attention-backend diagnostic briefly occupied GPU 0
before the matrix, so the matrix used otherwise-idle GPU 1. This is a different load regime from the original
screening; only matched arms in this directory are compared.

## Results

| Searches | Tree searches/s | Graph searches/s | Graph slowdown | TT hit rate | Evaluations avoided | Pruning | Rerooting |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 4,667.0 | 4,658.9 | 0.17% | 1.21% | 0.79% | 0.138 s | 0.794 s |
| 3,200 | 4,075.0 | 4,006.9 | 1.67% | 2.32% | 1.35% | 0.242 s | 1.092 s |

TT hit rate is verified semantic-identity hits divided by probes. Evaluation avoidance is graph correction backups
divided by correction backups plus performed neural evaluations. Pruning and rerooting are cumulative wall-clock
time within each graph arm.

The optimized 3,200-search graph arm performed 479,017 neural evaluations versus 502,451 for the tree arm, although
the independently evolving searches do not visit identical positions and this difference is not a controlled
per-position saving estimate. Raw counters are retained in the JSON artifacts.

## Interpretation and review risks

- The pathological overhead was implementation-caused. Incremental edge accounting, linearithmic graph pruning,
  allocation-free trajectory reuse, and direct multimap buckets removed nearly all of it.
- Broader identity is confirmed by deterministic native tests for order-independent, count-preserving repetition
  contexts. These throughput runs do not isolate its incremental hit-rate effect from changed search trajectories.
- A single 120-second pair at each budget is a screening result, not a confidence interval. Repeat pairs are needed
  before making a small overhead claim precise.
- Throughput viability does not establish playing strength. A matched tree-versus-graph strength evaluation is the
  next acceptance gate.
- The opening harness reconstructs roots from FEN, so it does not preserve the actual eight-ply opening history.
  This is consistent across arms but can understate reuse patterns present in retained self-play roots.

## Artifacts

- `01-1000-tree.json`, `02-1000-graph.json`, `03-3200-graph.json`, `04-3200-tree.json`: sustained arms
- `smoke-1000-graph.json`: five-second preflight smoke
- `post-run-state.txt`: source/model identity and post-run GPU/process state
