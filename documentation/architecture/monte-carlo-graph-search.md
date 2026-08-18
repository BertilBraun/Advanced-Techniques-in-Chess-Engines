# Monte-Carlo graph search

## Scope and source

The native engine provides two explicit search algorithms. Monte-Carlo tree search remains the default and the
scientific control. Monte-Carlo graph search is opt-in through an `algorithm` discriminated configuration and uses
canonical shared state nodes rather than a neural-evaluation cache.

The design follows Johannes Czech, Patrick Korus, and Kristian Kersting, “Improving AlphaZero Using Monte-Carlo
Graph Search,” *Proceedings of ICAPS 31* (2021), pp. 103–111,
[doi:10.1609/icaps.v31i1.15952](https://doi.org/10.1609/icaps.v31i1.15952),
[arXiv:2012.11045](https://arxiv.org/abs/2012.11045). In particular:

- one canonical node holds the shared state value, visits, policy, and descendants;
- each incoming parent/action edge retains its own value and visit statistics;
- selection records the exact traversed trajectory for reservation and reverse backup;
- a transposition edge is corrected toward the more informed shared-node value when their estimates diverge;
- only the traversed trajectory is backed up, rather than enumerating every possible parent path.

This is the node-and-edge scheme in the paper’s “Data-Structure” section and Algorithms 1–2. The implementation uses
the existing AlphaZero PUCT, FPU, forced-root-playout, virtual-loss, batching, and value-discount conventions for a
controlled comparison.

## Correction invariant

The printed paper defines the residual as `edge - target` in Equation 8, then uses that residual with a sign in
Equation 9 that would move an ordinary simple-moving-average update away from the target. Its prose also describes
an absolute threshold while one pseudocode branch omits the absolute value. The native implementation makes the
stated invariant explicit:

```text
target = shared_child_value
correction = (edge_visits + 1) * target - edge_value_sum
```

Before clipping, adding this one sample makes the incoming edge mean exactly equal to the shared-node target. The
sample is clipped to the game value range `[-1, 1]`, and both residual signs are compared by absolute magnitude.
Deterministic native tests cover the unclipped equality and clipped monotonic behavior.

`transposition_value_threshold` controls when that correction replaces further descent. It is not a state-identity
or hash tolerance: the child must already be an equality-verified transposition. Once both the shared child and the
new incoming edge have completed visits, the engine compares their child-perspective means. A difference greater
than the threshold stops that simulation at the shared node and backs up the correction without neural inference;
a difference at or below the threshold continues selection through the shared subtree. On the `[-1, 1]` value
scale, the default `0.01` is one percentage point. Zero corrects every non-identical estimate, while a larger value
continues through more transpositions. Canonical topology and shared node statistics exist independently of this
choice.

## State identity and cycles

Generic graph search obtains both hashing and collision-checking equality from the game contract. It never inspects
chess fields or uses packed neural inputs as identities.

Chess identity contains the current pieces, side to move, castling rights, en-passant state, exact halfmove clock,
and a canonical multiset of retained repetition positions. History order is excluded because future threefold
semantics depend on occurrence counts rather than traversal order. The fullmove display counter is excluded because
it has no rule or network meaning. Go identity contains every retained black/white history board, player, ko point,
consecutive passes, move number, komi, and maximum-move rule.

Czech et al. put a step counter in their transposition key to guarantee a DAG. This engine instead uses the complete
semantic state and also checks the active trajectory before creating a link. A would-be back-edge is not installed;
the game contract supplies its cycle value, the selected trajectory is backed up once, and cycle instrumentation is
incremented. Exact chess and Go recurrence normally differs through their semantic history or move counters, but
the guard keeps the generic graph implementation bounded.

## Concurrency, retention, and pruning

The tree owner serializes public search/evaluation mutations while inference workers remain asynchronous. Every
in-flight graph simulation owns its complete path. Reservation applies one virtual visit/loss to every selected node
and edge; completion or cancellation removes it exactly once. A canonical unexpanded node can have only one pending
inference, so another path can link to it but cannot enqueue a duplicate evaluation.

Rerooting and capacity pruning do not recursively delete “subtrees.” Rerooting marks nodes reachable from the new
root, sweeps the rest, and rebuilds incoming counts and transposition buckets. Capacity pruning detaches all incoming
links to the least-visited materialized leaf before reclaiming it. Statistics discount scales each live node and edge
once, preserving means, then applies graph-aware pruning. All retention operations require an idle graph.

## Configuration

Existing configurations resolve to the tree control without changes:

```yaml
search:
  algorithm:
    kind: tree
```

The graph path is selected separately:

```yaml
search:
  algorithm:
    kind: graph
    transposition_value_threshold: 0.01
```

The same discriminated algorithm is available for elapsed evaluation definitions. Interactive chess analysis uses
the corresponding typed runtime parameter and also defaults to tree search.

## Instrumentation and prepared benchmarks

Each retained root exposes cumulative raw graph counters: table probes/hits, verified identity comparisons,
transposition links, unique nodes and edges, avoided evaluations, corrections and clips, continued transpositions,
cycle cutoffs, retained/reclaimed/pruned objects, peak live nodes/edges, and identity/reroot/pruning CPU nanoseconds.
It also records every traversal into a multi-parent node, the shared child's completed visits, the incoming edge's
local completed visits, and their positive difference. That difference measures how much more accumulated evidence
was available at the shared node than through the selected parent edge; it is information exposure, not additional
simulations completed.

The repetition benchmark records per-measurement deltas beside existing inference, throughput, CPU, RAM, and GPU
telemetry. After each completed ply it also unfolds the live acyclic graph structurally: every distinct root-to-node
path becomes one hypothetical tree instance. The snapshot reports canonical versus unfolded nodes, edges, and
expanded nodes, shared-node count, maximum path multiplicity, and saturation. Thus an early two-parent merge counts
the shared node and every materialized descendant twice in the unfolded control. Snapshot traversal time is reported
separately and excluded from search throughput because it is diagnostic work, not production search.

This structural count is an exact property of the materialized DAG, but it is not a claim that graph node visits can
simply be multiplied by path count. Node visits aggregate trajectories from all parents, while each incoming edge
retains local visits and values. A counterfactual tree could allocate its visits differently after the paths split;
there is no exact tree-equivalent visit total without retaining every trajectory or actually running the tree
control. Use unfolded node instances to quantify topology/evaluation duplication and shared-visit advantage to
quantify reused statistical evidence.

The later fixed-budget matrix is frozen at `100`, `400`, `1_000`, `3_200`, `10_000`, `30_000`, and `100_000`
searches per move. For each budget, run matched tree and graph commands with the same model, openings, games,
parallel searches, batch size, device, warmup, and measurement steps. Example command shape:

```bash
python py/tools/benchmark_repetition_mcts.py \
  --model MODEL.jit.pt \
  --openings py/reference/chess-stockfish-8moves-v3-openings-50.tsv \
  --algorithm graph \
  --transposition-value-threshold 0.01 \
  --searches 100 \
  --minimum-measurement-seconds 120 \
  --games 16 \
  --parallel-searches 4 \
  --maximum-batch-size 256
```

Repeat with `--algorithm tree` and at every frozen budget. Strength evaluation uses the same immutable opening suite,
model, paired colors, and search budget through the existing Stockfish gauntlet, selecting `--algorithm tree` or
`--algorithm graph --transposition-value-threshold 0.01`. Record completed simulations and neural evaluations
separately because Czech et al. use
neural evaluations as the meaningful GPU-cost axis when correction and terminal backups execute on CPU.

These commands are preparation only. They must not be run on a GPU or production node until benchmarking is
explicitly authorized and isolated from live training.
