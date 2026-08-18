# Monte-Carlo graph-search paper audit and corrected screening

## Decision

Do not merge the graph-search implementation into the production path. The audit found and corrected one material
omission: the first incoming edge linked to an evaluated shared node must stop and back up that node's value rather
than immediately searching deeper. This is how the authors' released CrazyAra implementation realizes evaluation
reuse on a newly detected transposition. Correcting it increased measured evaluation reuse, but the exact-state
graph still saved too little work to offset its CPU and memory-locality overhead.

Tree search remains the default and recommended production implementation. This branch is rejection evidence, not
an accepted training change.

## Paper and released-code audit

Primary references:

- Czech, Korus, and Kersting, [Improving AlphaZero Using Monte-Carlo Graph
  Search](https://www.ml.informatik.tu-darmstadt.de/papers/czech2021icaps_mcgs.pdf), ICAPS 2021, especially
  Algorithms 1 and 2 on pages 106-107.
- The authors' released CrazyAra
  [selection implementation](https://github.com/QueensGambit/CrazyAra/blob/master/engine/src/searchthread.cpp#L93-L108)
  and [transposition traversal](https://github.com/QueensGambit/CrazyAra/blob/master/engine/src/searchthread.cpp#L248-L272).
- The released [trajectory backup](https://github.com/QueensGambit/CrazyAra/blob/master/engine/src/node.h#L820-L844)
  and [correction algebra](https://github.com/QueensGambit/CrazyAra/blob/master/engine/src/node.cpp#L1307-L1339).

The implementation matches the paper on the substantive graph-search invariants:

- one canonical node and complete shared descendant graph per verified state, rather than a neural-evaluation cache;
- node value/visit statistics shared across all incoming trajectories and edge value/visit statistics local to each
  parent/action;
- ordinary edge-local PUCT selection;
- explicit mini-batch trajectories, with virtual reservations removed exactly once;
- absolute edge/node residual comparison against the configurable 0.01 threshold;
- correction-only backups that update the incoming edge and traversed ancestors without falsely revisiting the
  shared child;
- correction after every crossed transposition during reverse backup; and
- backup of only the selected trajectory, never every possible parent path.

The paper's printed equations (8)-(9) have a sign inconsistency: substituting `Q - target` into equation (9) moves a
simple moving average away from the target. The released code instead uses `target - Q`, and the implementation uses
the same invariant-preserving form. With no clipping, one correction sample makes the incoming edge mean exactly
equal to the shared-node target.

The missed first-link case was visible in the released code: `add_new_node_to_tree` returns an already evaluated
transposition as a CPU-side backup value instead of descending to another neural leaf. Before commit `97d726f1`, a
new incoming edge had zero completed visits, so the threshold comparison was skipped and selection continued through
the shared subtree. Commit `97d726f1` now backs up the shared mean on that first edge visit, counts the avoided
evaluation, and leaves the shared node's visit count unchanged. A deterministic native test locks this behavior.

## Deliberate differences from the paper

The paper hashes the position together with a step counter to obtain a DAG. Its released chess verification requires
the same ply and rejects a currently repeated position, but otherwise approximates history. The paper explicitly
assumes a Markov state and acknowledges that history planes can violate that assumption.

This implementation cannot copy that approximation because the task requires exact rule semantics. Chess identity
includes pieces, side to move, castling, en passant, the exact halfmove clock, and the multiset of all retained
repetition-relevant positions and occurrence counts. The history order itself is discarded because future threefold
adjudication depends on occurrences, not their ordering. A future irreversible move clears obsolete history. This
identity is stricter than the paper and is the primary reason ordinary move-order transpositions do not merge.

Cycle rejection, mark/sweep rerooting, graph-aware pruning, retained-statistic scaling, and batched virtual-loss
handling are necessary adaptations for this engine; Algorithms 1-2 do not specify them. Their deterministic tests
pass and no evidence points to these adaptations suppressing valid exact-state links.

## Corrected measurements

Both corrected arms used commit `97d726f1`, assigned RTX 4070 SUPER device 1, the same deterministic 12x112 random
model and first 16 Stockfish 8moves-v3 final FENs as the earlier paired controls, batch size 256, four parallel
searches, and at least 120 seconds. Structural observation time is excluded from throughput.

| Searches | Tree searches/s | Corrected graph searches/s | Slowdown | Evaluations avoided | Avoided share | Unfolded tree nodes saved | Structural share |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 19,765.6 | 18,059.1 | 8.63% | 529 | 0.0249% | 1,803 | 0.0814% |
| 10,000 | 18,752.3 | 17,200.3 | 8.28% | 3,887 | 0.1769% | 12,054 | 0.5007% |

`Avoided share` is `evaluations_avoided / (inference_evaluations + evaluations_avoided)`. `Structural share` is the
number of node instances that an unfolded tree would duplicate divided by unfolded tree nodes. The unfolded count
properly multiplies the shared node and every descendant by the number of root-to-node paths, so an early diamond
receives more credit than a shared leaf.

At 10,000 searches, the graph observed a maximum path multiplicity of 10 and 34,055 more completed visits on shared
children than on the traversed incoming edges. Those statistics confirm full shared-subtree behavior. They do not
constitute additional completed simulations: the paper's usable savings are neural evaluations and allocated node
instances, not a multiplication of Monte-Carlo samples.

The corrected results remain unsuitable: less than 0.18% inference avoidance and 0.51% structural reuse do not
offset an approximately 8.3% search-throughput loss. A trained compatible model could change paths and threshold
crossings, but it cannot make this measured exact-state hit topology competitive without substantially more
transpositions. Relaxing history identity would be a different, approximate algorithm and would violate the current
state-identity requirement.

## Validation and artifacts

- Local WSL CompileCheck build: `cmake --build ~/advanced-chess-tests --parallel 4`
- Unified local native suite: `ctest --test-dir ~/advanced-chess-tests --output-on-failure` -- 1/1 passed in 1.31 s
- Exact corrected Release extension built successfully on the RTX 4070 SUPER node
- `01`-`04`: paired controls and pre-fix graph measurements from commit `1ad12382`
- `05`-`06`: corrected first-link graph measurements from commit `97d726f1`

No strength match was run. No production service, production artifact, or GPU 0 process was modified.
