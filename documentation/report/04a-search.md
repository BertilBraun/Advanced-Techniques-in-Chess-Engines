# 4A. Search investigations

This chapter expands the search summary in [Chapter 4](04-research-investigations.md). Statuses follow the
[search experiment ledger](../experiments/search.md); evidence grades follow
[Chapter 2](02-methodology-and-evidence.md).

## Fixed depth and the value of search

The broad offline evaluation asked three different questions: how much playing strength deeper search adds to a
fixed network, how well a search policy approximates a deeper target at equal compute, and how implementation choices
affect throughput. Those answers must not be conflated. Visits dominated the measured playing-strength axes, while
several heuristics produced smaller or unresolved differences. The final recipe therefore uses an observable staged
cap—300, 400, 500, 600, then 800 visits—rather than a controller whose output is hard to audit.

This is **S/P** evidence that search depth matters for the tested checkpoint, plus **R** for the exact training-time
schedule. There is no one-variable experiment proving that these generation boundaries are optimal. The principal
records are the [search evaluation](../benchmarks/chess-search-evaluation-rtx3060-20260826/README.md) and
[search synthesis](../analysis/chess-search-findings-20260827.md).

## Fast/full search and target density

The historical KataGo-style design randomly selected full-search moves and used cheap searches to advance the rest
of each game. Only full-search turns became primary policy targets. This could finish more games for a fixed search
budget, but it coupled four quantities:

- fraction of plies producing policy rows;
- quality of next-policy auxiliary targets;
- game and terminal-outcome throughput;
- inference-batch occupancy as cheap searches finished before expensive ones.

The long tail of remaining full searches underfilled batches. Parallel simulations recovered some device fill but
changed the search itself. Forced fast search after a late ply created a more serious failure: it removed endgame
policy rows and allowed a shallow cutoff value to contaminate targets for the whole game. The final recipe searches
every played move to the staged cap and handles cut games explicitly. Old 25%-full/75%-fast throughput results are
not direct measurements of the final all-full workload.

## Threshold stopping: opportunity audit, not negative Elo result

The first stopping study reconstructed possible savings from final visit distributions. It did not have temporal
traces proving that a live rule could have stopped at the inferred earlier point. It also showed why a “fast” search
was not target-free: its policy could supervise the preceding row's next-policy head, and full-search output affected
restart-state selection. The correct classification is **audited and declined, P/R**, not implemented and defeated
in a match. See the [R3 termination audit](../benchmarks/adaptive-search-termination-r3-20260813/README.md).

## Predicted budgets

The later allocator was a complete production implementation. A network head predicted the policy-divergence curve
over budget multiples; a corrector used search-derived features; a dual variable held average spend near one; and a
gate reverted to flat search until validation conditions passed. Deep-search labels, durable replay write-back,
TorchScript publication, calibration, and telemetry all worked.

Its proxy result was real: the live allocator captured useful KL headroom and allocated more compute to contested
positions. Its learning result was negative: adaptive runs trailed non-adaptive lineages by roughly 60–100 ladder
Elo. The leading mechanism is a mismatch between per-position fidelity and training value. A target can be close to
a deep policy in KL yet remain too self-referential to teach the next network. Because the proxy succeeded and Elo
failed, this is one of the project's clearest demonstrations that **P does not imply O/S**
([negative result](../analysis/adaptive-search-budget-negative-result-20260901.md)).

## Learned early stopping

Learned stopping moved the decision inside search so it could observe the tree. The decisive comparison copied a
trained checkpoint and rebuilt replay, then forked byte-identical arms. The strongest stopping arm skipped 14% of
nominal search but reduced generation time by only about 3%, because self-play overlapped training and the removed
work was mostly slack. Paired strength differences were unresolved and economically too small to measure in the
available horizon. This is **O/T** negative evidence from a strong shared-state control, not proof that stopping is
universally useless ([conclusion](../analysis/adaptive-search-conclusion-20260904.md)).

## Parallelism and batch fill

Multiple simulations per root increase schedulable inference and aggregate searches per second. Virtual loss and
stale sibling decisions mean they are not mathematically identical to serial search. The first final-era sweep was
later qualified because its batch cap did not bind in the intended regime. The batch-1600 rerun passed its divergence
acceptance test under the corrected load. Both records remain useful: one shows the danger of transferring a result
across serving regimes; the other supports the actual high-fill conclusion
([sweep](../benchmarks/parallel-searches-sweep-rtx4070s-20260906/README.md),
[rerun](../benchmarks/parallel-searches-rerun-batch1600-rtx4070s-20260906/README.md)).

## Tree reuse, FPU, forced playouts, and discount

The final bundle retains 60% of root visits across moves, reduced-parent-value FPU with reduction 0.2, forced
playouts with coefficient 1.5, and a 0.99 per-ply search-value discount. Their evidence is not equal:

- tree retention has implementation and throughput rationale but no isolated final-lineage Elo arm;
- forced playouts separate exploration from the pruned training target, following external evidence;
- reduced-parent FPU is configured and widely motivated but not isolated here;
- the search study did not statistically resolve a standalone value-discount benefit.

They are retained recipe choices (**M/R**, with limited **S/P**), not four measured Elo multipliers.

## Graph search and inference caching

The graph branch implemented shared descendant nodes, parent-local edges, correction terms, cycle rejection,
virtual loss, graph-aware rerooting/pruning, and exact chess history identity. Exact repetition semantics eliminated
most apparent board transpositions. At 1,000–10,000 searches, avoidable evaluations were tiny while graph bookkeeping
reduced throughput by about 6–9%. Higher-search tests found more hits but still lost throughput. The implementation
was rejected for this workload (**T/M**), not merely left unfinished.

Inference caching was audited separately without merging search state. Its ideal unbounded reuse ceiling was only
about 3.5% in the production-like mixed workload before synchronization, storage, eviction, and finite-capacity
misses. It was audited and declined. Both decisions are preserved in the
[archived graph/cache record](../plan/archive/chess-four-day-baseline-and-next-run-plan.md#10-transposition-graph-search-and-inference-cache-decision---rejected).

## Not attempted

Gumbel root search, sequential halving, dynamic candidate counts, and another generation of adaptive allocation
remained proposals. They may be discussed as future alternatives but not listed as project experiments.
