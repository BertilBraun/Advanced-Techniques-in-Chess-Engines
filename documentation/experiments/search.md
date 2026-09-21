# Search experiments

## Ledger

| Technique | Status | What the evidence establishes | Principal evidence |
| --- | --- | --- | --- |
| Staged fixed visit budgets | **Retained** | The final recipe increases the global cap from 300 to 800 visits. Search-depth evaluation shows large strength gains with visits, but does not isolate this exact schedule. | [Final config](../../py/configs/production/chess-final-config.yaml), [search evaluation](../benchmarks/chess-search-evaluation-rtx3060-20260826/README.md), [search findings](../analysis/chess-search-findings-20260827.md) |
| Learned per-position budgets | **Implemented and rejected** | The implementation met its mechanics gates, then production runs lost roughly 60–100 Elo relative to non-adaptive runs. | [negative result](../analysis/adaptive-search-budget-negative-result-20260901.md), [frozen-trunk gate](../benchmarks/adaptive-search-budget-probe-rtx4070super-20260827/README.md) |
| Learned early stopping | **Implemented and rejected** | A byte-identical fork skipped 14% of search in the strongest arm but improved generation time only about 3% and showed no detectable strength gain. | [final conclusion](../analysis/adaptive-search-conclusion-20260904.md) |
| Threshold stopping from final-visit proxies | **Audited and declined** | The R3 audit could estimate only heuristic fast-search opportunity; it lacked temporal traces and exposed target-quality risks. | [R3 audit](../benchmarks/adaptive-search-termination-r3-20260813/README.md) |
| Randomized fast/full search | **Superseded** | It generated policy targets only on full-search turns and cheap continuations elsewhere. It improved game completion economics, but sparse targets and the long-tail batching interaction complicated learning and throughput. The final recipe uses full search on every recorded move. | [historical design](../history/optimizations/mcts.md), [search findings](../analysis/chess-search-findings-20260827.md), [conversion investigation](../analysis/chess-conversion-investigation-20260826.md), [adaptive-budget replacement plan](../plan/adaptive-search-budget-20260827.md) |
| Forced fast continuation after a late ply | **Superseded** | Implemented to finish games cheaply, but the conversion audit found missing endgame policy rows and early cut-value contamination. The final recipe instead uses explicit cut handling. | [conversion investigation](../analysis/chess-conversion-investigation-20260826.md), [training-data comparison](../analysis/v8-training-data-comparison-20260826.md) |
| Tree reuse across moves | **Retained** | The final recipe retains 60% of root visits. Existing evidence supports mechanics and throughput use, not an isolated Elo gain. | [Final config](../../py/configs/production/chess-final-config.yaml), [R3 audit](../benchmarks/adaptive-search-termination-r3-20260813/README.md) |
| Parallel searches | **Retained** | Parallel requests fill inference batches and raise throughput in production-like mixed workloads. Uniform-map matches also show a search-quality penalty, so parallelism is a throughput/quality trade rather than free compute. | [search evaluation](../benchmarks/chess-search-evaluation-rtx3060-20260826/README.md), [sweep](../benchmarks/parallel-searches-sweep-rtx4070s-20260906/README.md), [large-batch rerun](../benchmarks/parallel-searches-rerun-batch1600-rtx4070s-20260906/README.md) |
| Reduced-parent first-play urgency | **Retained** | The final recipe uses a 0.2 reduction. It belongs to the final bundle, but the repository does not contain a clean isolated final-recipe Elo ablation. | [Final config](../../py/configs/production/chess-final-config.yaml), [research ledger](../../THINGS_TO_TRY.md) |
| Forced playouts with target pruning | **Retained** | Enabled with coefficient 1.5. The implementation follows the intended exploration/target separation, but its causal gain was not isolated in the final lineage. | [Final config](../../py/configs/production/chess-final-config.yaml), [research ledger](../../THINGS_TO_TRY.md) |
| Search-value discount | **Retained** | The final recipe uses 0.99 per ply. A direct search comparison found no statistically resolved benefit, so retention is a recipe choice rather than a proven gain. | [search evaluation](../benchmarks/chess-search-evaluation-rtx3060-20260826/README.md), [Final config](../../py/configs/production/chess-final-config.yaml) |
| Monte Carlo graph search / transposition sharing | **Implemented and rejected** | Exact-history graph search was 5.76–8.63% slower while avoiding only 0.0249–0.1769% of evaluations at 1k–10k searches; exact repetition semantics erase most apparent chess transpositions. | [archived decision §10](../plan/archive/chess-four-day-baseline-and-next-run-plan.md#10-transposition-graph-search-and-inference-cache-decision---rejected), [handoff](../plan/next-run-handoff-20260906.md) |
| Neural inference cache | **Audited and declined** | Measurement-only instrumentation found an ideal unbounded reuse ceiling near 3.5% in the production-like mix, before cache costs; the tracker itself added overhead. | [archived decision §10](../plan/archive/chess-four-day-baseline-and-next-run-plan.md#10-transposition-graph-search-and-inference-cache-decision---rejected) |
| Gumbel search, sequential halving, dynamic candidate count | **Proposed only** | These remained research candidates and were not production experiments. | [research ledger](../../THINGS_TO_TRY.md) |

## Fixed search depth and evaluation

Search adds substantial strength, but its benefit is regime-dependent. The broad chess study measured policy
fidelity, direct matches, node cost, parallel-search effects, and an oracle allocator on a frozen model. It found
that deeper search continued to improve match results, while parallelism traded some move quality for much better
batch fill in realistic self-play. Those are measured properties of the tested checkpoint and hardware, not a proof
that any particular training-time schedule is optimal ([benchmark](../benchmarks/chess-search-evaluation-rtx3060-20260826/README.md);
[synthesis](../analysis/chess-search-findings-20260827.md)).

The final configuration therefore keeps the simple, observable mechanism: a generation-indexed visit schedule. It
avoids making a learned controller part of the policy/value training loop and makes the amount of search attached to
each training target explicit.

## Why adaptive allocation was closed

The first learned allocator used deep searches to label how much a position benefited from more computation. Its
offline probe failed the mandatory gate, and later online evidence showed a strength regression rather than a gain.
The sophisticated labelling, curve fitting, validation, and publication machinery was removed; its architecture
document is explicitly marked superseded ([design record](../architecture/learned-search-budget.md)).

Learned stopping addressed a narrower question: stop after the root policy appears stable. The decisive experiment
forked three arms from the same trained checkpoint and frozen rebuilt replay. The mechanism saved search exactly as
intended, but most self-play already overlapped optimizer work, so it shortened only slack. The result was no
detectable strength difference and too little cadence gain to resolve economically. This is a measured negative
result, not merely a decision that implementation was too complex.

## Fast/full search and the batching tail

The historical KataGo-style design randomly chose expensive full-search turns and cheap fast turns. Only full turns
became primary policy samples. This completed more games and supplied terminal outcomes cheaply, but it also made
training density depend on the full-search probability. As fast searches completed, the remaining full searches
formed a long, underfilled inference tail. Parallel searches recovered throughput in that tail while reducing search
quality at equal visits. The project later removed the fast/full split; the final recipe searches every played move
to the scheduled fixed cap.

That history matters when reading old benchmarks: a result measured under a 25% full / 75% fast mix should not be
transferred directly to the final all-full workload.

## Graph search and exact chess identity

The graph branch was a real implementation, not a paper exercise: shared descendant nodes, parent-local edge
statistics, correction terms, cycle rejection, virtual loss, rerooting, pruning, and rule/history identity were all
implemented. Correct chess identity retains enough repetition history that most move-order coincidences cannot be
merged safely. The observed reuse was too small to pay for graph bookkeeping, even at large search counts. The
project consequently retained tree search and released the rejected implementation as historical evidence rather
than maintaining an alternate production mode.

## Remaining evidence gaps

- No isolated final-lineage ablation assigns Elo credit to reduced-parent FPU, forced playouts, or tree retention.
- The final all-full workload deserves its own parallel-search quality/throughput curve; old mixed-tail results are
  informative but not identical.
- The report should not describe Gumbel search, sequential halving, or dynamic candidate counts as attempted.
