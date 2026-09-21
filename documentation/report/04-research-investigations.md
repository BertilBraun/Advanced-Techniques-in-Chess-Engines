# 4. Research investigations

This chapter organizes the research by question rather than version number. Each section states what was learned and
whether it survives in the final recipe.

## Search budgets: fixed, mixed, and adaptive

Early recipes borrowed KataGo's distinction between expensive full searches and cheap fast searches. Fast searches
advanced games but were normally excluded as primary policy targets, because a low-visit target can be too close to
the network prior. This improved game throughput but complicated batching, target eligibility, endgame coverage, and
auxiliary targets. The [offline search evaluation](../benchmarks/chess-search-evaluation-rtx3060-20260826/README.md)
measured visit depth, parallelism, value correction, and target fidelity under controlled compute.

Two learned successors were implemented:

1. **Predicted per-position budgets.** A learned curve head and corrector assigned a budget before search. The system
   achieved roughly the predicted improvement on its KL fidelity proxy, yet ran 60–100 ladder Elo behind controls.
   The leading explanation is target quality: heavily cheapened searches can be self-referential even when their
   local KL metric looks efficient. This is an implemented negative result, documented in
   [the budget study](../analysis/adaptive-search-budget-negative-result-20260901.md).
2. **Learned early stopping.** A search-time rule stopped after observing the evolving tree. A forked, shared-state
   experiment found no detectable strength difference. Skipping 14% of search made generations only about 3% faster
   because training overlapped self-play. The mechanism worked; it did not move the critical path. See the
   [final adaptive-search conclusion](../analysis/adaptive-search-conclusion-20260904.md).

Both systems and their auxiliary inference contracts were removed. The final recipe uses a staged but position-fixed
visit budget. This is a stronger conclusion than “the code was inconvenient”: both approaches were retired on
strength and wall-clock evidence.

## Search parallelism, batching, and tree retention

Parallel simulations fill inference batches and can improve searches per second, but virtual loss changes which
branches are explored. Early sweeps appeared to show large Elo costs; a rerun in the regime where the batch cap
actually bound found the relevant divergence test passed. The paired evidence is
[the original sweep](../benchmarks/parallel-searches-sweep-rtx4070s-20260906/README.md) and its
[superseding batch-1600 rerun](../benchmarks/parallel-searches-rerun-batch1600-rtx4070s-20260906/README.md).

Tree retention across played moves remains enabled through `retained_root_visit_fraction`. It reuses relevant search
work without merging rule-distinct positions. Throughput tuning also established that admission order matters in a
mixed-budget population: cheap searches can occupy slots while expensive targets remain active, leaving the tail
underfilled.

## Monte Carlo graph search and inference caching

A separate graph implementation supported shared descendant nodes, parent-local edge statistics, transposition
correction, cycle rejection, graph-aware rerooting, and exact chess history identity. It was rejected. In
production-like measurements, exact reusable transpositions were too rare to repay an approximately 6–9% throughput
loss; chess repetition rights make many superficially identical boards distinct search states. Exact neural-input
caching had only a small ideal hit-rate ceiling before lookup, synchronization, storage, and eviction costs.

The production structure is therefore a tree. The completed audit and numbers live in the historical
[four-day baseline plan, section 10](../plan/archive/chess-four-day-baseline-and-next-run-plan.md#10-transposition-graph-search-and-inference-cache-decision---rejected).
This is an audited rejection, not evidence that graph search is universally ineffective.

## Network architecture and policy head

Attention trunks initially plateaued partly because generation-zero inference exports produced almost uniform policy
priors. A shape-based bootstrap calibration fixed that defect across architectures. Once the comparison was repaired,
a matched frozen-replay study showed that the actionable improvement came from the from-to policy head, not the
attention trunk: applying the head to the existing CNN recovered most of the quality gain at far lower serving cost.
The [attention viability study](../benchmarks/chess-attention-viability-rtx3060-20260827/README.md) documents its
shortened training horizon, single-seed limitation, and inherited dataset defects.

The final recipe retains convolutional trunks, global context pooling, and the from-to head. Full attention and
transformer trunk sharing were investigated but not promoted. The report does not convert supervised cross-entropy
gaps into final-run Elo without online match evidence.

## Progressive model sizing

Small early networks generate search much faster, while larger networks become worthwhile only after data and
strength justify their cost. Isolated measurements confirmed substantial early throughput differences across model
sizes, motivating a ladder rather than a fixed large model. Promotion based only on a predetermined generation can
waste compute or grow too late, so the final design starts candidates when searched-Elo improvement per hour falls
below a configured threshold and promotes only after loss catch-up.

Progressive sizing is retained. The evidence includes the
[throughput benchmark](../benchmarks/progressive-sizing-throughput-rtx4070super-20260823/README.md) and
[accepted architecture](../architecture/progressive-model-sizing.md). Candidate-growth incidents also exposed a
TensorRT refit hazard, discussed in Chapter 5.

## Replay size, reuse, and sample selection

Replay reuse affects optimizer cadence as well as statistical efficiency because many schedules were indexed by
generation. The project found that changing replay ratio could leave short-run wall-clock strength similar while
advancing every generation schedule at a different real-time pace. Later frozen-replay and online investigations
separated optimizer choice, reuse, and initialization; the
[v35–v42 audit](../analysis/v35-v42-regression-audit-20260913.md) is the main causal record.

The final recipe uses reuse 4 and grows capacity from 600,000 to 20 million positions. Sampling is 30% uniform with
the remainder weighted by capped policy surprise. Regret-like information is used to select restart states rather
than as a second opaque per-sample training weight. These choices follow the project's own measurements and the
KataGo-inspired evidence review in
[Reference recipes for a compute-poor run](../analysis/reference-recipes-for-a-compute-poor-run.md).

Reanalysis appeared in an earlier runtime, then was removed with its disk replay/sidecar ownership model. It was not
re-established as a controlled final-campaign experiment. It must be described as superseded infrastructure and an
unattempted modern redesign, not as a negative result.

## Openings, restart states, and difficult positions

Uniformly shallow random openings increase diversity but do not deliberately revisit positions where the current
system struggled. Restart-state sampling archives bounded, recent candidates with sufficient branch mass, a
non-terminal value range, and remaining game length. Half of final-recipe games begin from shallow random play and
half from restart states. This is the project's practical form of prioritizing difficult states.

The design is retained, but its standalone Elo contribution was not isolated in a terminal ablation. Claims should
therefore be limited to mechanism, configuration, and observed operation unless a dedicated comparison is added.

## Resignation and endgame conversion

Resignation saves search only when a calibrated threshold satisfies an upper confidence bound on false non-losses.
Continuation games audit the threshold and preserve examples in which the losing side must actually convert. This
became important after runs reached won positions but repeatedly hit the ply cap. The
[conversion investigation](../analysis/chess-conversion-investigation-20260826.md) traced the problem to early data
poisoning and endgame coverage rather than to one simple resignation-rate difference.

The final recipe retains calibrated resignation, a 20% continuation probability for triggered games, staged maximum
game lengths, and a search-root value for censored games. The
[cut-game value study](../benchmarks/cut-game-value-target-rtx4070super-20260825/README.md) supports the root-value
fallback while documenting its limits.

## Auxiliary targets

Many auxiliary heads were proposed over the project: next policy, remaining game length, legal moves, future search
value, irreversible progress, and search correction. Small fixed-batch studies established that the wiring trained
and motivated reduced weights, but there was no budget for multi-day ablation of every head. The adaptive-budget
head and search-correction paths were removed with their consumers.

The final recipe retains only next-policy supervision at weight 0.15 and remaining-game-length regression at weight
0.1. They are training-only. Their inclusion is a recipe choice supported by diagnostics and prior work, not a
claim that this project isolated their individual Elo contribution.

## Optimizer and learning-rate investigations

The optimizer story changed with the inference architecture. Earlier production used AdamW. Once QAT and pre-fold
serving became central, matched frozen-replay screens tested SGD with Nesterov momentum, fold timing, and pre/post-fold
schedules. The screens showed stable learning, favored the historical high pre-fold schedule, and found higher
post-fold rates improved short-horizon replay fitting within the tested range. See the
[SGD replay screen](../benchmarks/chess-sgd-replay-screen-rtx4070s-20260913/README.md),
[pre-fold factorial](../benchmarks/chess-sgd-prefold-factorial-rtx4070s-20260914/README.md), and
[post-fold sweep](../benchmarks/chess-sgd-postfold-lr-rtx4070s-20260914/README.md).

Those were stationary-data diagnostics, not Elo proofs. The final run is the online test of the assembled SGD recipe.
