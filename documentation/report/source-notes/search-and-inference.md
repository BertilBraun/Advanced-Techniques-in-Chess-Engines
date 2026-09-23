# Search and inference investigation dossier

This is a source dossier for the eventual technical report, not report prose. It is organized by research question
rather than by run, branch, or implementation date. Each section records what was asked, what was actually tried,
how the mechanism worked, what the evidence supports, why the approach was retained or rejected, the traps that
affected interpretation, and what remains unknown.

The dossier deliberately separates four axes that are easy to collapse into one headline number:

- search quality at a fixed network and compute budget;
- quality of the policy or value target written to replay;
- inference or search throughput under a defined workload;
- end-to-end learning progress per unit wall time.

A favorable result on one axis is not evidence of a favorable result on another. In particular, a faster actor is
not automatically a faster learner, and a search target that is closer to a deep-search target is not automatically
a more useful learning target.

## Fixed search budgets

### Question

How much does ordinary fixed-budget search improve play and target quality, and is an auditable global budget a
reasonable default?

### Approaches

- Fixed visit sweeps against Stockfish measured strength with one frozen network.
- Deep-search comparisons measured how closely shallower policies approximated a much deeper reference policy.
- A staged fixed schedule increased the global visit cap as the network and training system matured.
- Search constants such as the exploration coefficient, value discount, first-play urgency, and parallel leaf count
  were screened separately from the visit budget.

### Mechanism

The native engine runs policy-guided Monte Carlo tree search. Each neural evaluation supplies legal-move priors and
a WDL value. Selection uses PUCT, expansion is batched across active roots, and values are backed up with alternating
perspective and an optional per-ply discount. A fixed visit limit is directly observable and does not depend on a
learned controller, calibration state, or a position-difficulty proxy.

### Evidence and result

Visits were the dominant measured search-strength variable. In the principal frozen-network sweep, scores against
the fixed opponent rose from 0.318 at 200 visits to 0.537, 0.580, 0.662, and 0.748 at 400, 600, 1,000, and 1,600
visits. The fitted middle-to-deep trend was about 81 Elo per doubling, with uncertainty too large to interpret every
adjacent step as a precise effect. At 600 visits, the target selected the same best move as its own 10,000-visit
reference only 72.4% of the time. Target fidelity was still improving at the deepest measured reference.

The evidence supports the broad decision to spend more search as resources permit. It does not prove that any exact
training-time schedule is optimal. The fixed schedule should therefore be described as a transparent engineering
choice supported by a strong depth trend, not as a separately optimized algorithm.

### Rationale

Fixed budgets survived because they were predictable, measurable, and strong. They also removed several failure
surfaces introduced by adaptive controllers: controller calibration, distribution shift, cheap low-information
targets, delayed policy publication, and uncertainty over what compute savings reach the critical path.

### Pitfalls

- A historical evaluation path silently forced zero first-play urgency while self-play used reduced-parent FPU. The
  measured penalty was about 76 Elo, so older ladder measurements were systematically pessimistic.
- A visit limit is not necessarily new work. A retained tree can begin a move with existing visits.
- Offline target fidelity and fixed-network playing strength say nothing directly about learning outcomes.
- Individual search-depth arms had broad uncertainty. The monotone multi-budget trend is more reliable than an
  isolated adjacent contrast.

### Unknowns

- The exact contribution of each fixed schedule boundary has not been isolated.
- Target fidelity under the complete retained-tree, parallel self-play workload was not measured across every
  budget.
- The eventual report still needs an end-to-end account of how visit depth trades against games, replay admissions,
  and optimizer progress in the final workload.

### Sources

- [Chess search findings](../../analysis/chess-search-findings-20260827.md)
- [Full search evaluation record](../../benchmarks/chess-search-evaluation-rtx3060-20260826/README.md)
- [Current native search description](../../system/search-and-self-play.md)
- [Current search implementation](../../../cpp/src/search/SearchEngine.hpp)
- [Search executor and batched scheduling](../../../cpp/src/search/SearchExecutor.hpp)

## Randomized fast and full searches

### Question

Can most moves use a cheap search to complete more games while a random minority receives an expensive search and
becomes a high-quality policy target?

### Approaches

- A KataGo-inspired playout-cap-randomization design selected a minority of moves for a full search and used a cheap
  search to advance the remainder of the game.
- Only full-search moves became primary policy rows. Cheap searches could still influence move choice, terminal
  outcomes, next-policy auxiliaries, and subsequent retained trees.
- Different full and cheap budgets were used while the same basic random-selection scheme remained in place.
- Parallel leaf searches were increased to refill the inference batch after cheap searches finished.

### Mechanism

The intended trade is between independent completed-game outcomes and policy-target density. A game can be advanced
with many cheap moves, reserving expensive searches for a random subset of positions. In KataGo's setting, long 19x19
Go games provide relatively few independent terminal outcomes for a large amount of per-game search. Completing
more games can therefore address value-label scarcity even though most positions do not yield policy targets.

That transfer was poorly matched to chess. Chess games are much shorter than 19x19 Go games and positions are often
more decisive. The value objective was already learning without evidence that terminal outcomes were the limiting
resource. Discarding cheap-search positions as primary training rows therefore reduced policy-target density without
a demonstrated compensating value-learning benefit.

The phrase “25% full search” describes row density, not compute waste. Compute share depends on both budgets. With a
25% full probability, 600 full visits, and 150 cheap visits, cheap moves consume 42.9% of the nominal search work:
`0.75 * 150 / (0.25 * 600 + 0.75 * 150)`. With 600 and 100 visits, the cheap share is 33.3%. Those cheap moves are
not literally useless: they advance games and affect outcomes. The problem is that they consume substantial search
while producing no direct primary policy row, and the hypothesized value-target payoff was not the bottleneck here.

### Evidence and result

The design had a serious systems interaction. When cheap searches retired, only the full-search minority remained.
In a 512-game workload with a 25% full / 75% cheap mix, the long tail contained about 128 active trees. With one
parallel leaf per tree, average inference batches fell to 86 against a cap of 320. Four parallel leaves raised the
average to about 268 and improved aggregate search throughput by roughly 20%, but the full searches then paid the
quality cost of more stale, virtually reserved traversals.

The approach also caused target-semantic hazards:

- a cheap search could supply the next-policy auxiliary target for the previous retained row;
- full-search outputs influenced restart-state selection;
- forcing cheap searches after a late ply deleted endgame policy targets;
- using a shallow terminal-cut search could propagate a weak soft value target through an entire capped game;
- full-search forced-playout pruning meant stored targets were not equivalent to raw intermediate visit vectors.

The design was superseded by searching every recorded move to the selected fixed cap. It was not rejected by a
clean, standard-chess Elo ablation that changed only playout-cap randomization. The justification combines workload
economics, observed batch-tail behavior, target density, the absence of a demonstrated value-label bottleneck, and
later negative evidence from feeding very cheap adaptive-search targets into training.

### Rationale

The project retained KataGo's underlying lesson—search compute and training-target eligibility are separate design
choices—but did not retain its fast/full recipe. For this chess workload, completing more games was less valuable
than preserving a dense stream of sufficiently searched policy rows. Search depth could then be increased globally
or allocated only if the allocation preserved learning value.

### Pitfalls

- “Four times more games” is a nominal intuition, not a measured end-to-end gain when full and cheap searches have
  nonzero unequal costs.
- “Fast positions are discarded” applies to primary replay admission, not to all downstream effects.
- The batching tail made a policy intended to save compute require extra leaf parallelism, changing search quality.
- A result measured with uniform full searches does not predict the mixed-workload tail.
- The causal chess-versus-Go interpretation is well motivated but is not a controlled cross-game experiment.

### Unknowns

- There is no matched long training run comparing all-full search against randomized fast/full search at equal
  wall-clock compute.
- The exact value-learning benefit of additional completed chess games was not isolated.
- A redesigned scheme that retains carefully weighted cheap-search rows was not tested as a direct successor.

### Sources

- [Historical MCTS optimization description](../../history/optimizations/mcts.md)
- [Search findings: target density and batching tail](../../analysis/chess-search-findings-20260827.md)
- [CPU and batch-fill study](../../benchmarks/self-play-search-cpu-i7-11370h-20260824/README.md)
- [Adaptive termination audit: target interactions](../../benchmarks/adaptive-search-termination-r3-20260813/README.md)
- [Cut-game target study](../../benchmarks/cut-game-value-target-rtx4070super-20260825/README.md)
- [KataGo, “Accelerating Self-Play Learning in Go”](https://arxiv.org/abs/1902.10565)

## Threshold-based early termination

### Question

Can a search stop once its leading action appears uncatchable or sufficiently concentrated, without changing the
stored target or played move materially?

### Approaches

- Completed-game records were audited for final visit concentration, top-two margins, root values, child values, and
  apparent headroom below the nominal cap.
- Hypothetical uncatchable-leader rules with minimum search fractions were reconstructed from final distributions.
- A wider offline study compared threshold rules with deeper search policies and mapped apparent confidence to
  marginal benefit.

### Mechanism

A threshold rule examines the current visit distribution and stops if the leading action appears secure. Such a
rule is only safe if it preserves more than the eventual argmax: the partial policy target, forced-playout pruning,
temperature-based move distribution, value estimate, resignation decision, restart eligibility, and auxiliary
targets can all change before the nominal limit.

### Evidence and result

The completed-game audit found a real opportunity signal but could not establish an implementable saving. A
deliberately optimistic reconstruction suggested that a 75%-minimum rule might remove 15.3% of nominal cheap-search
visits, or 6.38% of all nominal limits when full searches were assigned no savings. The records did not contain
intermediate visit traces, starting retained visits, actual completed simulations, or the full raw child vector for
forced-playout searches. The result was therefore an opportunity estimate, not an observed saving.

The deeper offline study explained why a monotone concentration threshold was a poor allocator. Marginal value was
inverted-U shaped: very diffuse positions and already-decided positions gained little, while moderately concentrated
but contested positions gained most. The threshold stopped precisely in a region where additional search was more
valuable than random allocation. Every tested rule had the wrong sign after accounting for compute.

This approach was audited and declined. It was not implemented and then defeated in an Elo match.

### Rationale

The audit prevented a plausible but underidentified heuristic from entering production. A rule that knows only the
final distribution cannot show when that distribution became stable, and a final winner does not establish that an
earlier distribution would have been a safe learning target.

### Pitfalls

- Final-state data cannot identify the first safe stopping time.
- Nominal visits are not actual added visits when roots retain statistics.
- Leader stability does not imply policy-distribution stability.
- Decisive root values were not more concentrated; value magnitude was an unsafe confidence proxy.
- Savings in simulations do not convert directly to wall time when inference and training overlap.

### Unknowns

- A shadow trace with periodic raw-tree snapshots was specified but not retained as the final approach.
- A rule designed around target divergence rather than top-one stability could behave differently.

### Sources

- [Adaptive termination opportunity audit](../../benchmarks/adaptive-search-termination-r3-20260813/README.md)
- [Search findings: non-monotone marginal value](../../analysis/chess-search-findings-20260827.md)

## Predicted per-position search allocation

### Question

Can the network predict how much search a position deserves before search begins, hold mean compute constant, and
obtain deeper-search target fidelity where it matters?

### Approaches

- Scalar difficulty and threshold-style allocation were explored and found too weak.
- The implemented allocator predicted the whole curve of policy divergence over eight budget multiples.
- Deep searches supplied checkpointed policy curves; a network auxiliary head predicted log-KL to the deep policy.
- A small scripted corrector combined those predictions with root observables.
- A Lagrangian selector chose the budget, while a dual variable held mean spend near one.
- Safety gates fell back to uniform search until warm-up, calibration gain, and spend conditions passed.

### Mechanism

For each position the model predicted `log(KL(deep policy || policy at candidate budget))`. The corrector consumed
the predicted curve plus top-visit share, policy entropy, ply, baseline visits, and source age. After enforcing a
monotone curve, native code selected the budget minimizing predicted divergence plus a compute-price term. Deep
label generation, replay persistence, corrector training, scripted publication, native consumption, dual-state
persistence, and telemetry were all implemented.

### Evidence and result

The controller succeeded on its own proxy. At approximately matched mean spend it captured about 23% of the
available policy-KL headroom, consistent with the offline expectation, and behaved qualitatively sensibly: cheap
budgets went to later or apparently decided positions, while deeper budgets went to contested positions. On the
measured mean curve, its target fidelity resembled about 1.18 times uniform search at 0.967 times spend.

It nevertheless learned worse. Repeated production attempts trailed comparable non-adaptive training by roughly
60–100 ladder Elo. The leading explanation is a proxy mismatch, not a mechanical failure. The controller optimized
closeness to a deep policy at each current position. It could not measure how a cheap target changes the next
network. More than a third of positions were cut to a mean fraction near 0.36, and nearly 9% received only one
eighth of baseline search. Those shallow targets were close to the network's own prior and were still trained at
full weight, making them potentially self-referential.

This mechanism is a hypothesis consistent with the evidence, not a proven mediation analysis. What is proven is
the stronger and more interesting result: the proxy and controller worked, while the learning outcome was negative.

### Rationale

The allocator was rejected because the desired outcome is learning progress, not policy-KL capture. A controller
that correctly optimizes the wrong objective is not rescued by better engineering of the same objective.

### Pitfalls

- A controller can meet its calibration and compute-spend targets while degrading learning.
- Prediction before search cannot react to information revealed inside the tree.
- Curve labels derived from the same search tree can contain structural bias.
- Calibration and dual state introduced multiple production failure modes: duplicated pins, reversed isotonic
  projection, badly seeded dual variables, and feature standardization explosions at schedule boundaries.
- Independent training runs were noisy enough to obscure modest effects; shared-state forks were needed later.

### Unknowns

- Down-weighting or excluding very cheap policy rows was proposed but not tested as a clean causal follow-up.
- A controller trained directly on downstream learning value was not developed.
- Dynamic allocation that observes the evolving tree remains a different research problem.

### Sources

- [Predicted-budget negative result](../../analysis/adaptive-search-budget-negative-result-20260901.md)
- [Predicted-curve design record](../../plan/search-budget-curve-20260830.md)
- [Frozen-trunk allocator probe](../../benchmarks/adaptive-search-budget-probe-rtx4070super-20260827/README.md)
- [Learned-search-budget architecture](../../architecture/learned-search-budget.md)

## Learned stopping inside search

### Question

Can a learned controller observe the evolving search tree, stop positions dynamically, preserve strength, and turn
the saved search into faster learning?

### Approaches

- The stopping system observed tree-derived features unavailable to a pre-search predictor.
- Deep-search labels, calibration, a learned policy, native stopping, scripted publication, and extensive telemetry
  were implemented.
- The decisive comparison forked one trained checkpoint and rebuilt replay, then copied the same frozen model,
  optimizer, and replay state into baseline and stopping arms.
- Two stopping ceilings tested different compute-saving aggressiveness.

### Mechanism

The controller could stop after seeing how the tree evolved rather than committing to a budget before search. This
addressed the central conceptual weakness of predicted allocation. The training runtime, however, overlapped half
of self-play with each optimizer quantum. Search removed during the overlapped period was slack rather than
critical-path time.

### Evidence and result

The most aggressive arm skipped about 14% of nominal search, and credit-wait time ordered exactly as expected. The
generation cadence improved by only about 3%. Paired strength differences from the shared state were approximately
+1.7 Elo with standard error 9.9 and -4.2 Elo with standard error 10.1, depending on the ceiling. No strength effect
was detected.

At the measured learning rate, the cadence improvement was worth about one Elo over three hours, far below the
experiment's resolution. The system was rejected because its measurable end-to-end value was economically tiny,
not because learned stopping was proven universally incapable of working.

### Rationale

The experiment correctly asked whether saved simulations reached the training critical path. They mostly did not.
This is a systems-level negative result: a functioning algorithm can be irrelevant when it optimizes work hidden
under another stage.

### Pitfalls

- Early measurements in an uncalibrated regime mischaracterized the controller's steady behavior.
- Independent runs confounded initialization, replay, and early learning; only the shared-state fork resolved the
  comparison.
- “Search saved” and “training faster” were separated by overlap and scheduler slack.
- The subsystem was large and operationally costly, with several production-stopping defects.

### Unknowns

- A non-overlapped or inference-bound training topology could value the same 14% search saving differently.
- A much stronger stopping signal might save enough work to move the critical path, but this was not demonstrated.

### Sources

- [Final learned-stopping conclusion](../../analysis/adaptive-search-conclusion-20260904.md)
- [Adaptive stopping design](../../plan/adaptive-stopping-plan-20260901.md)
- [Preserved off-main stopping implementation](https://github.com/BertilBraun/Advanced-Techniques-in-Chess-Engines/tree/adaptive-stopping-final/py/src/search_stopping)

## Native MCTS mechanics and target semantics

### Question

Which search mechanics are part of the retained system, what do they do, and which have isolated evidence rather
than only implementation or external rationale?

### Approaches and mechanisms

#### PUCT and neural evaluation

Selection combines an edge's backed-up value with an exploration term derived from the network prior and parent
visits. Legal policy normalization happens at the native inference boundary. WDL output is converted into the value
used by search. The project moved rules, tree ownership, selection, expansion, backup, and batching into C++ rather
than treating Python as an inner-loop search server.

#### First-play urgency

Unvisited actions need an initial value. The retained reduced-parent formulation initializes them below the current
parent estimate by 0.2. Alternatives included zero and a larger reduction. The frozen-network comparisons did not
resolve a clean difference between reduced-parent settings, but the mismatch between zero-FPU evaluation and
reduced-FPU self-play produced a large apparatus error. The setting is retained as part of the coherent recipe, not
claimed as an isolated final-run Elo gain.

#### Virtual loss and parallel leaves

In-flight traversals reserve their selected path so concurrent descents do not choose the same leaf. Completion or
cancellation removes each reservation exactly once before applying the real backup. This enables multiple leaves
per root to join a batch but makes decisions stale relative to serial MCTS. Virtual-loss weights from 0.25 to 1.0
did not materially separate in the tested binding regime; the retained value is 1.0.

#### Forced root playouts and target pruning

Forced playouts ensure low-prior root actions receive exploration. The visits added purely by the force rule are not
blindly written into the policy target: pruning removes unsupported forced visits according to ordinary PUCT. This
separates exploration from supervision. The mechanism is implemented and externally motivated, but the repository
does not contain a final-workload isolated Elo ablation for its coefficient.

#### Dirichlet root noise and temperature

Root priors are mixed with Dirichlet noise during self-play. Moves are sampled from visits under a decreasing
temperature until a greedy cutoff. Search certainty and played-move randomness are therefore not the same quantity;
an early-stop rule must preserve the intended sampling distribution, not merely the eventual argmax.

#### Value discount

Backups apply a 0.99 per-ply discount. Frozen-network comparisons of no discount and a stronger discount had
intervals spanning zero. It remains a recipe choice without an isolated causal strength claim.

#### Tree retention

After a played move, the selected child becomes the new root. Stored statistics are discounted to 60%, and the next
request specifies additional visits rather than pretending the retained root is fresh. Retention saves work and
preserves deeper analysis, but it complicates all nominal-budget accounting. A model refresh or incompatible search
change resets the tree.

#### Batched leaf scheduling

The executor advances many game roots and can request several leaves per root. Inference runs asynchronously through
preallocated slots. One selection pass gives a tree one in-flight descent before returning to the pool, so effective
per-root concurrency is limited by both the configured parallel count and the ratio of inference capacity to active
trees.

### Evidence and result

The retained implementation is a coherent native tree-search system. Only some components have isolated local
measurements. Search depth has strong evidence. First-play urgency has compelling evidence that evaluation and
self-play must agree. Parallel leaves have workload-specific throughput and weaker quality evidence. Tree retention,
forced playouts, discount, noise, and the exact PUCT setting should not each be advertised as measured Elo gains.

### Rationale

The report should distinguish “retained because it is an implemented part of the selected AlphaZero recipe” from
“retained because a controlled experiment measured an isolated improvement.” This avoids turning a final
configuration into a fictional additive ablation table.

### Pitfalls

- Stored forced-playout-pruned targets are not raw root visit vectors.
- Parallelism can be configured but inert if active trees already exceed inference capacity.
- Root retention makes a visit cap different from newly executed simulations.
- An evaluation search with different FPU, noise, forced-playout, or batching semantics is not the same algorithm.

### Unknowns

- No complete factorial ablation exists for the retained search bundle.
- The interaction between retention, parallel leaves, and target fidelity at the final serving shape remains only
  partially measured.

### Sources

- [Native search system](../../system/search-and-self-play.md)
- [Search tree](../../../cpp/src/search/SearchTree.hpp)
- [Search engine](../../../cpp/src/search/SearchEngine.hpp)
- [Search executor](../../../cpp/src/search/SearchExecutor.hpp)
- [Forced-playout pruning](../../../cpp/src/search/ForcedPlayouts.hpp)
- [Self-play worker and rerooting](../../../py/src/self_play/worker.py)
- [Search findings](../../analysis/chess-search-findings-20260827.md)

## Parallel search, batching, and tree retention

### Question

How should concurrency be arranged so the GPU remains full without silently changing search quality or individual
game latency?

### Approaches

- The system varied concurrent games, operating-system processes, inference workers, inference batch caps,
  outstanding batches, batching timeouts, and parallel leaves per root.
- Uniform-search microbenchmarks were compared with the real mixed-search workload.
- A large-batch rerun forced per-root parallelism to bind after an earlier sweep left it mostly inert.
- CPU profiling decomposed tree selection, result processing, backup, encoding, allocation, and inference wait.
- Trees retained statistics across played moves to avoid discarding prior work.

### Mechanism

There are three different concurrency layers:

1. many games supply independent roots;
2. several leaves may be in flight within one root, protected by virtual reservations;
3. the inference runtime combines leaves into device batches and allows bounded outstanding submissions.

More roots produce independent batching with minimal search distortion but increase game latency and host memory.
More leaves per root fill batches when roots are scarce, but make MCTS less serial and can weaken search. More
processes increase model/runtime duplication and host contention. These are different levers and must not be
reported as a single “batch size.”

### Evidence and result

In the mixed fast/full workload, four parallel leaves raised search throughput by about 20% at 512 active games
because they refilled the full-search tail. In a uniform workload the gain was only about 3.6%. Where parallelism
actually bound, the fixed-network point estimates worsened monotonically: roughly -20, -36, and -45 Elo for two,
four, and eight leaves relative to one, though the individual comparisons were not statistically resolved.

An initial later sweep suggested almost no cost per doubling, but its batch capacity was too small relative to tree
count for the knob to bind. A deliberately oversized-batch rerun made it active and estimated about -6.4 ± 4.7 Elo
per doubling, still unresolved but materially steeper. That rerun was diagnostic, not a production throughput
configuration.

CPU profiling found tree selection, result processing, and backup to be first-order costs. An optimization reduced
native search CPU per simulation by roughly 21–28% in the mixed workload without changing mean root visits or batch
fill. This established headroom, not guaranteed actor throughput, because the binding resource depends on the node.

### Rationale

The retained topology balances device fill against stale-tree quality, memory, CPU saturation, and latency. It is a
workload-specific systems choice. The project correctly treats realistic actor throughput and isolated search
quality as two sides of the decision.

### Pitfalls

- A non-binding parameter produces a convincing but meaningless null result.
- Concurrent matrix arms can inherit draining-thread-pool artifacts; per-arm wall clock is not throughput.
- Uniform full-search benchmarks miss the mixed-workload tail.
- Search throughput does not equal completed-game throughput or admitted replay throughput.
- More games can improve batching while making each game much older by the time it completes.

### Unknowns

- The precise quality cost of the final low parallel count at each fixed budget is not resolved.
- The final all-full workload removes the original fast/full tail, so older topology optima should not be treated as
  timeless.

### Sources

- [Search findings: binding behavior and quality/throughput trade](../../analysis/chess-search-findings-20260827.md)
- [Large-batch parallelism control](../../benchmarks/parallel-searches-rerun-batch1600-rtx4070s-20260906/README.md)
- [Initial parallel-search sweep](../../benchmarks/parallel-searches-sweep-rtx4070s-20260906/README.md)
- [Native CPU profile](../../benchmarks/self-play-search-cpu-i7-11370h-20260824/README.md)
- [Submission optimization](../../benchmarks/self-play-submission-8xrtx4070super-20260824/README.md)
- [Multiworker graph replay](../../benchmarks/self-play-graph-multiworker-8xrtx4070super-20260824/README.md)

## Monte Carlo graph search and transpositions

### Question

Can identical chess states reached by different move orders share neural evaluations, descendants, and accumulated
statistics strongly enough to outperform ordinary tree search?

### Approaches

- A complete, optional Monte Carlo graph-search implementation was built rather than approximated by a neural cache.
- Canonical shared nodes stored state-level values, visits, priors, and descendants.
- Incoming parent/action edges retained local value and visit statistics for ordinary PUCT selection.
- Corrections reconciled an incoming edge with a more informed shared-node value.
- Exact game-semantic hashing and collision-checking equality determined whether nodes could merge.
- Cycle rejection, full trajectory reservations, graph-aware rerooting, mark/sweep pruning, capacity reclamation, and
  retained-statistic discounting were implemented.
- The implementation was audited against both the paper and the released CrazyAra code, revealing and correcting a
  missing first-link behavior.
- Tree and graph controls were measured from low budgets through tens of thousands of searches.

### Mechanism

A graph node may have several incoming edges. Node statistics and its descendant graph are shared, while each
parent's edge statistics remain local. Selection records the exact traversed path so virtual loss and backup touch
only that trajectory. When a new incoming edge reaches an already evaluated shared node, the corrected
implementation backs up the shared mean rather than immediately descending to another neural leaf. On later
traversals, a sufficiently different local edge estimate can receive a correction toward the shared value without a
new neural evaluation.

Exact chess identity is stricter than board equality. It includes pieces, side to move, castling rights, en-passant
state, the halfmove clock, and repetition-relevant history counts. Those fields affect the future legal result, so
merging board-identical but history-distinct nodes would change the game. This exactness removes most apparent
move-order transpositions.

### Evidence and result

At ordinary budgets, equality-verified links were absent or negligible and no neural evaluations were avoided.
Graph bookkeeping reduced throughput by roughly 2.3–8.8%. At 30,000 and 60,000 searches, verified table hits rose
to 2.37% and 3.46%, but avoided evaluations remained approximately 0.0001% and 0.0348%; throughput was still 7.06%
and 5.76% lower.

The paper audit corrected the first-link omission and repeated 1,000- and 10,000-search controls. Avoided evaluations
rose to 0.0249% and 0.1769%, and structurally unfolded trees would have duplicated 0.0814% and 0.5007% more nodes.
The graph nevertheless remained 8.63% and 8.28% slower. Maximum path multiplicity and shared-child visit advantages
confirmed that full descendant/statistic sharing was active; the negative result was not caused by implementing
only a leaf cache.

No strength match was run after the corrected audit because the measured compute economics were already strongly
negative. MCGS was implemented, validated, corrected, measured, and rejected for exact-history chess under this
workload.

### Rationale

The method depends on enough reusable state topology to repay hashing, collision verification, edge/node
indirection, graph maintenance, and poorer memory locality. Exact chess history semantics made that topology too
sparse. Relaxing identity might increase hits, but it would be a different approximate algorithm and could merge
states with different draw rights.

### Pitfalls

- A board hash is not a valid chess-state identity when repetition and the fifty-move rule matter.
- Transposition-table hit rate is not the same as neural evaluations avoided.
- Shared visits are information exposure, not extra completed Monte Carlo simulations.
- The first implementation missed a behavior present in the reference code; auditing the paper alone was
  insufficient.
- High-search, high-parallelism stress tests establish mechanics and asymptotics, not ordinary self-play value.
- The name “graph” is also used by CUDA graphs elsewhere; CUDA graph replay is unrelated to MCGS.

### Unknowns

- Approximate state identity was deliberately not evaluated.
- A different game with more exact transpositions may benefit even though chess did not.
- A substantially different network-to-CPU cost ratio could change the break-even point, but reuse would still need
  to grow by orders of magnitude relative to the corrected ordinary-budget measurements.

### Sources

- [Archived production decision](../../plan/archive/chess-four-day-baseline-and-next-run-plan.md#10-transposition-graph-search-and-inference-cache-decision---rejected)
- [Preserved MCGS design](https://github.com/BertilBraun/Advanced-Techniques-in-Chess-Engines/blob/mcgs-rejected/documentation/architecture/monte-carlo-graph-search.md)
- [Corrected paper audit](https://github.com/BertilBraun/Advanced-Techniques-in-Chess-Engines/blob/mcgs-rejected/documentation/benchmarks/monte-carlo-graph-search-paper-audit-rtx4070s-20260818/README.md)
- [Low- through high-budget graph benchmark](https://github.com/BertilBraun/Advanced-Techniques-in-Chess-Engines/blob/mcgs-rejected/documentation/benchmarks/monte-carlo-graph-search-rtx4070s-20260818/README.md)
- [Preserved native MCGS tests](https://github.com/BertilBraun/Advanced-Techniques-in-Chess-Engines/blob/mcgs-rejected/cpp/test/TestMonteCarloGraphSearch.cpp)
- [Czech, Korus, and Kersting, “Improving AlphaZero Using Monte-Carlo Graph Search”](https://arxiv.org/abs/2012.11045)

## Neural inference caching

### Question

Can repeated neural inputs be served from a cache, and can worker topology be reorganized to expose enough sharing
to make lookup, storage, synchronization, and memory worthwhile?

### Approaches

The repository contains two materially different cache investigations and they must not be conflated.

1. An implemented inference-result cache used a bounded sharded LRU inside each self-play process. Many MCTS
   threads within a process shared it. Topology experiments reduced process count and increased threads so more
   searches shared each cache and model instance.
2. A later production-shaped opportunity audit removed result reuse and instrumented exact encoded neural inputs.
   All inference workers owned by one `SearchExecutor` shared one unbounded tracker, every position was still
   evaluated, and model refresh cleared the tracker. This measured the most optimistic gross opportunity before
   implementing another cache.

Separately, the MCGS experiment shared rule state and search statistics. That is not an inference cache.

### Mechanism

An inference cache needs a key with the same semantics as the network input, a collision policy, stored policy/WDL
outputs, capacity and eviction, synchronization across submitters, and a model-revision boundary. Wider sharing can
increase reuse but also requires fewer, larger process domains or interprocess coordination. Fewer processes can
reduce duplicated caches and model instances but increase thread contention and make failure isolation coarser.

The measurement-only audit hashed the exact contiguous encoded input plus its dimensions with 128 bits. Rows were
observed at native submission before ordinary model execution. Same-batch and prior-batch repeats were counted
separately. Because the tracker was unbounded and never avoided work, its hit rate was an upper bound; a real finite
cache could only do worse before considering collisions and overhead.

### Evidence and result

The implemented per-process cache was measured with eight processes per GPU, three search threads per process, and
96 games per process. At capacity 1.5 million entries per process it achieved a 0.970% hit rate. Disabling it made
game updates about 0.88% faster in the short stochastic comparison and reduced summed worker peak memory by about
7.12%. This directly showed that the implemented cache consumed substantial memory without a demonstrated speedup.

Topology work then attempted to share caches more broadly inside each process: process count was reduced while the
MCTS thread count and per-process cache capacity were increased. This was a real architectural effort, not merely a
sentence in a future-work list. It could not make reuse cross an operating-system process boundary, however, and
the cache remained entangled with topology, model duplication, and thread contention.

The later opportunity audit tested 512 distinct starting games and exact production-style move progression. Fixed
150-search workloads repeated only 1.33% of inputs; fixed 800-search workloads repeated about 4.24%. The
decision-relevant 25% 800-search / 75% 150-search mixture repeated 3.5485% with one parallel leaf and 3.5295% with
two. Across roughly 1.83 million positions per arm, same-batch reuse was zero and one input. Following retained
trees for six moves raised the rate only to about 4%.

The audit also documented why apparently spectacular preliminary numbers were false. A harness that recycled a
fixed 50-opening suite and repeatedly restarted finished games produced 98.5% and 37.6% repeat rates. Once starts
were diverse and the workload matched self-play, those figures disappeared. Even the unbounded tracker imposed
about 3.65% throughput overhead in a separate short control and grew continuously. A finite, synchronized cache
would add eviction, output storage, device transfers, and more misses to an opportunity of similar magnitude.

The second cache design was therefore declined before implementation. Together, the two investigations are
stronger than either alone: one shows that a real bounded cache with narrow sharing failed, and the other shows that
widening sharing within a production search executor did not expose enough exact reuse to justify a new design.

### Rationale

Neural caching is valuable only if identical encoded positions recur often enough within one model revision. Broad
self-play diversity, root noise, history planes, and distinct game trajectories made exact reuse rare. Merging
workers solely to enlarge a cache domain traded against process isolation, batching, memory, and CPU contention
without changing the basic scarcity of repeated inputs.

### Pitfalls

- Fixed opening suites and automatic restarts can manufacture enormous cache hit rates.
- A cache shared by threads is not shared by processes; “per GPU” and “per process” are different domains.
- Cache hit rate is not net speedup. Hashing, synchronization, copies, storage, and eviction must be paid.
- Same-batch coalescing was effectively nonexistent; almost all opportunity required retaining outputs over time.
- Cache entries must be invalidated on model refresh.
- Encoded-input equality is intentionally different from graph-search state identity: a neural cache reuses only
  outputs, not rule state, visits, values, or descendants.
- The old historical optimization note claimed a 10–15% cache rate without preserved representative evidence; the
  controlled benchmarks supersede that claim.

### Unknowns

- No interprocess or GPU-wide cache was implemented. The measured upper bound indicates little room, but the exact
  coordination cost was therefore not measured.
- A highly repetitive analysis service or fixed-opening evaluator could have different economics from self-play.
- Cache behavior for the final all-full search mixture was not remeasured, though the fixed 800-search arm provides
  a relevant upper bound near 4.24%.

### Sources

- [Implemented cache benchmark](../../benchmarks/self-play-mcts-node-arena-20260720/README.md)
- [Preserved cache opportunity audit](https://github.com/BertilBraun/Advanced-Techniques-in-Chess-Engines/blob/inference-cache-declined/documentation/benchmarks/inference-cache-hit-rate-20260818/README.md)
- [Archived cache decision](../../plan/archive/chess-four-day-baseline-and-next-run-plan.md#10-transposition-graph-search-and-inference-cache-decision---rejected)
- [Bounded sharded-cache implementation commit](https://github.com/BertilBraun/Advanced-Techniques-in-Chess-Engines/commit/837cfd49)
- [Shared-result cache synchronization commit](https://github.com/BertilBraun/Advanced-Techniques-in-Chess-Engines/commit/fcbdb4d9)
- [Cache-heavy topology selection commit](https://github.com/BertilBraun/Advanced-Techniques-in-Chess-Engines/commit/8914afce)

## Native inference, TorchScript, `torch.compile`, and TensorRT

### Question

Which execution path supplies neural evaluations to search most efficiently without changing model semantics, and
which apparent compiler speedups survive comparison with the actual production path?

### Approaches

- Search and inference coordination moved from Python messages into a native C++ pipeline.
- Direct inference eliminated serialized request/response handling from the search loop.
- TorchScript supplied a trimmed policy/WDL artifact, fused model operations, and supported CUDA graph capture.
- `torch.compile` was tested in eager model diagnostics and in trainer configurations.
- TensorRT FP16 was integrated as a native fixed-batch backend.
- Post-training INT8, partial quantization, SmoothQuant, weight-only variants, FP8, quantization-aware training, and
  quantization-friendly residual blocks were investigated.
- Refittable TensorRT templates separated expensive engine construction from frequent weight publication.

### Mechanism

The retained inference boundary exposes only policy logits and normalized WDL probabilities. The native pipeline
preallocates host/device slots, accepts leaves from many roots, runs a dedicated inference thread, and allows a
bounded number of outstanding batches. TorchScript can capture several batch-size CUDA graphs to reduce launch
overhead. TensorRT uses fixed-batch ONNX deployment graphs and engine templates; recurring publication refits named
weights and quantization constants into a compatible template.

`torch.compile` and TorchScript are not interchangeable labels for “compiled.” An eager/compiled PyTorch diagnostic
can favor `torch.compile` while a fused TorchScript artifact still wins in the real serving comparison. TensorRT is
another compiler/runtime boundary with its own graph, shape, precision, tactic, refit, and fidelity contracts.

### Evidence and result

#### Native direct inference

The native port removed Python inner-loop coordination and enabled large cross-game batches. Historical speedups
span changing hardware, models, and workloads, so they should be described as engineering progress rather than one
controlled multiplicative result. The important architectural result is that native search owns encoded inputs,
submission, result processing, and tree mutation.

#### `torch.compile`

In a direct-policy eager diagnostic, `torch.compile` improved batch-64 forwards by roughly 27–33% across the tested
model sizes. That was not the production comparison. The corrected benchmark used the actual fused TorchScript
artifact and found it faster than the earlier compiled path. The project therefore did not retain `torch.compile`
for search inference. It remained useful in selected training experiments, but compiled execution also caused a
gradient-probe failure because a captured activation became a sibling graph output without the expected autograd
path. This is both a performance and observability lesson.

#### TorchScript and CUDA graphs

TorchScript was the validated serving control and remains the bootstrap/fallback path. CUDA graph replay reduced
host dispatch but introduced capture/replay concurrency constraints. The system serialized replay per device,
shared one transfer/replay stream, and tested multiple worker topologies. Faster kernels exposed submission,
copying, and result processing as first-order costs; isolated model positions per second did not determine actor
throughput.

#### TensorRT FP16 and INT8

Native TensorRT FP16 reached about 1.86 times the TorchScript BF16 search rate in one matched backend benchmark.
TensorRT INT8 added about 1.31 times over its FP16 denominator in the measured quantization-oriented model. These
ratios combine runtime, precision, and sometimes architecture, so the denominator must always be named.

The terminal retrospective adds an eight-hour online boundary check. Matched campaigns differing in quantization
and engine templates ended at generations 324 versus 331 and 40.6 versus 41.3 million training presentations; the
INT8/QAT arm was marginally behind. Its trainer processed about 16,933 samples/s versus 19,029 for the FP16 arm,
because fake-quantization work slowed optimization. Their single-rung ladder estimates, 1,800 and 1,786 Elo, do not
resolve a strength difference. The control spent almost all of its time on the smallest network stage, where INT8
has the least inference advantage, so it is a bounded negative result rather than proof that INT8 cannot help a
larger actor workload.

Naive full-trunk post-training INT8 was fast but semantically invalid: policy and value outputs changed
catastrophically. Calibration sweeps, partial early-block quantization, SmoothQuant, weight-only routes, and FP8 did
not meet the joint fidelity/speed gate. The successful route changed the trainable architecture to bounded scaled
post-activation residual blocks, trained with fake quantization, kept sensitive heads and linear layers outside
INT8, and served a pre-fold deployment copy.

Refit made per-checkpoint publication practical, but it exposed a particularly dangerous failure. TensorRT could
optimize a template around equal quantization scales, report successful refit after those scales became distinct,
and still compute invalid outputs at high optimization levels. Pairwise-distinct template scales plus a safer
optimization level fixed the controlled failure. Random unmasked probes had hidden the problem; real encoded chess
positions, legal-action masking, policy KL, and WDL error became the meaningful fidelity checks.

### Rationale

TorchScript was superseded as the main post-bootstrap serving runtime because TensorRT offered a larger compatible
throughput path. `torch.compile` was rejected specifically for search inference because it did not beat the actual
TorchScript control, not because compilation never helped PyTorch. INT8 was retained only after model architecture,
training, export, refit, and semantic validation were designed together.

The end-to-end control sharpens the causal language: the completed campaign's roughly 1.9-fold production advantage
over its predecessor cannot be attributed to FP16-to-INT8 conversion. The directly measured INT8-versus-FP16 online
pair showed no generation or presentation-rate win. The supported systems account is that moving from TorchScript
to TensorRT supplied the large runtime gain, while INT8 remained a stage- and workload-dependent microbenchmark
gain whose QAT trainer cost could cancel it end to end.

### Pitfalls

- Comparing `torch.compile` with eager mode does not answer whether it beats fused TorchScript.
- Comparing TensorRT INT8 with TorchScript mixes precision, runtime, and possibly architecture; FP16 TensorRT is the
  clean runtime denominator.
- A successful engine build or refit is not evidence of semantic correctness.
- Random positions and unmasked top-one accuracy can overstate meaningful policy divergence.
- Fixed-batch engine throughput does not equal self-play throughput; host submission, search CPU, batching, game
  completion, replay admission, and trainer overlap intervene.
- CUDA graph replay and Monte Carlo graph search share the word “graph” but solve unrelated problems.
- Compiler capture can invalidate diagnostic instrumentation even when training loss appears normal.

### Unknowns

- The final large model's exact engine/template/refit identity, realized precision path, and matched throughput still
  need to be frozen from the terminal archive.
- Final telemetry must show how often TensorRT, FP16 fallback, and bootstrap TorchScript actually ran.
- End-to-end Elo benefit comes only from the final learning curve; backend throughput alone cannot establish it.
- The online INT8/FP16 control changed quantization state and engine templates together. It bounds the combined
  deployment choice; it does not isolate individual kernels, QAT regularization, or template construction.

### Sources

- [Inference experiment ledger](../../experiments/inference-and-throughput.md)
- [`torch.compile` eager diagnostic](../../benchmarks/chess-direct-policy-inference-rtx4070s-20260818/README.md)
- [Corrected TorchScript control](../../benchmarks/chess-direct-policy-kernel-controls-rtx4070s-20260818/README.md)
- [Native TensorRT backend](../../benchmarks/tensorrt-native-backend-rtx4070s-20260912/README.md)
- [TensorRT INT8 architecture screen](../../benchmarks/tensorrt-int8-architecture-screen-rtx4070s-20260912/README.md)
- [TensorRT salvage investigation](../../benchmarks/tensorrt-int8-salvage-rtx4070s-20260912/README.md)
- [Frozen-replay QAT and refit study](../../benchmarks/tensorrt-int8-replay-screen-rtx4070s-20260912/README.md)
- [TensorRT template failure investigation](../../benchmarks/int8-template-staleness-rtx4070super-20260921/README.md)
- [Current inference boundary](../../system/inference-and-evaluation.md)
- [Native inference pipeline](../../../cpp/src/search/InferencePipeline.hpp)
- Local terminal retrospective and TensorBoard evidence: `C:\Users\berti\Downloads\RECAP.md` and
  `C:\Projects\AZ\.codex-diagnostics\final-2026-09-23\evidence-tensorboard.tgz` (pending a committed comparison
  table)

## Cross-cutting lessons for the report

### Treat major rejected systems as full investigations

MCGS, inference caching, predicted allocation, learned stopping, and compiler/runtime selection each involved a
substantial design, implementation, validation, and measurement effort. They should each receive a question,
mechanism, evidence, rejection reason, and limitation. Listing them as “not retained” after the successful system
erases the scientific content.

### Organize by decision, not chronology

The relevant structure is:

1. what resource or learning bottleneck was hypothesized;
2. what mechanism was designed to address it;
3. what proxy and outcome were measured;
4. whether the mechanism worked;
5. whether it improved the outcome that mattered;
6. why it was retained, superseded, or rejected.

Temporal narration is useful only inside a pitfall when the order explains the diagnosis—for example, a misleading
cache harness being corrected by a representative one, or a TensorRT “stale template” hypothesis being replaced by
the equal-scale optimization diagnosis.

### Preserve negative-result boundaries

- Threshold stopping was audited and declined; it was not beaten in a strength match.
- Predicted allocation worked on KL and failed on learning outcome.
- Learned stopping saved search but did not move the critical path enough to resolve a benefit.
- MCGS was fully implemented and corrected, then rejected on exact-state reuse economics; no final strength match was
  necessary or run.
- A bounded process-local cache was implemented and unfavorable; a later broader cache design was declined after an
  upper-bound audit.
- `torch.compile` helped an eager diagnostic but lost to the production TorchScript comparison.
- Post-training INT8 was fast and invalid; quantization-aware architecture and training, not looser acceptance, made
  INT8 usable.

### Keep the metrics ladder explicit

For every optimization, report as many of these as the evidence supports, without substituting one for another:

1. model positions per second;
2. neural evaluations avoided;
3. Monte Carlo simulations per second;
4. searches or moves completed per second;
5. games completed;
6. replay rows admitted;
7. optimizer steps or training generations per hour;
8. playing strength per wall-clock hour.

## Consolidated unresolved questions

- Which final serving artifacts and precision paths actually handled each training interval?
- What was the matched final-model throughput of the largest TensorRT deployment?
- How much did the final all-full search workload change the best process/game/leaf topology relative to the older
  mixed workload?
- Can final telemetry separate core inference gains from search throughput, games, replay admissions, and learner
  cadence?
- Which retained MCTS choices have only recipe/external support, and which have local isolated evidence?
- Should a future study revisit cheap targets only with explicit target weighting or exclusion?
- Are any cache or graph-search conclusions intended to transfer beyond diverse chess self-play? The current evidence
  does not justify that generalization.
