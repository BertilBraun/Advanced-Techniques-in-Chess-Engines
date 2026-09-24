# Evaluation methodology and transferable failure studies

This is a source dossier for the technical report, not publication prose. It is organized by measurement question
and failure mechanism. Internal training-run identifiers are intentionally absent from the narrative. Where a
source filename contains one, the link is described by the evidence it contains rather than by that identifier.

The central rule is that the project measured several different objects which cannot be substituted for one
another:

- **playing strength**: game outcomes against a fixed opponent under a specified search, opening, colour, and
  adjudication protocol;
- **policy quality without meaningful tree search**: direct legal-policy move selection, or the closely related
  one-root-expansion instrumentation used during training;
- **search-target fidelity**: distance between a shallow search policy and a deeper reference policy on fixed
  positions;
- **model-kernel throughput**: positions evaluated per second in a warmed inference harness;
- **search throughput**: simulations or completed search requests per second in the native actor topology;
- **live replay throughput**: accepted training rows produced per second after games finish and materialization
  rules are applied;
- **optimizer cadence**: training quanta or optimizer updates completed per unit wall time;
- **learning cadence**: strength gained per unit wall time under the complete online system.

An improvement at an earlier boundary is useful only if it survives every downstream boundary relevant to the
claim. Faster kernels do not guarantee more completed games; more simulations do not guarantee more admitted rows;
more rows do not guarantee fresher information; faster optimizer quanta do not guarantee better targets; a target
closer to deep search does not guarantee higher playing strength; and higher fixed-network strength does not prove a
better self-play learning recipe.

## Policy-only evaluation

### Question

How strong is the learned policy before meaningful tree-search improvement, and can policy learning be monitored
separately from search quality?

### Mechanisms and alternatives

The repository has two related but non-identical policy measurements:

- The scheduled training-time ladder gives the native search one root expansion and then selects the highest-visit
  move. This deliberately traverses the ordinary legal-action, deployment-artifact, and native inference boundary.
  With one search, it is effectively the network policy after root expansion, but it should not be described as
  literal direct argmax.
- The terminal evaluation tool supports direct masked-policy argmax through `PolicyActionSelector`. That path runs
  the deployment model, masks to legal actions, and chooses from the resulting policy without MCTS. This is the
  appropriate meaning of **policy only** in the final evaluation table.

Both remove root noise and stochastic move sampling. They answer whether the policy itself improved, not how much
strength search adds. The distinction between the one-expansion instrumentation and direct selection must remain in
the report because apparently identical labels otherwise hide a real protocol difference.

### Evidence and rationale

The previous terminal suite retained a 400-game direct-policy match alongside searched matches. That made it
possible to report the network's policy floor and the incremental benefit from search under one opening suite. The
final suite retains that structure: direct masked-policy play measured 1,658 benchmark Elo, while the searched matrix
measured the incremental benefit across four search budgets. The protocol details and intervals are in the final
result record.

Policy-only strength is also a useful diagnostic. If searched strength falls while policy-only strength does not,
the search or serving path becomes a stronger suspect. If both deteriorate together, training data, optimization,
initialization, or deployment fidelity become more plausible. It is not a sufficient diagnostic by itself because
value quality can affect search without changing policy argmax.

### Pitfalls and unknowns

- One search is not mathematically identical to direct policy argmax; the final report must name the actual selector.
- Policy-only Elo cannot be compared with searched Elo without retaining each opponent rung and candidate protocol.
- A policy match does not isolate the policy head: the trunk, representation, legal mask, deployment precision, and
  training distribution all contribute.
- The final direct-policy result is complete: 1,658 benchmark Elo through the float TorchScript artifact. It is a
  separate serving path from searched INT8 TensorRT play.

### Sources

- [Scheduled inference and evaluation contract](../../system/inference-and-evaluation.md)
- [Final evaluation protocol template](../../plan/v34-final-evaluation-and-distillation.md)
- [Direct policy selector](../../../py/src/evaluation/inference.py)
- [Match selector boundary](../../../py/src/evaluation/match.py)
- [Previous terminal evaluation archive](../../benchmarks/chess-terminal-v34-generation1465-rtx4070s-20260911/README.md)

## Fixed-search and deep-search strength evaluation

### Question

How much playing strength does search add at small, medium, and deep budgets, and how should a final checkpoint be
compared with an external engine?

### Mechanism

A fixed-search match constructs a fresh native root for each move, runs a configured number of additional visits,
and selects the action with the largest visit count, breaking ties by canonical action ID. Evaluation disables root
noise and forced playouts. Every quoted result must retain:

- candidate visits per move;
- parallel leaf searches;
- inference backend, precision, batch capacity, and outstanding-batch count;
- exploration and first-play-urgency settings;
- external-engine identity, thread count, hash size, and fixed nodes per move;
- opening manifest and hash;
- game count, colour pairing, maximum plies, and adjudication behavior.

Small fixed search, 64 visits in scheduled evaluation, is cheap enough to track throughout training. It is a
regression and progression instrument, not a terminal-strength ceiling. Earlier deep matches at approximately
10,000 and 80,000 visits showed that one frozen network remained search-scalable far beyond the scheduled ladder.
The completed final suite measured policy-only play and 100, 1,000, 10,000, and 100,000 searches per move, retaining
the opponent rung, parallelism, backend, and uncertainty for each condition.

### Opponent selection

Short Stockfish probes exist only to find an opponent near a 50% expected score. They are not publishable strength
measurements. The selected opponent then receives a full paired-opening match. A near-even opponent is statistically
preferable: the earlier deep-search probe chose an opponent that was too weak, producing a 77.4% score; a later
confirmation against a stronger rung moved the central estimate by only seven Elo but materially tightened the
conditional interval.

### Rationale

The combination of policy-only, small fixed search, moderate/deep search, and very deep search reveals whether gains
come from the learned policy, searchability of the policy/value pair, or both. A single headline search budget would
hide this distinction. Fixed budgets also make cross-checkpoint comparison auditable; a time limit would couple the
result to backend speed and hardware.

### Pitfalls and unknowns

- Parallel leaf searches trade throughput for stale or virtually reserved traversals. Results at different
  parallelism are not silently comparable.
- Saturated batched throughput is not single-game response latency. The earlier 80,000-visit timing result used 50
  simultaneous positions per GPU and averaged 5.31 seconds per position; that is a service-rate measurement.
- A fixed visit count can still inherit retained work in self-play, but match evaluation creates fresh roots.
- The final fixed- and deep-search results, selected Stockfish rungs, timings, and intervals are in the
  [result record](../../results/final-chess-run.md).

### Sources

- [Native match search implementation](../../../py/src/evaluation/match.py)
- [Previous terminal fixed-search protocol and results](../../benchmarks/chess-terminal-v34-generation1465-rtx4070s-20260911/README.md)
- [Search-depth and fidelity investigation](../../analysis/chess-search-findings-20260827.md)
- [Terminal evaluation protocol template](../../plan/v34-final-evaluation-and-distillation.md)

## Stockfish calibration, openings, colours, adjudication, and uncertainty

### Question

What does a Stockfish-ladder Elo number mean, and which controls make comparisons trustworthy?

### Calibration mechanism

The ladder uses one-thread Stockfish 13 at fixed nodes per move. Each node rung has an external anchor read from a
historical Stockfish-node curve connected to Fruit and the SSDF engine list. For one fixed opponent, performance Elo
is

`opponent anchor + 400 * log10(score / (1 - score))`.

For a multi-rung ladder, the implementation fits one rating whose logistic expected scores best match the weighted
observations across rungs. This scale is useful for internally consistent project comparisons. Its absolute zero is
conditional on the historical anchor chain and does not turn the result into a FIDE rating.

### Openings and colours

Evaluation uses checked-in opening manifests rather than freshly sampled openings. Each opening is played twice,
with the candidate taking both colours. The pair—not the individual game—is the resampling unit because the two
colours share opening difficulty. The final suite should retain the opening manifest hash, engine binary hash, match
seed, complete games, candidate colour, and termination reason.

Repeated openings are a variance-control device, not independent evidence. Reusing the same suite across
checkpoints supports paired comparisons, but those checkpoints must not be pooled as if every game were independent.
Opening-selection probes and final matches must also remain separate.

### Adjudication

Chess games reaching the evaluation ply cap are recorded as draws. The scheduled multi-rung Elo fitter excludes
ply-cap games because they are protocol artifacts rather than naturally resolved strength evidence; a rung with no
naturally completed game supplies no ladder signal. Fixed-opponent terminal summaries must state whether capped draws
were included in the reported score. This distinction should be verified from the archived final result rather than
assumed from the ladder code.

### Uncertainty

Match-score intervals are paired bootstraps over opening pairs. Ladder intervals additionally resample rungs so that
rung selection is not treated as fixed sampling evidence. Elo intervals transform the bootstrapped scores while
holding the historical rung anchors fixed. They therefore describe game-sampling uncertainty only; they do not
include systematic uncertainty in the SSDF-derived absolute scale, opponent implementation, opening distribution,
or protocol transfer to humans.

The report should publish W/D/L, score, paired interval, opponent node count, and conditional Elo interval together.
One-sided or clamp-touching intervals signal an under-bracketed ladder and must not be rendered as ordinary precise
two-sided estimates.

### Decision rationale

Paired openings, fixed-node opponents, and immutable artifacts give the project a reproducible internal scale. The
scale supports comparisons among models evaluated under the same protocol and a qualified statement that the model
lies above the top-human region under that calibration. It does not support “FIDE Elo,” a predicted score against a
specific human, or direct comparison with unrelated engine lists.

### Sources

- [Rating-scale audit and reporting rules](../../analysis/chess-elo-scale-and-reporting-20260911.md)
- [Ladder fit and paired-rung bootstrap](../../../py/src/evaluation/ladder.py)
- [Paired-opening match bootstrap](../../../py/src/evaluation/statistics.py)
- [Match construction, colour swap, and cap handling](../../../py/src/evaluation/match.py)
- [Immutable evaluation inputs](../../system/inference-and-evaluation.md)

## Cross-checkpoint curves, smoothing, and comparison

### Question

How can learning progress be shown without turning noisy scheduled evaluations or internal run boundaries into the
story?

### Mechanism

The natural x-axis is cumulative active training wall time. Checkpoints that continue the same learning state across
process restarts belong to one curve. Internal orchestration labels must not become separate series. Each plotted
point must still carry its checkpoint identity in the underlying data so that the figure remains auditable.

Only checkpoints evaluated with the same candidate budget, parallelism, Stockfish binary and rung anchors, opening
selection rule, and Elo fitter belong on one quantitatively comparable curve. Different rung sets can agree by
chance; a previous audit explicitly found that fits over different rung sets were not formally comparable even when
their central estimates nearly matched.

The runtime uses a bias-corrected exponential moving average of scheduled searched Elo to decide when to begin
training a larger model. The operational detector uses decay 0.90 and measures gain across a six-observation window,
requiring two consecutive complete below-threshold windows. That filter is a control signal, not a substitute for
publication data. A paper figure should show raw observations and, if useful, a clearly labeled descriptive smoother.
The smoother must not be presented with the raw pointwise confidence interval or forced to be monotonic.

### Cross-generation progression figure

The tracked figure compares representative historical checkpoints with the completed final training lineage at a
nominal 64-search ladder. The fitter changed across campaigns, so the figure is descriptive and the matched-estimator
plateau audit carries the quantitative comparison. Its construction rules were:

1. reconstruct cumulative active time across continuations;
2. verify identical ladder definitions and correct any known evaluation-path defects;
3. preserve raw points and their sample sizes;
4. distinguish interpolation or smoothing from observations;
5. state when a ladder saturates or loses upper rungs;
6. avoid inferring exact gains from adjacent points whose intervals overlap substantially.

The previous 10,000-visit checkpoint study is a warning: the 64-visit ladder preserved the broad curve shape but
understated the absolute level by roughly 550–600 Elo and saturated as the model outgrew its rungs. Thus the
64-search curve can demonstrate engineering progress under one fixed instrument, while terminal deep-search matches
must carry the final absolute-strength claim.

### Remaining boundary

The figure uses the established bias-corrected 0.95 EMA and the final lineage's exact 2.5-day cutoff. A complete
point-by-point shared-estimator reconstruction across every older campaign is not available, so adjacent displayed
curves do not establish exact Elo differences. The [matched-estimator plateau audit](#retrospective-matched-estimator-audit)
is the report's cross-campaign quantitative result.

### Sources

- [Deep-search re-evaluation of a complete learning curve](../../benchmarks/ladder-elo-vs-generation-rtx4070s-20260906/README.md)
- [Operational plateau detector](../../../py/src/training/progressive.py)
- [Progressive-sizing measurement semantics](../../architecture/progressive-model-sizing.md)
- [Final result and figure landing page](../../results/final-chess-run.md)

### Retrospective matched-estimator audit

The completed campaign exposed a comparison error that peak-based or same-tag plots conceal. The previous four-day
baseline logged only the single-rung fit under its generic ladder tag. The final campaign logged a three-rung
bracketed fit under that tag and retained a separate single-rung series. Comparing those generic tags directly would
therefore compare different estimators.

The retrospective used plateau windows rather than peaks and transferred the estimator difference only where both
fits were logged on the same 5,000-node rung. Across 35 such boundaries, the single-rung fit averaged 2.7 Elo below
the three-rung fit, with standard deviation 23.7. Applying that measured conversion to the previous plateau gives
2,283.9 Elo on the three-rung estimator; the final plateau averages 2,358.0, for a matched-estimator difference of
**+74.1 Elo**. The stated **approximately ±15 Elo** is a transfer/sensitivity allowance for the conversion, not a
game-level bootstrap confidence interval. It must be labeled as such.

The apparent alternative of roughly +102 Elo comes from comparing both campaigns' single-rung plateaus. That is
not the preferred result: during the final plateau the single-rung fit averaged 30.6 Elo above its own three-rung
fit, whereas earlier on the same 10,000-node rung it averaged 6.9 Elo below. A one-rung estimate changes bias as the
candidate's score moves away from 0.5. The publication-safe conclusion is therefore “about +74 Elo under a matched
three-rung estimator,” with the derivation and transfer uncertainty retained; neither raw peak difference nor the
larger single-rung number should be the headline comparison.

The same audit gives a descriptive, not causal, decomposition. With model shape and evaluation search settings held
the same, the one-expansion policy instrument improved from an estimated 1,689.0 to 1,719.3 Elo (**+30.3**), while
the 64-search estimate improved by 74.1 Elo. The residual **about +44 Elo** appears only after tree search. This says
that policy-only improvement does not account for the full matched gap; it does not identify which training change
caused the search-side residual, and it does not isolate the value head from policy/search interaction.

### Retrospective adaptive-rung failure

The previous baseline's terminal-looking uptick was an evaluation-state transition, not supported evidence of
continued learning. A single 100-game result at 3.938 active days scored 0.725 against the 5,000-node opponent,
roughly 3.4 standard deviations above the preceding 44-point plateau. That crossed the configured 0.70 advance
threshold. The next five estimates were therefore fitted against the 10,000-node anchor, 250 anchor Elo higher,
despite candidate scores of only 0.29–0.39; the final 0.285 score had already triggered retreat.

Two controls did not show the step: the policy-only ladder remained on its original rung, and fixed-dataset policy
accuracy stayed within 0.442–0.471 across the same boundaries. Descriptive EMA smoothing reduces the whole event to
roughly 20 Elo. This is sufficient to reject the narrative that the baseline was demonstrably rising at shutdown.
It is not sufficient to claim the underlying network changed by exactly zero. Every apparent ladder discontinuity
must therefore be checked against rung transitions before it is interpreted as learning.

### Retrospective sources and archive status

- The [compact final evidence index](../../evidence/final-chess-20260923/README.md) records the checksum-covered
  TensorBoard bundle and the committed 13-series ladder export, including stitched and raw seconds for the final
  lineage.
- The operator retrospective supplied on 2026-09-23 was checked against those archives where the underlying result
  manifests were available.
- The matched-estimator calculation and rung-event audit are not yet committed as a standalone machine-readable
  derivation. Publication should preserve the calculation inputs and script rather than cite this dossier alone.

## Transferable failure study: late-game target poisoning

### Failure

An early recipe attempted to save endgame search by switching to cheap searches before the game-length cap. Only
full-search observations were admitted as primary rows. The omitted observations were not random: they were the last
plies of the longest games, exactly the conversion phase the network needed to learn.

The failure compounded at the cap. A capped game's final value was bootstrapped from its last root value, but the
last root necessarily came from the cheap-search region. That one weak root estimate was then stamped, with
perspective changes, onto every admitted observation from the game—including opening positions. During the worst
early interval, about one ply in seven was structurally excluded from training, 27–32% of games reached the cap, and
roughly 36–38% of admitted rows inherited a cut-game value produced by an 81–136-visit search.

The model then produced drawn-out games it could not reliably convert. By the time the active replay distribution
looked healthy, the poisoned rows had been evicted, but the damaged weights and resulting self-play distribution
could persist. Looking only at the current replay buffer therefore initially hid the causal event.

The owner describes this as a closed feedback loop: missing full-search endgame targets left the late-game policy
weak; cheap searches driven by that policy played nearly randomly and rarely converted; the early cutoff then supplied
the heuristic value that trained the same weak behavior. Neither better play nor a trustworthy terminal target entered
the loop often enough for it to self-correct.

### Correction and lesson

The safe design aligns search eligibility and game termination: do not create an untrained late-game dead zone, and
do not propagate a low-quality cap estimate through an entire trajectory. Current cut semantics, restart-state
storage, auxiliary eligibility, and final-value provenance must be audited together rather than as independent
knobs.

The repair had two stages. First, the cutoff heuristic was replaced by one full search at the final cut position and
that searched root value supplied the bootstrap target. Later, the forced fast-search tail was removed so properly
searched endgame positions again entered replay. Other late-game changes were bundled around the same period, so this
chronology explains the mechanism without assigning an isolated Elo effect to either step.

This incident also demonstrates why replay snapshots are insufficient for diagnosing online learning. Telemetry and
historical materialization semantics are necessary when transient bad data can alter weights after the rows vanish.

### Evidence limits

The row-level reconstruction was available for the affected live store and later frozen references, but not for
every converting control. Some matched-period comparisons therefore use sampled telemetry rather than preserved
rows. The evidence strongly identifies the mechanism; it is not a clean one-field long-run ablation.

### Sources

- [Training-data and late-game reconstruction](../../analysis/v8-training-data-comparison-20260826.md)
- [Cut-game value-target benchmark](../../benchmarks/cut-game-value-target-rtx4070super-20260825/README.md)
- [Replay materialization implementation](../../../py/src/replay/materialization.py)

## Transferable failure study: architecture comparisons confounded by initialization and runtime

### Failure

Early convolution-versus-attention comparisons combined several changes:

- an action-plane policy representation rather than the established canonical-action head;
- inappropriate Kaiming/ReLU initialization for a bare policy projection;
- missing final normalization and convolution-oriented initialization in the attention trunk;
- different inference precision and backend behavior;
- replay-ingestion starvation and shared-GPU contention.

Real-position probes found initial policy-logit standard deviations of 8.4–11.9 for attention and 4.6 for the
convolutional plane head, versus about 1.0 for the established dense head. These produced near-one-hot random priors
in one path, while another failed path produced a nearly uniform prior. Online failure under those conditions was
not evidence that attention or the plane representation was intrinsically weak.

An inference benchmark introduced a second trap: an early eager/compiled measurement used FP32 and suggested one
ranking; a later measurement of the actual fused BF16 TorchScript artifact changed the conclusion. Shared GPU load
also made some recorded training-throughput numbers vary by more than fourfold for reasons unrelated to architecture.

### Correction and result

The controlled follow-up held the teacher dataset and training horizon fixed, compared heads on the same trunk, and
remeasured inference on an idle production-class GPU. It found that most of the apparent policy-quality gain came
from the from-to head, not the attention trunk. The best attention cell retained a modest proxy advantage but paid a
large memory and serving-throughput cost. This supports the retained convolutional trunk for this workload; it does
not support a general claim against transformers.

### Lesson

Architecture comparisons must align initialization, policy representation, objective, parameter accounting,
precision, runtime path, batch shape, hardware occupancy, and dataset. A one-seed frozen-teacher loss interval
measures held-out-position sampling, not seed variance or online Elo.

### Sources

- [Controlled attention, head, and throughput study](../../benchmarks/chess-attention-viability-rtx3060-20260827/README.md)
- [Failure diagnosis and initialization measurements](../../plan/chess-post-four-day-regression-analysis-20260820.md)
- [Corrected fused-runtime policy benchmark](../../benchmarks/chess-direct-policy-kernel-controls-rtx4070s-20260818/README.md)
- [Architecture and policy dossier](network-architecture-and-policy.md)

## Transferable failure study: configured seeds did not seed model construction

### Failure

The configured training seed reached self-play workers and evaluation data but not generation-zero network
construction. The model was instantiated before `torch.manual_seed` ran; trainer ranks then loaded the already
created checkpoint. Ranks within a distributed job agreed, but two supposedly matched jobs did not begin from the
same weights.

Bootstrap policy calibration obscured the defect. It scaled aggregate top-three policy mass toward a target but did
not make action rankings, relative gaps, or values identical. Observed initial top-one mass varied by nearly a factor
of two among nominally comparable initializations, and the policy-entropy ratio varied from 0.816 to 0.898.

### Correction and lesson

Model construction is now seeded immediately before instantiation and tested for same-seed tensor identity,
identical calibration, and different-seed divergence. A controlled online comparison must import one immutable
initial checkpoint or prove tensor identity; matching a configuration seed is not enough.

The incident invalidates causal attribution based on independent-from-scratch arms whose expected effect is smaller
than initialization and trajectory variance. Later adaptive-search work therefore forked byte-identical model and
replay states for the decisive comparison.

### Sources

- [Initialization confound and executable bisect](../../analysis/v35-v42-executable-bisect-20260913.md)
- [Source audit of the initialization boundary](../../analysis/v35-v42-regression-audit-20260913.md)
- [Adaptive-search conclusion and seed discovery](../../analysis/adaptive-search-conclusion-20260904.md)
- [Current bootstrap initializer](../../../py/src/training/bootstrap.py)

## Transferable failure study: misleading inference-cache measurements

### Failure

The first inference-repetition harness reported 98.5% repeated inputs at a low search budget and 37.6% at a higher
budget. Those striking values came from an evaluation-like workload: it repeatedly cycled a fixed 50-opening suite,
restarted completed games from the same openings, and used up to 64 parallel leaves. The harness measured its own
artificial recurrence, not production self-play.

The corrected harness used hundreds of distinct randomized starting positions, production root noise and search
constants, retained roots, realistic inference concurrency, and matched seeds. Pure fixed-budget search produced
only about 1.3% ideal repeats at 150 visits and 4.2% at 800 visits. The decision-relevant mixed workload produced
about 3.5% at both one and two parallel leaves, with essentially zero same-batch repetition.

### Architecture considered

The investigation did not merely count local cache hits. It explored the motivation for sharing a cache among more
self-play games by combining inference work behind common workers. Even under the favorable unbounded measurement,
the gross opportunity was too small to pay for finite capacity, lookup, synchronization, collision verification,
output storage, device transfers, and cross-worker coordination. The measuring hash set itself cost about 3.7% in a
short A/B and grew without bound—already comparable to the available gross saving.

### Decision and lesson

The cache was rejected for this self-play distribution. The transferable lesson is to measure an ideal upper bound
at the exact production submission boundary before designing cache architecture, and to preserve opening diversity,
game lifecycle, tree retention, root noise, batch topology, and model-refresh resets in that measurement.

### Sources

- [Inference-cache release record](https://github.com/BertilBraun/Advanced-Techniques-in-Chess-Engines/releases/tag/inference-cache-declined)
- [Preserved cache benchmark at its release tag](https://github.com/BertilBraun/Advanced-Techniques-in-Chess-Engines/blob/inference-cache-declined/documentation/benchmarks/inference-cache-hit-rate-20260818/README.md)

## Transferable failure study: adaptive-search proxy improved while strength did not

### Failure

A learned allocator predicted the divergence between deep and budget-limited search policies, then selected a
per-position budget under a mean-compute constraint. Offline and live telemetry showed that the mechanism worked:
it captured about 23% of measured oracle headroom, achieved approximately 1.22 times effective search compute at
slightly below unit spend, and allocated more work to contested positions.

Yet repeated online attempts were 60–100 Elo behind non-adaptive references. The likely mechanism is objective
mismatch. Per-position divergence to a deep search measures search fidelity, not the training value of the target
written to replay. The allocator assigned very cheap searches to many positions and admitted those shallow,
self-referential visit distributions at full policy weight. Improving the proxy could therefore worsen the next
generation's data.

A second learned system stopped search online after observing the tree. A byte-identical fork experiment proved that
it skipped 14% of search work as designed. It still produced no detectable strength difference: matched differences
were about two to four Elo with standard errors near ten Elo.

### Decision and lesson

Both adaptive systems were rejected. The first failed the strength test despite succeeding on its proxy. The second
saved work but not enough critical-path time to produce a resolvable learning benefit. Future adaptive-search work
would need target-admission or weighting rules and an end-to-end strength design, not merely a better predictor.

### Sources

- [Predicted-budget negative result](../../analysis/adaptive-search-budget-negative-result-20260901.md)
- [Learned-stopping fork conclusion](../../analysis/adaptive-search-conclusion-20260904.md)
- [Adaptive-stopping release record](https://github.com/BertilBraun/Advanced-Techniques-in-Chess-Engines/releases/tag/adaptive-stopping-final)

## Transferable failure study: saved search did not become wall-clock progress

### Failure

The learned stopper's headline was a 14% reduction in search work. End-to-end generation time improved only about
3%. Self-play overlapped with distributed training: part of the removed search occupied slack while the trainer was
already on the critical path. At the measured learning slope, the cadence gain was worth roughly one Elo over three
hours, an order of magnitude below the experiment's approximately ten-Elo paired resolution.

The pause-topology study found the same systems principle from another direction. Allowing more actors to run beside
the trainer increased search work but slowed each distributed training rank. Because DDP follows its slowest rank
and because full actor capacity returns after training, the generation-time spread among realistic choices was only
about 5%, not the multi-fold result suggested by a faulty imbalanced benchmark.

### Correction and lesson

Measure the critical path directly: trainer duration, concurrent actor output, credit wait, pause and refresh
barriers, checkpoint publication, and total boundary time. Simulations saved, actor searches per second, or isolated
trainer samples per second are intermediate counters. They become valuable only when the coordinator's dependency
graph turns them into more learning per wall hour.

### Sources

- [Learned-stopping end-to-end conclusion](../../analysis/adaptive-search-conclusion-20260904.md)
- [Actor/trainer overlap benchmark and placement bug](../../benchmarks/selfplay-pause-tradeoff-rtx4070s-20260902/README.md)
- [Training throughput and transition-barrier decomposition](../../benchmarks/chess-training-throughput-rtx3060-20260812/README.md)

## Transferable failure study: replay throughput did not equal fresh information

### Failure mechanism

The system separates simulations, completed games, materialized observations, accepted replay rows, sampled rows,
and optimizer presentations. These rates diverge for structural reasons:

- replay credit appears only after a game finishes;
- long games are right-censored in short throughput harnesses;
- only eligible observations materialize;
- game length controls positions per completion;
- replay reuse controls how many new rows are required per optimizer quantum;
- buffer capacity and selection weights control which accepted rows are sampled;
- concurrent actors are paused or contended during training and checkpoint activation.

A TensorRT actor benchmark measured approximately 1.86 times the simulation throughput of a close floating control,
while the live pipeline produced only 23.3% more accepted positions per second than an earlier comparable regime.
Games carried 29.5% fewer accepted positions, and the newer reuse setting required 28% more fresh rows per training
quantum. Materialization itself ran 6.4 times faster than arrivals and was not the bottleneck.

### Interpretation

Higher replay throughput can still mean less useful novelty if it comes from shorter, easier, duplicated, weakly
searched, or heavily reused trajectories. Conversely, lowering reuse slows optimizer generations while increasing
fresh information per presentation. Neither direction is a free throughput optimization; it changes the learning
problem.

### Lesson

Report the full chain: simulations/s, completed games/s, eligible positions/game, accepted positions/s, replay age
and diversity, configured and observed reuse, presentations/s, generations/hour, and Elo/hour. Short actor harnesses
should not use completed games or positions/s as steady-state metrics unless the initial population has drained and
right censoring is controlled.

### Sources

- [Native actor, live replay, and cadence decomposition](../../benchmarks/v39-selfplay-throughput-rtx4070s-20260913/README.md)
- [Replay and data-system contract](../../system/replay-and-data.md)
- [Data and replay investigation chapter](../04b-data-and-replay.md)

## Transferable failure study: TensorRT conversion and refit were mechanically successful but semantically wrong

### Failure sequence

The serving pipeline builds a structural TensorRT template, then refits it with each checkpoint. A production
template was built from an early quantized checkpoint in which many activation scales were numerically equal because
they saturated at the clip bound. At high TensorRT optimization levels, the builder optimized around those
equalities. A later checkpoint had distinct scales. The refitter accepted every weight, reported no missing weights,
and returned success—but the engine produced incorrect, nondeterministic logits after refit.

On 516 real positions, the faulty high-optimization template produced legal-policy KL around 1.05 and repeat-refit
logit deltas of 10–15. The same source at lower optimization levels, or with scale constants made distinct before
template build, produced legal KL around 0.0012 with deterministic repeated refits. Exact-zero weight folding was
tested and rejected as the cause.

This was especially dangerous with progressive sizing and quantization-aware training. Freshly grown models tend to
have saturated equal scales; later training separates them. Structural compatibility and refit API success therefore
did not imply semantic compatibility across checkpoint age.

### Correction and lesson

Template construction now separates equal quantization scales before building and defaults to a safer optimization
level. More importantly, every publication must compare outputs on real encoded probes: legal top-one agreement,
legal-policy KL, and WDL error. An import smoke, successful deserialization, complete refit-weight accounting, or one
synthetic tensor is insufficient.

The final recipe currently allows a fidelity deviation to warn rather than abort because quantized deployment is an
explicit experimental choice. A warning is evidence of deviation, not a passed fidelity gate. Final reporting must
archive the exact source checkpoint, ONNX graph, template hash, TensorRT/runtime identity, refit manifest, published
engine hash, and probe report.

### Related conversion traps

- FP16 passed throughput tests but failed the trained-model value-error acceptance gate.
- Integer gather buffers for the from-to policy head were once cast to BF16, corrupting indices that cannot be
  represented exactly; only floating model state may be converted.
- A projected native module imported successfully but lacked TensorRT support. A valid preflight must perform a real
  refit and native inference, not merely import the extension.
- Pre-fold and post-fold quantized graphs require separate templates because batch-normalization folding changes the
  graph structure.

### Sources

- [Equal-scale TensorRT refit failure and correction](../../benchmarks/int8-template-staleness-rtx4070super-20260921/README.md)
- [Native inference precision and quality gates](../../benchmarks/cnn-inference-throughput-rtx4070s-20260827/README.md)
- [TensorRT source/runtime audit](../../analysis/v35-v42-regression-audit-20260913.md)
- [Publication boundary](../../system/inference-and-evaluation.md)
- [TensorRT publisher](../../../py/tools/publish_tensorrt_engine.py)

## Transferable failure study: `torch.compile` was not one result

### Findings

`torch.compile` was tested at two different boundaries and produced different answers:

- In a direct-policy eager diagnostic, compilation improved PyTorch forwards by roughly 27–33%.
- The corrected serving comparison measured the fused BF16 TorchScript artifact actually loaded by the C++ runtime
  and found it faster than the earlier compiled diagnostic.
- In an eight-rank convolutional training benchmark, compilation reduced throughput by about 18% relative to eager
  execution and emitted a DDP gradient-stride warning.
- In attention training, compiled automatic attention kernels improved throughput substantially. Backend rankings
  differed between eager and compiled execution.

The production inference runtime also depends on serialized TorchScript, frozen constant enumeration, in-place
weight adoption, and CUDA graphs that capture stable addresses. An Inductor `OptimizedModule` and code cache cannot
replace that boundary without redesigning hot swap and deployment.

### Decision and lesson

Compilation is not a global feature flag. It must be evaluated per model family, batch shape, hardware, training or
inference boundary, and actual deployed artifact. The project did not retain it for native search inference and does
not enable it in the final convolutional trainer configuration. For serving, TensorRT later superseded the historical
TorchScript comparison as the final compiler. This does not contradict compilation's benefit in an attention training
microbenchmark.

### Sources

- [`torch.compile` eager inference diagnostic](../../benchmarks/chess-direct-policy-inference-rtx4070s-20260818/README.md)
- [Corrected fused TorchScript comparison](../../benchmarks/chess-direct-policy-kernel-controls-rtx4070s-20260818/README.md)
- [Convolutional DDP training comparison](../../benchmarks/chess-training-throughput-rtx3060-20260812/README.md)
- [Attention training comparison](../../benchmarks/chess-attention-training-rtx4070s-20260818/README.md)

## Publication checklist for evaluation claims

Before a numerical result enters the report, record:

1. checkpoint and deployment-artifact hashes;
2. source revision and resolved configuration hash;
3. candidate selector, visits, parallel leaves, inference backend, precision, and batch topology;
4. opponent binary hash, version, nodes, threads, hash memory, and any tablebase or pondering settings;
5. opening manifest/hash, selection rule, pair count, colour swap, and match seed;
6. maximum plies, termination counts, and treatment of capped games;
7. W/D/L, raw score, paired-opening interval, Elo transform, rung anchors, and the limits of that calibration;
8. hardware and wall time, distinguishing saturated service throughput from isolated latency;
9. whether the result measures strength, target fidelity, kernel throughput, search throughput, accepted-data rate,
   optimizer cadence, or learning cadence;
10. every known confound, failed gate, unavailable raw artifact, and non-isolated component.

Final checkpoint selection, policy-only strength, fixed- and deep-search results, distillation results, the
cross-generation progression figure, and selected-checkpoint training dynamics and replay counters are complete.
Wider node-wide self-play/search rates remain distinct from those coordinator counters. The report deliberately
omits total project spend rather than constructing it from incomplete billing records.
