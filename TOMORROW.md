# Tomorrow: post-9x9 review

Run this review only after the first four-hour 9x9 experiments are terminal and their late-training data, checkpoints, evaluations, TensorBoard logs, and replay stores have been preserved. Use the 8x96 SE run as the same-time baseline and compare the 6x64, 10x128, and 8x96 global-pooling runs against it.

Do not transfer thresholds or conclusions from 7x7. Prefer samples produced after at least two hours of training, compare equal wall-clock windows, and keep strength, learning quality, and throughput as separate measurements.

## Establish the 9x9 baseline

- Confirm clean completion, evaluation coverage, source revision, and intact late replay data for every run.
- Compare fixed-dataset accuracy and loss, KataGo results, same-time-baseline matches, self-play positions and games per hour, inference queries per second, optimizer steps, replay turnover, and GPU utilization.
- Inspect the new TensorBoard distributions: policy loss, WDL loss, target top-1/top-2/top-3 mass, target and predicted policy entropy, root/terminal/predicted values, value error, sample weights, replay age, and auxiliary-target error.
- Select the architecture that gives the best elapsed-time training result. Treat fixed-visit match strength as quality per search, not strength per second; retain throughput when judging the larger model.
- Check whether 9x9 still has a strong first-player advantage and an unusually easy value target before using it as a chess proxy.

## Adaptive search termination

First decide whether final 9x9 search targets are concentrated enough to justify deeper instrumentation.

- From late replay samples, measure top-1, cumulative top-2 and top-3 policy mass, entropy, effective candidate count, and the gap between the first and second moves. Report distributions by ply range and model generation.
- Estimate the fraction of positions ending with a clearly dominant move, for example top-1 mass above 0.70, 0.80, and 0.90. Also measure diffuse positions; a decisive root value is not evidence that the move choice is settled.
- Do not infer an exact safe stopping point from the final target alone. A final sharp target can still have changed late in the search, while a final diffuse target generally rules early stopping out.
- If only a small fraction of positions are sharp, defer the feature. If the fraction is material, add bounded trace instrumentation in a later change for selected full searches at intermediate visit counts such as 32, 64, 96, 128, 192, and 256.
- With traces, test whether the leader, visit gap, and top-k mass remain stable, and calculate target divergence and simulations saved. Prefer an exact unrecoverable visit-lead rule before heuristic value or Q thresholds.
- Any eventual experiment must log requested and completed simulations and compare equal elapsed time against spending the saved compute on ordinary self-play.

## Resignation

Calibrate resignation independently for 9x9; do not enable it from the 7x7 thresholds.

- Audit late completed games by threshold, model generation, and ply. For each possible threshold, record the first trigger, eventual result for the triggering player, remaining plies, and remaining search budget.
- Evaluate one, two, and three consecutive low-value observations on that player's own turns, with a minimum-ply gate. Alternating turns require separate persistence counters per player.
- Report candidate-game fraction, estimated searches saved, false-nonloss count, false-positive rate, and a Wilson upper confidence bound. Separate natural endings from maximum-ply adjudications.
- If the value head calibrates substantially during the run, prefer a generation-scheduled threshold or delay resignation until a minimum generation rather than using one aggressive threshold from initialization.
- Implement audit-only telemetry first. Active resignation requires a conservative threshold plus a permanent non-resigning holdout of roughly 5-10% so calibration remains observable and bad labels cannot become self-confirming.

## Replay duplication

Repeat the exact-state audit on every late 9x9 replay FIFO before implementing deduplication.

- Hash the complete canonical encoded state, including history planes. Do not symmetry-canonicalize the first audit.
- Report raw rows, unique states, duplicate excess, duplicate-group sizes, multiplicity distribution, and duplication by sample age and generation.
- Within duplicate groups, measure policy-target divergence, WDL disagreement, root-value spread, sample-weight differences, and whether duplicates came from the same or different generations.
- Estimate both consequences of deduplication: increased diversity and changed target weighting. Do not call duplicates redundant when their search targets differ.
- If duplicate excess is only a few percent, defer the implementation. If it is consistently material, first compare deduplication with unit multiplicity weight against no deduplication; test square-root multiplicity weighting separately rather than bundling both effects.
- Before implementation, decide whether replay capacity represents unique rows or raw occurrences, because that choice changes the effective history horizon. Any compaction must preserve FIFO ownership, auxiliary targets, sample credits, immutable training snapshots, and crash-safe atomic publication.

## Reanalysis

Decide whether replay targets are old enough, different enough, and long-lived enough to justify spending search compute on them.

- Use replay source generations and timestamps to report median, P90, P99, and maximum age, plus the wall-clock turnover time of the complete replay window.
- Compare the replay age with the rate of model improvement seen in same-time checkpoint matches and fixed-dataset predictions. A sample being several generations old is not important if adjacent models are nearly identical.
- On a small preserved trajectory sample only, reconstruct positions and search them with their source checkpoint and a current checkpoint. Measure policy divergence, root-value change, move-rank changes, and the cost per refreshed target.
- Compare that cost against producing the same number of fresh self-play searches. Reanalysis should be deferred if the replay turns over quickly, target drift is small, or fresh self-play produces more useful data per unit compute.
- Do not append refreshed rows for the real experiment: that mixes reanalysis with multiplicity and leaves stale targets active. A sound implementation requires reconstructible trajectory provenance, stable sample identities, and atomic newest-target overrides. Design it only after the audit shows a clear opportunity.

## Restart states and target diagnostics

- Recheck how often late 9x9 games produce restart-eligible positions under the current filters: absolute root value at most 0.3, the smallest two or three candidates covering at least 85% policy mass, and at least 15 remaining plies.
- Measure archive insertion rate, claim rate, exhausted positions, age/capacity eviction, fallback-to-initial-state rate, and whether restarted branches improve diversity rather than repeatedly reconstructing equivalent states.
- Inspect remaining-game-length target, prediction, and absolute-error distributions. The normalization scale is the configured 324 maximum plies; verify the head learns useful structure rather than merely predicting the dataset mean.
- Reassess FPU and forced playouts only from 9x9 strength, target-quality, and throughput results. Do not combine them with the selected architecture until their individual 9x9 effects are informative.

## Decision output

Produce one short report with the raw measurements, uncertainty, and one of three decisions for each topic: implement next, instrument first, or defer. Keep adaptive termination, resignation, deduplication, and reanalysis as separate decisions; none should be implemented merely because it helped or looked promising on 7x7.
