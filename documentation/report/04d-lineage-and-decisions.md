# 4D. Decision provenance and transferable incidents

Internal run numbers mostly identify restarts, recoveries, or infrastructure changes rather than independent
scientific experiments. They therefore remain in artifact paths and frozen provenance, not in the report's
explanatory vocabulary. This chapter groups the evidence by technical decision and preserves chronology only where
failure, diagnosis, and repair form a useful incident study.

## Baseline platform and wall-clock yardstick

The platform moved search and game ownership into C++, added batched direct inference, persistent distributed
training, credit accounting, external-engine evaluation, and recoverable model refresh. Small-board Go validated
shared platform contracts but did not receive a terminal campaign comparable to chess.

The previous four-day chess baseline became the learning-efficiency yardstick. Later work compared wall-clock
trajectory, not merely final strength. Its evidence predates some modern archive rules and remains historical rather
than current operational authority.

## Endgame conversion and target-stream audit

Two post-runtime-rework training attempts learned more slowly and repeatedly failed to convert overwhelmingly won
positions before the ply cap. Recovery plans generated many candidate explanations. The audit ruled out some,
narrowed others, and identified a target-stream defect invisible in aggregate loss: long-game tails were excluded
from primary policy training while shallow cutoff values were propagated too broadly.

Removing the affected rows did not immediately undo the learned damage. The repair combined longer caps, censored
unknown length targets, removal of the forced-fast tail, and searched root-value targets for cut games. The incident
established a reusable practice: inspect replay contents and completed trajectories before blaming the optimizer or
network. Sources include the
[post-baseline regression analysis](../plan/chess-post-four-day-regression-analysis-20260820.md),
[conversion investigation](../analysis/chess-conversion-investigation-20260826.md), and
[training-data comparison](../analysis/v8-training-data-comparison-20260826.md).

## Adaptive-search controller audit

The adaptive-search programme tested predicted budgets and learned in-search stopping. Intermediate failures included
duplicated configuration pins, an inverted isotonic projection, a poorly seeded dual variable, unstable feature
standardization, and a missing checkpoint-visits contract. Those failures are useful engineering evidence but do not
determine the final efficacy result.

The decisive comparison forked byte-identical model and replay state. It reduced online noise enough to show that
learned stopping worked mechanically but did not improve detectable strength or the training critical path. Adaptive
search was therefore removed rather than retained as dormant complexity.

## Wall-clock strength and data throughput

One mature training campaign demonstrated why Elo per optimizer boundary is insufficient: model strength plateaued
while the rate of completing boundaries also collapsed. A deep fixed-opponent match anchored the checkpoint's
strength under a larger search budget.

The later four-day baseline supplied the last complete public result before the final campaign and a joined dynamics
record of games, positions, presentations, optimizer steps, schedules, model size, throughput, and strength. Its
replay then supported compression, TensorRT, QAT, and optimizer studies. It remains a verified predecessor, not a
result that can be silently assigned to the final recipe.

## Deterministic initialization and causal hygiene

The transition to quantization-aware training initially appeared to regress learning. Audit showed that code,
initialization, export, fold state, backend, and replay all had to be separated. The configured seed had not reached
network construction, so apparently matched arms began from different tensors. The repaired controlled comparison
mainly establishes a reproducible protocol; it does not supply a completed efficacy result.

The [regression audit](../analysis/v35-v42-regression-audit-20260913.md) and
[executable bisect](../analysis/v35-v42-executable-bisect-20260913.md) should be read as causal hygiene, not as a
winner table.

## Quantized serving and progressive sizing

Frozen-replay and native-backend studies led to scaled-post-activation QAT, pre-fold deployment copies, TensorRT
templates, and Nesterov SGD. Production-topology controls separated core backend gains from live replay throughput.

The final campaign then exposed two controller failures. Indexing a fractional candidate multiplier by candidate
progress produced two quanta per boundary rather than the intended one/two alternation. More importantly, the
candidate's extra replay presentations made its training loss incomparable with the active model's loss; loss-based
promotion admitted a much weaker larger network. The controller now indexes the multiplier by the outer training
boundary and gates promotion on repeated head-to-head matches.

A separate function-preserving growth procedure widened and deepened the trained medium network, trained the added
capacity, and rebuilt its QAT state. The larger model reached parity but did not establish a stronger plateau. The
reported checkpoint is therefore the medium network. This supports a bounded conclusion that capacity was not the
immediate constraint under the tested recipe, not a general claim that larger models cannot help.

The frozen provenance must still identify the exact source revision, configuration, checkpoint, and inference
artifact for every accepted or discarded interval. Those identifiers belong in the reproducibility manifest rather
than the narrative.

## Decision-quality lessons

The strongest reusable practices were:

- compare wall-clock progress and fresh-data volume, not an internal boundary counter alone;
- fork shared state when testing small online effects;
- treat inference exports as part of model identity;
- separate proxy, throughput, and strength evidence;
- preserve corrected conclusions and failed arms;
- freeze the exact result identity even when the living recipe continues to evolve.
