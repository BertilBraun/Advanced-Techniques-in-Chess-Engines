# 4B. Data, replay, and curriculum investigations

The final model is inseparable from the data system that trained it. This chapter distinguishes changes to which
positions are generated, which stored positions are sampled, and how often they are presented. The complete status
ledger is [Data generation and replay experiments](../experiments/data-and-replay.md).

## Replay reuse is also a clock

Replay ratio is training presentations divided by admitted fresh positions. In a credit-funded runtime it determines
how quickly a generation completes, so it also changes every generation-indexed visit, learning-rate, replay-size,
temperature, and evaluation schedule in wall-clock time. Earlier reuse comparisons therefore changed more than
sample efficiency.

The final value of 4 is retained, but there is no clean final-lineage online optimum. V39 contains useful throughput
arithmetic, while v34 dynamics show why high presentations without enough fresh target diversity can have diminishing
returns. Final reporting must include both configured reuse and empirical presentations per admitted unique row.

## Growing replay

Capacity grows from 600,000 to 20 million rows rather than allocating the terminal window immediately. Early in a
run, a huge nominal capacity contains no extra information; later, a wider window preserves strategic variety and
reduces concentration on a narrow recent policy. The exact staged schedule was not compared with a fixed-window
control. It is a motivated design (**R/M**, supported by observational **O**) rather than a measured multiplier.

Replay reporting should include occupancy, age percentiles, unique rows presented, wrap behavior, and age by sample
probability. A raw capacity number is insufficient.

## Sampling and sample weight

Policy-surprise sampling emphasizes rows where search disagreed with the network prior while reserving 30% uniform
probability and capping surprise. It changes draw probability. Typed sample weights are a different objective input;
ordinary final-recipe rows have primary weight 1.0. The report must not describe surprise sampling as proof that
arbitrary loss weighting worked.

TD-error prioritized replay, recency weighting, and deduplication were proposed but not established. Literature
motivates surprise weighting; no isolated online final-lineage ablation assigns it Elo credit.

## Random openings and restart states

Half of games begin after zero to eight uniformly sampled legal plies. The other half begin from recent archived
states filtered by value, remaining length, age, and branchable visit mass. Restart selection includes a uniform
component and a difficult-state preference. This is not the learned regret network in RGSC; it is the project's
typed restart heuristic.

The mechanism broadens data generation in two ways: shallow openings diversify early trajectories, while restarts
spend complete games on positions the current policy found consequential. Both are retained but were not isolated in
one-variable Elo arms.

## Endgame conversion and target poisoning

Runs v7/v8 often reached overwhelmingly won positions and hit the ply cap. The causal audit separated several
hypotheses that initially looked equivalent. Tablebase removal and remaining-length censoring did not explain the
difference. Resignation rate alone did not distinguish the runs. The strongest data finding was an early window in
which forced fast continuation excluded the last plies of long games from primary policy training while shallow
cutoff values were propagated across trajectories.

That window later aged out of replay, but weight damage could persist. The final design eliminates the forced-fast
tail, lengthens game caps over time, censors unknown remaining-length labels, and uses the search-root value for a cut
game. The [conversion investigation](../analysis/chess-conversion-investigation-20260826.md),
[sample-stream comparison](../analysis/v8-training-data-comparison-20260826.md), and
[cut-value benchmark](../benchmarks/cut-game-value-target-rtx4070super-20260825/README.md) together provide
observational **O**, diagnostic **P**, and mechanics **M** evidence—not a clean factorial ablation.

## Calibrated resignation

Production resignation begins only after enough generations for calibration. It chooses a threshold using recent
triggered games under a false-nonloss upper confidence bound, relaxes gradually, and permanently continues 20% of
triggered games. The early canary established persistence and audit mechanics with an intentionally aggressive
threshold. It did not prove final safety.

The terminal archive must report threshold trajectory, trigger counts, continuation outcomes, false non-losses,
saved plies/search, and any interaction with restart games. Until then the claim is “calibrated and audited,” not
“zero false resignations.”

## Auxiliary target materialization

The next-policy target depends on a later observation, so a cheap or missing future search can affect an otherwise
valid primary row. Remaining game length becomes unknown when a game is cut and is explicitly censored. These
eligibility rules are part of replay semantics, not training-loop conveniences. Overfit and sample audits establish
that the targets are trainable and wired correctly; they do not isolate Elo.

## Reanalysis and publication freshness

A bounded synchronous reanalysis path existed in the older v10 replay design. It used materialized override sidecars
and was removed when replay ownership changed. No controlled current-pipeline efficacy experiment exists. Reanalysis
is therefore superseded infrastructure, not a negative result.

A historical attempt to publish every 100 optimizer steps was rejected because publication cost outweighed expected
freshness. This is a local engineering result, not evidence that more frequent publication can never help.

## Overlap is not full asynchrony

The final topology pauses two of four self-play processes per GPU while all GPUs train. Other actors continue, so
self-play and optimization overlap. Credit quanta, checkpoint publication, and actor refresh remain coordinated.
The pause sweep showed that the best fraction changed with visit regime, supporting a configured topology rather
than a universal 50% rule. Fully asynchronous learning was proposed but not implemented.

## Replay infrastructure as evidence

The fixed-layout columnar store, parallel materialization, atomic claims, quarantine path, rejection-rate ceiling,
prefetch, and credit ledger are systems contributions. Loader and credit benchmarks provide **T/M** evidence. They
do not show stronger chess directly, but they prevent corrupted or stalled ingestion from masquerading as a learning
plateau.
