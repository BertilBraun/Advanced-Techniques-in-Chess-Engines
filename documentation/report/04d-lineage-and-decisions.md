# 4D. Lineage and decision chronology

Version numbers in this repository often mark recovery, resume, or infrastructure changes rather than independent
scientific experiments. This chronology groups them into decisions and records what kind of inference is justified.

## Baseline platform and four-day yardstick

The early project moved search and game ownership into C++, added batched direct inference, persistent DDP, credit
accounting, external-engine evaluation, and recoverable model refresh. Go 7x7 and 9x9 validated shared platform
contracts but did not receive a terminal campaign comparable to chess.

The August four-day chess run became the learning-efficiency yardstick. Later work compared wall-clock trajectory,
not merely final strength. Its evidence predates some modern archive rules and remains historical rather than current
operational authority.

## Recovery and the v7/v8 conversion failure

After platform changes, new chess runs learned more slowly and sometimes failed to convert winning positions. The
recovery plans generated many candidate explanations. Subsequent audits rejected some, narrowed others, and found
target-stream defects that were not visible in aggregate loss. This phase established an important practice: inspect
the actual replay population and completed trajectories before attributing a plateau to optimizer or architecture.

The central sources are the [post-four-day regression analysis](../plan/chess-post-four-day-regression-analysis-20260820.md),
[conversion investigation](../analysis/chess-conversion-investigation-20260826.md), and
[v8 data comparison](../analysis/v8-training-data-comparison-20260826.md).

## Search campaign v13–v30

Versions v13–v20 developed predicted budgets; v21–v30 developed learned stopping. Several run-killing defects were
found along the way, including duplicated configuration pins, an inverted isotonic projection, a poorly seeded dual,
unstable feature standardization, and a missing checkpoint-visits contract. Those failures do not erase the final
controlled tests, but they limit claims based on intermediate runs.

The decisive methodological advance was the byte-identical fork from a mature checkpoint and rebuilt replay. It
reduced paired noise enough to conclude that stopping worked mechanically but did not improve the critical path or
detectable strength. Adaptive search was then removed rather than left dormant.

## v29 and v34: strength versus wall clock

The v29 ladder series showed both a model-strength plateau and a collapsed generation rate. That distinction matters:
Elo per generation can look healthy while the system produces generations too slowly. A deep generation-936 match
then anchored mature-checkpoint strength under a larger search.

V34 supplied the last complete public result before the final run, plus a joined dynamics record of games, positions,
presentations, optimizer steps, schedules, model size, throughput, and strength. It also supplied the replay used for
compression, TensorRT, QAT, and optimizer screens. V34 is a verified predecessor, not a placeholder result to be
silently assigned to the final recipe.

## v35–v42: why short runs were not causal

The QAT transition initially appeared to regress learning. Audit showed that code, initialization, export, fold
state, backend, and replay all needed separation. The configured seed had not seeded network construction, so
apparently matched arms were not initialized identically. The controlled V35-code/V42-generation-zero record mainly
establishes a reproducible protocol; its README does not contain a completed efficacy result.

The [regression audit](../analysis/v35-v42-regression-audit-20260913.md) and
[executable bisect](../analysis/v35-v42-executable-bisect-20260913.md) should be read as causal hygiene, not as a
simple winner table.

## QAT, progressive serving, and V89–V93

Frozen-replay and native-backend studies led to scaled-post QAT, pre-fold deployment copies, TensorRT templates, and
Nesterov SGD. Production-topology controls then separated core backend gains from live replay throughput.

V89 is the intended final recipe lineage. V90–V93 include resume and correctness changes, particularly the
progressive candidate latch and TensorRT fidelity/refit incidents. The final report must reconstruct which source
revision and configuration governed each interval, what training state continued unchanged, what downtime or
discarded evidence occurred, and which checkpoint/inference artifact is selected. It should not report “V93” as if
it were an unrelated fresh training run.

## Decision-quality lessons

Across the lineage, the strongest reusable practices were:

- compare wall-clock and data volume, not generation number alone;
- fork shared state for small online effects;
- treat inference exports as part of model identity;
- separate proxy, throughput, and strength gates;
- preserve corrected conclusions and failed arms;
- freeze the exact result identity even when the living recipe continues to evolve.
