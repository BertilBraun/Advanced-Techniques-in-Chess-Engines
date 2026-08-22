# Chess four-day baseline and next-run plan

> Superseded 2026-08-20 by `documentation/plan/chess-recovery-plan-20260820.md`. The run this document plans
> was not launched. Retained for its baseline-freeze procedure, strength-calibration method and
> rejected-alternatives record. Do not take configuration values or acceptance criteria from this document.

## Status and constraint

This document records the agreed research plan. It does not authorize a run, implementation, evaluation, or change
to the active training process. The current run is the four-day baseline. There is budget for at most one further
four-day experimental run, so every proposed change must first pass a cheaper audit, benchmark, or evaluation.

Checkpoint averaging and model gating are out of scope. The next run will publish each completed model directly,
as the current run does.

## 1. Freeze the four-day baseline

Stop cleanly after the first completed generation at or beyond four days and preserve the following locally:

- native and inference model checkpoints;
- optimizer, scheduler, generation, and random-state checkpoint;
- resignation calibration state;
- effective configuration and source revision;
- TensorBoard events and the composed R3-to-R4 timeline;
- run logs, evaluation results, and exact totals for optimizer steps, generated games, admitted positions, wall time,
  and GPU-hours;
- hashes and a small machine-readable artifact manifest.

The full replay payload is not part of the required local archive. Before discarding it, measure its physical size.
Preserve the replay metadata, logical contents summary, age and generation distributions, and a representative
sample. Retain the full replay only if its measured size is acceptable or if an exact continuation branch is later
required. The model, optimizer, configuration, and run evidence are sufficient for the frozen paper result; the full
replay is necessary only to reproduce the exact next optimizer batches or continue with the same data state.

## 2. Establish playing strength

Evaluate the frozen model at these candidate modes:

- policy only;
- 64 searches per move;
- 1,000 searches per move;
- 10,000 searches per move;
- 50 milliseconds per move;
- 1 second per move;
- 5 seconds per move.

Use the existing calibrated Stockfish fixed-node ladder. For each candidate mode, first run a small exploratory sweep
over the ladder. Select the rung closest to a 50% candidate score, then run 400 games against that rung using paired
openings and reversed colors. Report WDL, score, paired confidence interval, descriptive Elo difference, and the
approximate calibrated Elo. The calibration uncertainty and dependence on hardware, openings, and time control must
remain explicit.

### Observed baseline results

The completed exploratory sweeps used eight calibrated Stockfish 13 rungs with 10 games per rung. Brackets below are
match-sampling intervals from the descriptive ladder fit; they exclude uncertainty in the approximate Stockfish-to-Elo
calibration. The day-four confirmations used 200 games at the selected rung rather than the planned 400. Policy-only
and 50-millisecond modes were omitted by decision.

| Candidate mode | Day-three ladder Elo | Day-four ladder Elo | Day-four 200-game confirmation |
| --- | ---: | ---: | --- |
| 64 searches | 1,949 [1,843, 2,027] | 1,915 [1,868, 2,005] | vs. 1,900: 95-41-64, 57.75% [51.75%, 63.50%]; approximately 1,954 Elo [1,912, 1,996] |
| 1,000 searches | 2,522 [2,400, 2,646] | 2,461 [2,390, 2,543] | vs. 2,500: 72-87-41, 57.75% [52.75%, 62.75%]; approximately 2,554 Elo [2,519, 2,591] |
| 10,000 searches | 2,806 [2,709, 2,908] | 2,720 [2,636, 2,817] | vs. 2,700: 96-75-29, 66.75% [61.50%, 72.00%]; approximately 2,821 Elo [2,781, 2,864] |
| 1 second | 2,584 [2,512, 2,646] | 2,678 [2,636, 2,709] | vs. 2,700: 44-109-47, 49.25% [44.25%, 54.25%]; approximately 2,695 Elo [2,660, 2,730] |
| 5 seconds | 2,688 [2,564, 2,806] | 2,763 [2,657, 2,862] | Unavailable; the compute node was lost before a completed result was preserved. |

The confirmation intervals likewise describe paired match sampling only. Timed results are specific to the evaluated
hardware and search implementation; neither table column should be read as a calibration-uncertainty interval.

The final-model evaluation must not reuse the generation-445 result as its strength estimate, although that result
can serve as a historical comparison. Timed modes must run on an otherwise idle GPU allocation and record actual
latency, inference batch behavior, and achieved simulations per move.

Persist completed games and per-move telemetry from all ladder and confirmatory matches. At minimum, record the ply,
position identity, network WDL, root WDL/Q, root visits, post-pruning visit distribution, policy entropy, chosen move,
halfmove clock, repetition count, material, and terminal reason.

## 3. Explain long games from the evaluation corpus

Use the evaluation games across all search and time budgets to determine whether extra plies are removable shuffling
or necessary play. Report game-length mean, median, P90, P95, and maximum, split by candidate mode, outcome, terminal
reason, and opponent rung.

Classify long tails using reproducible signals:

- repeated full game states;
- reversible non-pawn and non-capture sequences;
- plies since the last irreversible move;
- material and pawn-structure changes;
- root-value and chosen-move-value changes during apparently idle sequences;
- conversion from strongly winning positions;
- eventual resignation, rule terminal, ply cap, tablebase result, or material adjudication.

Estimate how many plies can be removed by deleting exact cycles and how many low-progress plies remain without an
exact repetition. Do not introduce a shorter-game objective unless the analysis demonstrates a meaningful population
of removable play.

The current forced-fast tail already excludes approximately the final 40 plies of a capped 200-ply game from policy
training. Measure the actual training-position distribution by ply and phase before further downsampling endgames.
If late positions are overrepresented or strongly correlated, prefer stratified per-game or per-phase sampling over
blanket removal.

## 4. Late-cap tablebase adjudication

The candidate next-run behavior is deliberately narrow:

1. Play normally before the configured maximum game ply.
2. At the maximum only, probe the optional game-specific terminal oracle.
3. If the chess position is covered by Syzygy, assign its exact WDL as the game result.
4. Otherwise use the configured material fallback.
5. Do not derive a policy target from the tablebase and do not use it to guide ordinary pre-cap search.

This improves value labels for already excessive games without suppressing ordinary seven-piece endgame experience
before the cap. The generic platform should own only an optional terminal-oracle contract; chess owns the Syzygy
implementation.

Retain **200 plies** as the recommendation. Current games are about 100 plies on average and P90 is about 174, while
raising the cap beyond 200 correlated with a pronounced training deterioration. The existing evidence distinguishes
the candidates as follows:

| Cap | Existing-evidence assessment |
| ---: | --- |
| 180 | Only six plies beyond the observed P90, so it would truncate a material part of the ordinary long tail. |
| 200 | Twenty-six plies beyond P90, preserves the established late-game tail, and avoids the previously harmful longer-cap regime. |
| 220 | Adds ten percent more capped-game search than 200 without current coverage or strength evidence that those plies improve labels. |
| 300 | Re-enters the long-game regime associated with deterioration and spends 100 extra plies before applying the same safety adjudication. |

The external references are deliberately treated as design context rather than transferable numeric tuning. AlphaZero
used a remote 512-step draw cutoff, scored Go at its separate 722-step cutoff, and used no endgame tablebases
([supplementary methods](https://arxiv.org/pdf/1712.01815)). Current Lc0 self-play has a 450-ply safety cutoff and
can expose Syzygy to its search
([official source](https://github.com/LeelaChessZero/lc0/blob/master/src/selfplay/game.cc)); that always-available
search behavior is explicitly not adopted here. KataGo plays games to completion, reduces visits after sustained
low win rate, stochastically downweights those late samples, and uses Go-specific mechanisms to finish games sooner
([paper, Appendix D](https://arxiv.org/abs/1902.10565)). Together these sources support having a safety mechanism and
preserving natural endgame experience, but none justifies increasing this project's empirically problematic cap.

Before authorizing a run, still compute exact historical counterfactual coverage at 180, 200, 220, and 300 from the
preserved corpus. That audit may reject 200 if it reveals unexpected label coverage, but absent such contrary evidence
200 is the coherent choice. Do not choose 150 without evidence because it would truncate a nontrivial part of the
ordinary distribution.

## 5. Architecture and throughput decision

The previous Chess network had 3,578,232 trainable parameters: 2,603,720 in the shared backbone and 974,512 in the
output heads. Its two dense policy heads alone contained 967,232 parameters. The next run deliberately makes a clean
break from that representation: Chess now emits direct 76-plane policy logits, normalizes only over legal moves, and
has no dense-policy compatibility or checkpoint migration path.

Controlled RTX 4070 SUPER measurements found no direct-policy throughput regression. A direct-policy 5x256 attention
control reached 61.0k positions/s with automatic dispatch and 63.5k with memory-efficient SDPA at batch 64, compared
with approximately 59.8k positions/s for the historical dense-policy control. The direct-policy CNN control was also
effectively unchanged at 36.2k positions/s versus approximately 35.9k historically. The lower throughput of the largest progressive
model is explained by its 15 sequential attention blocks, not by the policy rework. The complete controlled comparison
is recorded in [the kernel-control benchmark](../../benchmarks/chess-direct-policy-kernel-controls-rtx4070s-20260818/README.md).

The selected production-equivalent progressive models measured as follows at batch 64, BF16, fused TorchScript, and
memory-efficient SDPA:

| Stage | Backbone | Training parameters | Inference parameters | Positions/s |
| --- | ---: | ---: | ---: | ---: |
| `chess-attention-500k` | 6x96, 3 heads, FFN 192 | 474,754 | 467,219 | 53,746.6 |
| `chess-attention-2m` | 10x160, 5 heads, FFN 320 | 2,104,642 | 2,092,179 | 36,706.3 |
| `chess-attention-4m5` | 15x192, 6 heads, FFN 384 | 4,500,898 | 4,485,971 | 26,126.9 |

The exact final run and environment are recorded in
[the final progressive benchmark](../../benchmarks/chess-direct-policy-final-progressive-rtx4070s-20260818/README.md).

## 6. Progressive model sizing

The production configuration uses exactly three independently trained stages:

| Start time | Model |
| --- | --- |
| 0 days | `chess-attention-500k` (6x96) |
| 0.75 days | `chess-attention-2m` (10x160) |
| 2.0 days | `chess-attention-4m5` (15x192) |

Each candidate starts from random initialization and trains from the shared replay stream; weights are not transferred
between model sizes. The next eligible candidate is promoted only after its loss EMA beats the active model by the
configured threshold for the required consecutive training quanta. Only the active checkpoint is published to
self-play. This preserves the early update rate of the small network, moves to a model near the useful capacity of the
last run without spending parameters on dense heads, and reserves the deeper 4.5M model for the latter half of the run.

The configured promotion rule uses loss EMA decay 0.8, ten warmup quanta, and a maximum candidate-to-active relative
loss of 1.01. Evaluation matches remain independent strength evidence rather than blocking the automated promotion.
All three stages use the same attention architecture family.

Reference: [Accelerating Self-Play Learning in Go](https://arxiv.org/abs/1902.10565).

## 7. Existing and candidate auxiliary targets

Remaining game length and opponent next-policy prediction are already implemented and configured. They are training
heads only: the current inference export strips auxiliary heads, so search cannot consume remaining-length predictions
without extending the inference contract and native search parameters.

The next targets worth investigating are:

- short-horizon values, predicting outcomes or searched values at several future horizons;
- search-improvement uncertainty, predicting whether additional simulations will materially change the root target;
- plies until irreversible progress, only if the long-game analysis shows reversible shuffling that total remaining
  length does not distinguish.

Short-horizon values are the leading candidate because they may provide denser value supervision and features for
adaptive search. A 3,200-search teacher target for every training position is too expensive. Search-improvement
uncertainty must therefore use sparse audit positions, naturally available full searches, or a cheaper online signal.

Plies-until-progress is only a predictor. It cannot discourage shuffling unless search consumes it through a bounded
utility term or it supports another stopping/control decision. Do not add it merely because the label is easy to
compute.

## 8. Moves-left search use

The current policy target is the root visit distribution. A remaining-length prediction changes the learned policy at
the current position only if it affects PUCT visits; using it solely to choose the final move changes later trajectories
but not the current visit target.

There is no budget for multiple multi-day moves-left runs. Use the trained baseline head for a cheap screening test
after exposing it through inference:

- baseline search with no length term;
- final-root selection with a bounded length preference;
- PUCT exploitation with a bounded, decisive-value-gated length preference.

Run evaluation matches and measure Elo, decisive-game length, repetitions, and conversion failures. This screening
can reject obviously harmful formulations but cannot prove the self-play learning benefit. If the mechanism is chosen
for the one experimental run, its actual contribution must be reported as part of that run and interpreted without a
full multi-day ablation.

### Screening result and decision

The cheap screen completed on 2026-08-18 with the frozen R4 generation-624 checkpoint. Each mode played 50 games as
25 paired openings with colors reversed, using exactly 64 searches per candidate move against Stockfish skill level 4.
All modes reused the same openings and seeds.

| Mode | W-D-L | Score | Decisive plies mean / median / P90 | Candidate-win plies mean / median / P90 |
| --- | ---: | ---: | ---: | ---: |
| No length term | 50-0-0 | 1.00 | 98.9 / 96 / 137 | 98.9 / 96 / 137 |
| Final-root preference | 48-1-1 | 0.97 | 96.9 / 96 / 138 | 97.0 / 96 / 138 |
| Decisive-gated PUCT preference | 46-1-3 | 0.93 | 95.8 / 93 / 124 | 98.0 / 94 / 124 |

Final-root selection improved none of the 50 matched opening/color games and worsened two; PUCT improved none and
worsened four. Final-root selection overrode the ordinary visit leader ten times. The modest changes in mean game
length are not a useful conversion gain: the final-root median and tail did not improve consistently, and the shorter
PUCT decisive-game statistic includes newly introduced losses. There were no maximum-ply adjudications.

Final-root-only use also does not meet the learning objective. It changes later trajectories, but it does not alter the
current root visit distribution and therefore does not teach the policy the length preference at that decision. PUCT
does alter the policy target, but the tested bounded additive formulation sacrificed playing strength. The existing
unconditional remaining-game-length head predicts duration under the observed policy rather than action-conditional
time to safe conversion, so a decisive absolute value gate does not make candidate moves WDL-equivalent.

Do not include remaining-game-length search utility, final-root length selection, or a game-length-reduction objective
based on this head in the next multi-day run. Reconsider the topic only after a separate sibling-action calibration
shows that an outcome-conditioned target can rank conversion time among genuinely WDL-equivalent moves. Such a
redesign is new research, not unfinished work for the planned run.

## 9. Adaptive search budget

The initial design is an online progressive search, not a fixed-position learned classifier:

1. Run 400 simulations.
2. Stop if the post-pruning root visits have a dominant, stable move and no competitive second move.
3. Otherwise continue to 800 and then 1,200 or 1,600 simulations while checking root-policy and root-value stability.
4. Permit 3,200 simulations only for positions that remain unresolved and also carry a strong difficulty signal.

Candidate signals include top-one visit share, top-one versus top-two margin, visit-distribution change between
checkpoints, Q convergence, policy entropy, network-versus-search disagreement, and short-horizon-value disagreement.
No single threshold such as 70% visits is sufficient across all positions. Dirichlet noise and forced playouts distort
raw visits, so stopping and stability decisions should use the pruned decision distribution where applicable.

The implemented learned signal predicts the larger of policy correction and value correction. Policy correction is
the total-variation distance between the clean pre-noise network prior and the final post-pruning searched policy;
value correction is half the absolute difference between final root Q and the network WDL scalar. Use `0.40` as the
provisional minimum prediction for unlocking the tail. This is a semantics- and risk-based default: a predicted 0.40
already represents a material correction, while allowing a weak head centered near sigmoid 0.50 to fail safely toward
more search. It is not calibrated by the random-model mechanics audit and must be revisited with a trained head.

Calibrate the rule on positions drawn from early, middle, and late checkpoints rather than only the final policy. Run
each audit position to the maximum budget and ask at which earlier checkpoint its selected move, visit target, and root
value had effectively stabilized. The audit set validates an online rule; it is not assumed to be a stationary training
distribution.

The experiment must report:

- mean and distribution of simulations per full-search position;
- agreement with the maximum-budget selected move;
- visit-policy divergence and root-value error;
- total self-play throughput;
- strength at equal average search compute.

Reference: [Learning to Stop: Dynamic Simulation Monte-Carlo Tree Search](https://arxiv.org/abs/2012.07910).

## 10. Transposition, graph-search, and inference-cache decision - rejected

The full Monte-Carlo graph-search experiment is complete and will not proceed to production. The separate graph path
implemented shared nodes and descendant statistics, parent-local edge statistics, transposition correction, virtual
loss, cycle rejection, graph-aware rerooting and pruning, and exact chess rule/history identity. Tree search remained
the control and default throughout.

Corrected sustained RTX 4070 SUPER measurements still showed an 8.63% throughput loss at 1,000 searches and an 8.28%
loss at 10,000 searches. Only 0.0249% and 0.1769% of inference evaluations, respectively, were avoidable, while the
hypothetical unfolded tree contained only 0.0814% and 0.5007% more node instances. Earlier 30,000- and 60,000-search
tests with 64 parallel searches found equality-verified transposition-table hit rates of 2.37% and 3.46%, but graph
throughput remained 7.06% and 5.76% lower. Exact repetition semantics exclude most apparent move-order
transpositions because their retained histories remain relevant to future threefold adjudication.

The result does not justify the implementation and maintenance cost. Do not merge the graph-search branch, add an
evaluation-only graph mode, or continue graph-search tuning unless this decision is explicitly reopened based on new
evidence. Tree search is the production search structure.

Inference caching was evaluated separately because identical encoded neural-network inputs can be reused without
merging rule state, visits, values, descendants, or history-distinct search nodes. Measurement-only instrumentation
tracked exact encoded-input equivalence while continuing to evaluate every position normally. In the production-like
mixed workload of 25% 800-search requests and 75% 150-search requests, the ideal unbounded-cache upper bound was only
3.5485% with one parallel search and 3.5295% with two. Across approximately 1.83 million inference positions per arm,
same-batch reuse was zero and one evaluation, respectively. The rate rose only to approximately 4% after following
retained trees for six moves.

This opportunity is borderline unusable as a production optimization. It precedes finite-capacity misses and the
costs of lookup, synchronization, output storage, device transfer, and eviction, while the unbounded measurement
tracker itself showed approximately 3.65% throughput overhead in a separate control comparison and grew without
bound. Do not implement an inference cache or merge the instrumentation branch. Preserve
`codex/inference-cache-hit-rate` as rejection evidence; its committed report and raw evidence are at
`ade6c5ba`. Reopen the decision only if a materially different workload demonstrates substantially greater exact
encoded-input reuse.

Reference: [Monte-Carlo Graph Search for AlphaZero](https://arxiv.org/abs/2012.11045).

## 11. Explicit exclusions

- No checkpoint averaging.
- No model gating.
- No second baseline repetition.
- No multiple multi-day auxiliary-target ablations.
- No always-on tablebase use before the late game cap.
- No graph search, transposition reuse, or inference cache; the completed audits rejected all three for the current
  workload.
- No assumption that transformer throughput will match CNN throughput.

## 12. Authorization gate for the experimental run

Before proposing the final experimental configuration, complete and review:

1. the frozen baseline archive and final Elo/search ladder;
2. long-game and shuffling analysis from the evaluation corpus;
3. the late-cap and tablebase cutoff decision;
4. CNN-versus-attention throughput and frozen-data learning benchmarks;
5. a reliable progressive-size transition test;
6. the adaptive-search audit;
7. the completed transposition and inference-repetition audits, which rejected graph search and inference caching;
8. the completed moves-left screening, which rejected remaining-length search utility for the planned run.

Select one coherent package for the single four-day experimental run. Any mechanism that has not passed its cheap
screen remains out of that run.
