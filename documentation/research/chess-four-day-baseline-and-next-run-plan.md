# Chess four-day baseline and next-run plan

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

## 5. Architecture and throughput experiments

Benchmark both convolutional and attention-based families at three approximate capacities:

- 1 million parameters;
- 3.5 to 4 million parameters, matching the current network;
- 8 to 10 million parameters.

For every architecture, measure on the eight RTX 3060 node:

- training positions per second at global batch size 2,048;
- standalone inference throughput over relevant batch sizes;
- end-to-end self-play simulations per second with the production topology;
- GPU memory, host overhead, and compilation behavior;
- fixed-replay policy, WDL, and auxiliary validation quality after equal optimizer samples and equal wall time.

Lower attention throughput is acceptable if it yields enough additional learning per sample or per wall-clock hour.
The decision metric is expected final strength within a four-day wall-clock budget, not raw throughput alone.

The architecture pilot may train candidates on a frozen dataset, but it must not become multiple self-play training
runs. Use the frozen-data results, published chess evidence, and throughput measurements to select one architecture
family for the single experimental run.

## 6. Progressive model sizing

The tentative four-day schedule is:

| Wall time | Approximate capacity |
| --- | ---: |
| 0 to 12 hours | 1 million parameters |
| 12 to 48 hours | 3.5 to 4 million parameters |
| 48 to 96 hours | 8 to 10 million parameters |

The transition method is an open experiment, not an assumed weight morphism. KataGo trained the next larger network
concurrently on the same replay data and switched only when its average loss caught up to the smaller model. Its main
run switched from 6x96 to 10x128, 15x192, and 20x256 networks at approximately 0.75, 1.75, and 7.5 days. It did not
simply widen the active checkpoint in place.

Concurrent catch-up training consumes hardware that this project may need for self-play. Compare these transition
options cheaply:

- concurrent larger-model training on the same replay, following KataGo;
- sequential initialization from the smaller model where the architecture permits exact identity-preserving depth
  growth;
- fresh larger-model initialization trained on accumulated replay with a short warmup.

For any transition, require the larger model to catch up on fixed replay loss and a small match before it replaces the
self-play model. This is a transition criterion, not permanent generation-by-generation gating. Keep one architecture
family throughout a run; do not transition from a CNN body to a transformer body mid-run unless a separate experiment
demonstrates a reliable transfer method.

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

## 10. Transposition audit and graph-search decision - rejected

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

Inference caching is a separate question because it can reuse identical neural-network inputs without merging rule
state, visits, values, or descendants. Any cache work must begin with measurement-only instrumentation and its own
acceptance decision; it is not a continuation of Monte-Carlo graph search.

Reference: [Monte-Carlo Graph Search for AlphaZero](https://arxiv.org/abs/2012.11045).

## 11. Explicit exclusions

- No checkpoint averaging.
- No model gating.
- No second baseline repetition.
- No multiple multi-day auxiliary-target ablations.
- No always-on tablebase use before the late game cap.
- No transposition implementation before measuring reuse.
- No assumption that transformer throughput will match CNN throughput.

## 12. Authorization gate for the experimental run

Before proposing the final experimental configuration, complete and review:

1. the frozen baseline archive and final Elo/search ladder;
2. long-game and shuffling analysis from the evaluation corpus;
3. the late-cap and tablebase cutoff decision;
4. CNN-versus-attention throughput and frozen-data learning benchmarks;
5. a reliable progressive-size transition test;
6. the adaptive-search audit;
7. the transposition reuse audit;
8. the completed moves-left screening, which rejected remaining-length search utility for the planned run.

Select one coherent package for the single four-day experimental run. Any mechanism that has not passed its cheap
screen remains out of that run.
