# Second four-day chess run: adaptive search and auxiliary-target plan

Status: accepted design; implementation and production configuration are not yet authorized.

This document is the decision record for full-search allocation and auxiliary training targets in the second
four-day chess run. It supplements
[`chess-four-day-baseline-and-next-run-plan.md`](chess-four-day-baseline-and-next-run-plan.md). The run should aim
beyond 3000 calibrated Elo while remaining simple enough to implement, inspect, and validate before the single
remaining full training attempt.

## 1. Fixed decisions and exclusions

- Retained-root accounting remains unchanged. A configured search budget is a desired total root-visit count, and
  search adds approximately the difference between that target and the discounted retained visits already present.
  Small over- or under-runs caused by in-flight work are acceptable.
- Native parallel leaf searches are increased to four. Virtual loss remains responsible for separating simultaneous
  selections. Eight-way tree parallelism is rejected because it introduces more stale-selection and unwanted
  exploration than this run justifies.
- Fast searches remain simple: retain the tree and continue until the configured fast visit target. Fast-search
  positions do not create primary policy rows, although later observations may supply future auxiliary labels.
- Game-length selection utilities and moves-left PUCT terms remain rejected. Long games are not treated as a defect in
  search allocation.
- The 200-ply game cap and late-cap five-piece Syzygy adjudication remain in force.
- Adaptive search does not depend on graph search.
- The production model stages are the current approximately 0.5M, 2M, and 4.5M attention configurations. The older
  1M/4M/9M proposal is superseded.
- Replay reuse remains 10, with capacities 0.3M/0.6M/1.0M/1.5M/2.0M/2.5M beginning at generations
  0/25/50/100/250/400.

## 2. Progressive search schedule

The minimum adaptive full search is 400 total root visits. Maximum full-search visits and full-search probability are
staged independently:

| Generations | Full-search probability | Minimum visits | Maximum visits | Learned gate | Gate visit |
| ---: | ---: | ---: | ---: | :---: | ---: |
| 0-29 | 1.00 | 400 | 400 | Off | - |
| 30-49 | 0.50 | 400 | 500 | Off | - |
| 50-69 | 0.35 | 400 | 800 | On | 600 |
| 70-149 | 0.25 | 400 | 1000 | On | 700 |
| 150-249 | 0.25 | 400 | 1200 | On | 800 |
| 250-349 | 0.20 | 400 | 1600 | On | 1000 |
| 350-449 | 0.20 | 400 | 2400 | On | 1400 |
| 450+ | 0.20 | 400 | 3200 | On | 1800 |

For a gated stage, the gate visit is exactly the midpoint between the minimum and maximum:

```text
gate_visit = minimum_visits + (maximum_visits - minimum_visits) / 2
```

The first 30 generations deliberately use 100% fixed 400-visit full searches. They train the initial network and the
search-correction head without allowing an untrained correction prediction to affect search. Generations 30-49 add
only one optional 100-visit increment and still do not use the learned gate.

The existing fast-search schedule remains 50, then 100, then approximately 150 visits. The existing staged learning
rate remains a separate concern and should not be coupled to the search-control implementation.

## 3. Online full-search algorithm

### 3.1 Accounting

The executor continues to receive absolute total-root-visit targets. It calculates the simulations still needed from
the current discounted retained-root visits. No exact checkpoint barrier, retained-root special case, or correction
for a few parallel in-flight simulations is required.

### 3.2 Observation cadence

Every full search reaches at least 400 visits. Beginning at 400, the native executor evaluates the deterministic stop
rule every 100 completed root visits until the configured maximum. It retains only the small rolling state needed for
the previous 100- and 200-visit comparisons.

Checking after every simulation would add synchronization and noise without useful precision. Checking every 100
visits approximates continuous termination while keeping control flow and telemetry comprehensible.

### 3.3 Deterministic stop rule

Let `n` be the current root visits:

```text
progress = clamp((n - 400) / 1200, 0, 1)
minimum_top1_share = 0.70 - 0.20 * progress
minimum_margin = 0.45 - 0.30 * progress
```

The search is stable enough to stop when all of the following hold:

1. The post-pruning visit leader has not changed during the last 200 visits.
2. The absolute root-Q change over the last 100 visits is at most 0.04.
3. Either the top-1 post-pruning visit share is at least `minimum_top1_share`, or the top-1 minus top-2 share is at
   least `minimum_margin`.

After 1600 visits, the confidence thresholds remain at their floors of 0.50 top-1 share and 0.15 margin. The rule is
still evaluated every 100 visits through a possible 3200-visit maximum. Stable positions therefore never receive an
unchecked jump from 1600 to 3200.

The policy statistics use the same forced-playout-pruned view used for the stored primary policy target. Root
Dirichlet noise is applied once in the existing manner and requires no separate stopping correction.

Entropy, prior disagreement, root-network value disagreement, and full policy-distribution drift are useful
telemetry but are not additional deterministic gates in the first implementation.

### 3.4 Learned tail gate

The learned search-correction prediction is evaluated once per full search, at the configured gate visit:

1. If the deterministic rule says stop before or at the gate, stop.
2. If the deterministic rule still says continue at the gate, compare the root's predicted search correction with
   the configured tail-unlock threshold.
3. If the prediction is below the threshold, stop at the gate.
4. If it passes, unlock the remainder of the stage's maximum budget. Continue evaluating the deterministic rule every
   100 visits until it stops or the maximum is reached.

The prediction is constant for that root, so repeatedly applying the same learned threshold after the gate would add
no information. Only the deterministic evidence changes after the tail is unlocked.

The initial configuration exposes one `minimum_search_correction_to_unlock_tail` value. Its production value must be
calibrated after implementation from measured target, prediction, and search-stability distributions; it is not fixed
by this plan.

## 4. Search-correction target

### 4.1 Meaning and construction

Search correction measures how strongly the final full search corrects the raw network policy or value. It is built
from the final snapshot of every primary full-search row, regardless of the number of visits at which that search
stopped.

For legal action `a`:

```text
searched_policy(a) =
    post_pruning_visit_count(a) / total_post_pruning_visits

policy_correction =
    0.5 * sum(abs(searched_policy(a) - clean_network_prior(a)))

network_value = network_win_probability - network_loss_probability

value_correction =
    0.5 * abs(final_root_Q - network_value)

search_correction = max(policy_correction, value_correction)
```

The clean network prior is normalized over legal actions before root Dirichlet noise. Both correction components and
the combined label naturally lie in `[0, 1]`. Total variation is deliberately used instead of Jensen-Shannon
divergence because it is bounded, directly interpretable as moved probability mass, and requires no additional scale
hyperparameter.

Using variable final search budgets introduces a mild endogenous bias: deeper searches have more opportunity to
correct the network. The final result is nevertheless the most informative search already purchased by production.
Telemetry must split correction targets and prediction errors by final-visit bucket so that calibration can detect a
head that merely learns the adaptive budget distribution.

### 4.2 Training and inference contract

- Output: one sigmoid scalar in `[0, 1]`.
- Loss: Smooth L1.
- Loss weight: 0.10.
- Perspective: invariant under player-to-move canonicalization.
- Materialization: every primary full-search row has the target; no separate eligibility mask is necessary.
- Inference export: retained.
- Native consumption: the single tail-unlock decision described above.

All other auxiliary heads are stripped from inference artifacts. Search correction must remain exported because the
native search cannot use a prediction absent from the inference model. Its inference transport cost is one scalar per
evaluated position.

The current completed observation already contains final searched visits and final root Q, but not the clean root
prior or raw network root value. The smallest typed boundary addition is to return the final clean-prior/search-policy
total variation and network root value as scalar diagnostics. Python materialization then constructs the target. No
prior vector, checkpoint policy, or audit trace needs to cross the boundary.

## 5. Auxiliary target set

Only primary full-search positions become replay rows. Future observations, including fast searches, may provide
labels for those rows where specified.

| Target | Weight | Exported | Native search use |
| --- | ---: | :---: | --- |
| Next policy | 0.10 | No | None |
| Remaining game length | 0.05 | No | None |
| Four-ply future search value | 0.05 | No | None |
| Plies until irreversible progress | 0.025 | No | None |
| Legal-move prediction | 0.025 | No | None |
| Search correction | 0.10 | Yes | Tail unlock |

The total auxiliary weight is 0.35. Search correction has the same weight as next policy because it is the only
auxiliary prediction used operationally. The lower-weight heads are representation-learning regularizers. Loss scales
and gradient contributions must still be reported individually because equal numeric weights do not imply equal
gradient magnitude.

### 5.1 Next policy

- Label: the searched policy observation at ply `t + 1`.
- Eligibility: a following search observation exists.
- Perspective: action IDs are transformed into the current sample's canonical orientation.
- Loss: legal-action-masked policy cross-entropy.
- Storage: the existing sparse policy representation and mask.

### 5.2 Remaining game length

```text
target = remaining_game_plies / 200
```

- Eligibility: every completed-game primary row.
- Perspective: invariant.
- Loss: Smooth L1.
- Storage: one float.
- Search use: none; this target must not re-enter selection or PUCT utility.

### 5.3 Four-ply future search value

For a future search observation at exactly `t + 4`:

```text
target(t) = root_Q(t + 4)
```

Root Q is expressed from the player to move at its root. The general conversion back to the player at `t` is
`(-1)^offset * future_root_Q`; the four-ply offset is even, so no sign change is required.

If the game terminates before `t + 4`, there is still exactly one target for row `t`: the final scalar result from the
player at `t` perspective, with win `+1`, draw `0`, and loss `-1`. There are no imaginary post-terminal root values.
Across consecutive replay rows the terminal sign alternates naturally because each row uses its own player-to-move
perspective.

- Eligibility: a `t + 4` observation exists, or the completed game supplies a terminal result before it.
- Source: a future fast or full search is acceptable.
- Normalization: none; range `[-1, 1]`.
- Loss: Smooth L1 with beta approximately 0.1.
- Storage: one float plus eligibility if the trajectory is incomplete or malformed.

### 5.4 Plies until irreversible progress

An irreversible-progress event is a pawn move, capture, or castling-right change, matching the repository's
repetition-history reset semantics.

```text
target = min(plies_until_event, 16) / 16
```

- If no event occurs in the next 16 observed plies, the target is 1.0.
- If the game terminates before 16 plies without an event, the right-censored target is masked.
- Perspective: invariant.
- Loss: Smooth L1.
- Storage: one float plus eligibility.

Unlike plies since the last pawn move or capture, this future target is not reconstructed from the encoded rule-50
counter. It supplies trajectory information to the shared backbone and does not control search.

### 5.5 Legal-move prediction

The head has the same `76 x 8 x 8` action-plane shape as chess policy output. Each action is an independent binary
classification:

```text
target(a) = 1 if a is legal
            0 otherwise
```

Use sigmoid binary cross-entropy with balanced positive and negative components:

```text
loss = 0.5 * mean(BCE over legal actions)
     + 0.5 * mean(BCE over illegal actions)
```

This avoids the trivial all-illegal solution caused by the sparse positive class. Every replay row is eligible. Legal
action IDs already exist in the sparse policy payload, so the target requires no additional replay storage. The head
is representation supervision for move geometry, pins, checks, castling restrictions, and promotions; native move
generation remains authoritative.

## 6. Typed configuration and implementation ownership

Python owns one canonical typed self-play search configuration. Fixed and adaptive budgets are genuine variants in a
discriminated union rather than a mode plus nullable fields. The adaptive variant owns:

- minimum visits;
- staged maximum visits;
- 100-visit observation interval;
- 200-visit leader-stability window;
- root-Q tolerance;
- top-1 share and margin threshold formula;
- learned-gate activation generation;
- midpoint gate rule;
- calibrated tail-unlock threshold.

The resolved native configuration is passed once when a model generation becomes active. C++ owns live root state,
rolling checkpoint statistics, stop decisions, and final diagnostic calculation. Python owns generation schedules,
target materialization, loss construction, and reporting.

The inference output contract includes policy, WDL/value outputs, and search correction. Training-only auxiliary
outputs are absent from the exported artifact. Do not add Python callbacks to the native search loop or duplicate the
same semantic configuration across transport models.

## 7. Telemetry

Per full-search root, report at least:

- retained starting visits, final visits, and newly completed simulations;
- configured maximum and learned-gate visit;
- final stop reason: deterministic, learned gate, or maximum;
- top-1 share, top-1/top-2 margin, leader changes, and root-Q delta at each 100-visit observation;
- predicted and target search correction;
- policy-correction and value-correction components;
- final-visit bucket;
- whether the learned tail was offered, denied, or unlocked.

Aggregate by generation, model stage, game ply, final-visit bucket, and full/fast workload:

- mean, median, P90, P95, P99, and maximum full-search visits;
- proportion stopping at every 100-visit level;
- proportion reaching the gate and proportion unlocking the tail;
- search-correction target and calibration curves;
- full-search policy rows per hour;
- inference batch-size distribution and searches per second;
- percentage of total compute spent above the learned gate.

## 8. Pre-run validation

Calibration occurs after implementation but before production configuration approval.

1. Sample early-, middle-, and late-generation positions from the first run, including opening, middlegame, endgame,
   restart-root, decisive, and balanced positions.
2. Search each audit position continuously to 3200 visits and retain 100-visit snapshots.
3. Replay the deterministic rule and candidate learned thresholds offline without rerunning search.
4. Compare every proposed stopping point with the 3200 snapshot using selected-move agreement, policy total
   variation, root-value error, and leader stability.
5. Measure predicted search-correction calibration, recall for positions whose search remains unstable after the
   gate, and calibration by final-visit bucket.
6. Run a short production-shaped canary to measure average visits, tail batching, rows per hour, and throughput.
7. Run small paired matches against a fixed-search control at equal measured average search compute.

Initial acceptance targets are:

- at least 99% selected-move agreement with the 3200 reference overall;
- mean root-Q error at most 0.02;
- mean policy divergence at most 0.02 under the chosen reported distribution metric;
- late average full-search visits approximately 900-1100;
- no uncontrolled 2400/3200 tail;
- at least 90% of fixed-control self-play throughput at equal average compute;
- enough full-search policy rows per hour to sustain replay reuse 10 without avoidable starvation;
- no material strength loss in the equal-compute paired match.

Four-way parallel search is the production decision. Its batch sizes, policy divergence, and throughput should still
be reported by the canary, but a separate two-versus-four selection experiment is not a prerequisite.

## 9. Failure modes and controls

- **Premature deterministic stopping:** compare all candidate stopping points with continuous 3200 audit snapshots;
  retain selected-move, policy, and root-value acceptance gates.
- **Learned feedback loop:** the label comes from the final search produced by the adaptive policy, so report targets
  by final-visit bucket and retain deterministic evidence as an independent stop condition.
- **Uncalibrated early head:** keep the gate disabled through generation 49 and calibrate the unlock threshold before
  the production run.
- **Difficulty predicts budget rather than correction:** inspect conditional calibration at the same final-visit
  counts and the separate policy/value correction components.
- **Retained-root accounting errors:** preserve the current total-visit invariant and report starting, final, and new
  simulations separately.
- **Tail batch collapse:** use four parallel leaves, monitor inference batch-size distributions above the learned
  gate, and keep 2400/3200 access gated.
- **Sparse primary-policy coverage:** monitor full-search rows per hour when probability falls to 20%; do not reduce it
  to 15% without new evidence.
- **Model-stage distribution shift:** report stopping and calibration metrics separately for every promoted model.
- **Auxiliary interference:** report per-head losses and gradient-scale indicators; the legal and irreversible-progress
  heads are the first optional heads to reduce if shared learning degrades.
- **Legal-head class imbalance:** use separately averaged legal and illegal BCE components.
- **Terminal future-value sign errors:** test alternating player perspectives explicitly around mate and other
  terminal trajectories.

## 10. Rejected alternatives

- Exact checkpoint barriers or correction for a few in-flight simulations.
- Per-simulation stopping checks.
- An unchecked jump from 1600 to 3200.
- Repeated use of the same constant learned prediction after its one gate decision.
- Universal 3200-search teachers or randomized high-budget teachers for every target.
- A learned head that independently authorizes stopping without deterministic search evidence.
- Eight parallel leaves.
- Graph-search-dependent allocation.
- Moves-left or game-length search utility.
- Eight-ply or longer future-value targets.
- A normalized legal-move distribution; legality is independent binary classification.

## 11. Literature and inference boundary

- [KataGo](https://arxiv.org/abs/1902.10565) demonstrates that mixed search budgets and auxiliary targets can improve
  training efficiency, and documents forced-playout pruning. It does not establish the numeric chess stopping
  thresholds in this plan.
- [KataGo's methods documentation](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md) describes
  additional value targets and search heuristics. Its reported benefits motivate compact auxiliary supervision, not
  wholesale transfer of Go-specific heads.
- [Learning to Stop: Dynamic Simulation Monte-Carlo Tree Search](https://arxiv.org/abs/2012.07910) demonstrates a
  learned stopping approach in NoGo and 9x9 Go. It supports the general feasibility of learned allocation but does not
  demonstrate chess self-play policy-target safety or the thresholds chosen here.
- [AlphaZero](https://arxiv.org/abs/1712.01815) supports policy/value learning from self-play search. It does not
  provide an adaptive stopping algorithm for this workload.

The numeric rules, progressive schedule, search-correction construction, and auxiliary weights in this document are
repository-specific engineering decisions. They must be validated on this chess implementation rather than
attributed to those sources.

## 12. Phased implementation and approval

1. Add typed configuration variants and shadow native 100-visit telemetry without changing stop behavior.
2. Add final network-value and prior/policy-correction diagnostics to the typed completed-game boundary.
3. Add replay materialization, model head, objective, inference export, and calibration reporting for search
   correction.
4. Implement deterministic termination and the single learned tail gate.
5. Add the four training-only auxiliary targets and their materialization, loss, augmentation, and terminal tests.
6. Run the offline 3200 audit and select the deterministic and learned thresholds.
7. Run the production-shaped canary and equal-average-compute paired matches.
8. Present the measured results and proposed production YAML for explicit approval.
9. Only after approval, commit the final production configuration and launch the second four-day run.

Each phase requires relevant C++ and Python tests, `ruff format`, `ruff check --fix`, and the repository pytest command
with `--import-mode=importlib`. No phase may provision or touch the compute node until separately authorized.
