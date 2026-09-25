# 4.2. Getting more learning from each game

Once self-play finishes a game, the learner has to decide what it can learn from that game and when to see those
positions again. It can start future games from an interesting branch, admit searched positions to replay, and
revisit positions where search corrected the network's move preference. These choices do different jobs: a restart
creates new experience, while replay sampling reuses experience already collected. A game stopped at the ply cap
also needs a trustworthy value target because its actual outcome is unknown.

Figure 3 follows a position from game generation to training. The sections below examine the decisions at each
boundary and the evidence behind the retained choices. In particular, drawing a row more often is not the same as
increasing its loss weight, and admitting a row is not the same as funding an optimizer step.

![Generation, admission, selection, weighting, and presentation credit as distinct replay decisions](figures/replay-decision-path.svg)

Figure 3: Starting positions determine which games are searched; admission determines which observations become
replay rows; sampling determines which rows recur in training. Rows have unit loss weight, and newly admitted rows
fund optimizer work only after durable replay append.

## Starting games where information is likely

The start mixture tackles two coverage problems. Half of games begin after a uniformly selected zero to eight
random legal plies. These prefixes diversify the opening cheaply; they are reconstructed in the recorded history
but not searched or trained directly. Drawing zero plies also keeps some games at the ordinary initial position.

The other half begin from archived self-play states, falling back to random openings when a worker's restart
archive is empty. An archived position is eligible only if at least 15 plies remained in its source game, its
absolute root value was at most 0.8, and two or three leading actions covered 85% of visit mass. The branch actually
played is marked used. A restarted game reserves an untried plausible alternative and forces that action after
reconstructing the prefix, so new search explores a branch the source game did not.

The archive gives 30% probability to uniform selection. Otherwise it favors the square root of *value correction*:
half the absolute difference between the searched root value and the raw network value. The square root prevents
one extreme correction from dominating. Age and capacity bounds remove old states, reservations prevent duplicate
claims within a worker, and exhausted positions leave the archive. Here, “difficult” means that search corrected
the network substantially; difficulty was not independently labeled. Archives are local to workers rather than
globally deduplicated.

This differs from the learned targeted search control [5] explored in prior work: the archive uses observed search
disagreement and untried branches, not a trained restart policy. Restart priority creates new trajectories; replay
surprise, discussed below, revisits stored positions. Neither restart priority nor the start mixture and filters
has an isolated Elo estimate. Together they broaden branch coverage and return the learner to positions where
search changed its prior.

## What becomes a training position

Not every move in a recorded game has a search target. Random opening-prefix moves and the reconstructed prefix of
a restart game were played or replayed to reach a starting position; they were not searched and create no replay
row. A normal searched move does. A final search performed only to value a cut position can also become a row even
though it selects no played action. Sparse policy storage retains at most 60 actions and records how much visit
mass was discarded. These rules determine the data set before sampling or weighting can act on it.

## Replay is both memory and clock

A replay window has to retain enough varied positions without teaching from very old play forever. A small FIFO
forgets early policies quickly, but also forgets openings and endgames that have not recently appeared. A large
window preserves breadth while keeping older targets alive. The system therefore begins with a small logical
window and expands it as new games fill the store; merely reserving twenty million slots would not create twenty
million distinct positions.

The physical memory map is preallocated once, while its logical capacity grows through
0.6, 1.2, 2.0, 2.8, 4, 6, 8, 12, 16, and 20 million rows. Growth avoids pretending that an empty early window offers
diversity and later allows a broader policy history to remain available. A mature earlier campaign showed why
freshness must be measured rather than inferred: strength continued to improve after fixed-dataset policy accuracy
largely saturated, while larger models and deeper search reduced the rate of new positions. That observation
motivates the wider window but does not isolate this exact schedule as a strength improvement.

Replay reuse introduces a second tradeoff. The configured ratio is the number of optimizer presentations funded by
each newly admitted row. In the retained setting, four presentations are credited per row; a 500-step quantum at a
global batch of 2,048 therefore requires 256,000 newly appended rows. Credits are issued only after append and flush,
and a persistent ledger prevents a restart from spending them twice.

Changing reuse also changes the pace of training, evaluation, publication, visit schedules, and replay growth,
because all advance at quantum boundaries. Higher reuse funds more updates from each game; lower reuse gives each
update fresher positions if the actors can supply them. Short controls at ratios four, 6.25, and eight remained
matched at their shared evaluation boundaries despite different update rates. They lasted only about 90 minutes,
however, and the archive no longer has their full raw curves. The completed campaign also changed replay capacity,
optimizer, objective weighting, and inference alongside reuse. Ratio four favors freshness, but the record does not
give it an isolated final-strength effect. The actual number of presentations per distinct admitted position can
also differ from the configured ratio.

## Choosing which positions recur

Search is most informative when it substantially changes the network's initial move preference. Within the live
window, the sampler therefore favors rows with high *policy surprise*: divergence between search's visit
distribution and the network prior. Seventy percent of draws follow this signal, capped at 2.0 so extreme rows
cannot dominate; the remaining 30% are uniform to preserve coverage. A row can recur across optimizer steps but is
drawn only once within a global batch. This is related to prioritized replay [4], though it uses a different signal.

The optimizer learns from this prioritized distribution without inverse-probability correction. Surprise can
reflect search noise, target age, game phase, or legal-move count as well as genuine difficulty; the 70/30 mix and
cap do not have an isolated online strength estimate.

Loss weighting is separate. Each row has a positive sample weight which, after batch-mean normalization, multiplies
the primary and eligible auxiliary losses. Rows in the selected recipe use weight 1.0. Earlier replay could
aggregate duplicate positions and encode multiplicity through weights; that established the mechanism, not a
playing-strength gain from arbitrary weights. Thus the final system sees policy-surprising positions more *often*;
it does not increase their per-draw loss weight. TD-error priority, recency weighting, and global deduplication
remained proposals.

## Ending games without corrupting their labels

Resignation can save a large amount of search, but a false resignation turns a drawable or winning position into an
incorrect terminal label. A fixed value threshold was therefore replaced by continuing measurement. Twenty percent
of games are designated at creation as continuation games and can never resign. They reveal the outcomes of
hypothetical triggers for thresholds from -0.99 through -0.70. A trigger requires both the root value and the best
visited child's backed-up value to cross the threshold; evidence from capped games is excluded because their natural
outcome remains unknown.

Over a rolling window, a candidate threshold needs at least 100 triggered continuation games and a one-sided 95%
upper confidence bound on false non-loss no greater than 2.5%. The threshold may become more conservative
immediately but relaxes by at most 0.01 per publication. An intentionally aggressive canary verified trigger
journaling, persistence, and exclusion of a capped continuation. It tested the mechanism, not the safety of a
production threshold. The retained system therefore calibrates and audits resignation continuously; neither zero
error nor a separate Elo gain has been established.

A ply cap creates a harder target problem because there is no observed result at all. The project compared material
heuristics, raw network value, values from adjacent searches, and a fresh search at the cut position. Playing games
beyond the normal cap supplied later outcomes for evaluation. At the earlier measured checkpoint, the cut-position
search achieved Brier score 0.444 and cross-entropy 0.756, compared with 0.491 and 0.851 for material divided by 39.
At the later checkpoint the corresponding scores were 0.193 and 0.374 versus 0.388 and 0.707. Search-root sign
accuracy exceeded 98% in both cohorts; calibration, not merely sign, distinguished the targets. The retained worker
therefore performs one full search at the actual cut position and uses its root value as the bootstrap.
The scalar root value `v` becomes a soft WDL target: with `r = 1 - |v|`, the win, draw, and loss components are
`max(v, 0) + r/3`, `r/3`, and `max(-v, 0) + r/3`. Materialization then applies the configured per-ply blur.

The cut policy addresses the most damaging data failure found in the project. A cheap-search tail excluded late
positions from primary replay while a shallow cutoff estimate supplied the value target for earlier rows in the
same capped game. This plausibly reinforced weak conversion: fewer searched endgame examples, more capped games,
and more weakly grounded targets. Replay eviction could remove the offending rows without undoing their effect on
the network. The measured incidence and multi-change repair are examined in Chapter 6; neither this reconstruction
nor the recovery isolates one cause. The general data rule is to audit cut-value provenance and policy-row
eligibility together.

## Knowing what each target means

An eventual game result is informative, but a position many moves before that result should not receive an equally
sharp target. During materialization, the terminal WDL target is blurred toward uniform by 0.998 for each remaining
game ply. During optimization, a scheduled coefficient eventually blends up to 10% of the stored search-root scalar
into that discounted outcome. Search has its own 0.99 per-tree-ply discount, which changes backups, move selection,
and the root value before replay is written. These mechanisms act at different points in the learning loop. The
cut-position benchmark favors searched values over the tested material heuristic; it does not separately measure
the ordinary 10% blend or either discount.

Future-dependent auxiliary targets require equally explicit eligibility. The retained next-policy target uses the
sparse visit distribution from the following searched observation; it is unavailable if that observation does not
exist. Remaining game length is known only when the game reaches a natural result or valid resignation, so every row
from a cut game marks that target ineligible. The objective masks and renormalizes auxiliary losses over eligible
rows rather than treating missing labels as zeros. Overfit tests showed that the multi-head targets were learnable,
and replay audits checked censoring and offsets; their independent playing-strength contribution remains unmeasured.

Broader auxiliary bundles were set aside while diagnosing several simultaneous training problems, not because a
controlled ablation found them harmful. Future-derived labels must come from complete trajectories and be masked
whenever the future is unknown.

## Fresh targets versus fresh positions

Reanalysis would spend search on fresher targets for existing positions instead of new states and outcomes. An
earlier replay design implemented bounded synchronous re-search with source-bound target overrides, but no
controlled study compared its learning return with fresh self-play. It disappeared with that replay schema.
Reanalysis is therefore superseded infrastructure, not a negative efficacy result.

Model publication addresses freshness from the other side. Publishing every 100 optimizer steps imposed enough
serialization, validation, and activation overhead to be rejected, but no preserved matched strength curve gives
that choice an effect size. The retained 500-step boundary trades target age against publication cost, and its
wall-clock cadence depends on replay reuse.

## Overlap without full asynchrony

Training proceeds in coordinated credit-funded quanta while half the actor processes continue self-play. A pause
sweep found only a narrow, workload-dependent difference between balanced choices, so the retained fraction is an
operational setting rather than a universal optimum. Publication, replay snapshots, and optimizer commits remain
synchronized; fully asynchronous learning was proposed but not implemented. The throughput measurements and a
corrected device-placement artifact are covered in Chapter 5.

## Making the data trustworthy

Completed games enter a typed circular store through bounded, worker-owned materialization. Atomic writes,
quarantine with a fatal rejection ceiling, immutable trainer snapshots, deterministic cross-rank sampling, and
credit only after durable append keep targets and optimizer progress auditable under load. These are correctness
and throughput properties, not direct chess-strength ablations; Chapter 5 covers the implementation.

## What the retained curriculum establishes

The final data recipe combines staged replay growth, reuse four, policy-surprise sampling with a uniform component,
unit row weights, shallow random openings, branch-reserved restart states, calibrated resignation, direct
cut-position bootstraps, explicit target eligibility, coordinated publication, and partial actor/trainer overlap.
Most of these choices coexist in one successful system and do not have one-variable online Elo estimates.

The cut-position comparison favors a searched value over the tested material heuristic, while the late-game
poisoning incident shows what happens when target eligibility and value provenance are mishandled together. More
presentations do not necessarily mean more information: reuse, freshness, diversity, and schedule pace move
together. The resulting curriculum preserves a broad stream of searched positions and the meaning of each admitted
target.
