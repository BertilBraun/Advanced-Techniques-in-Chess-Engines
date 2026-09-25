# 4.2. Getting more learning from each game

More games help only if they add positions with interpretable targets. The chess curriculum diversifies starts,
stores searched positions in a growing replay window, samples policy-surprising rows more often, and funds four
training presentations per admitted row. It also treats a resolved outcome differently from a game stopped at the
ply cap, where a searched value proved a better target than the tested material heuristic.

The curriculum acts at several points in the data flow. *Generation* chooses trajectories to search; *admission*
turns eligible observations into durable rows; *selection* chooses batch rows; and *weighting* changes their
contribution to the loss. Presentation credit separately governs when training may advance. Restart priority acts
on future games, while policy surprise acts on stored rows.

![Generation, admission, selection, weighting, and presentation credit as distinct replay decisions](figures/replay-decision-path.svg)

Figure 3: The retained curriculum changes where games start and which valid positions recur, but leaves admitted
rows at unit loss weight. The separate presentation-credit ledger allows optimizer work only after replay append
and flush; a sampled row is not the same unit as a newly generated row.

## Replay is both memory and clock

A replay window mediates between freshness and breadth. A small FIFO quickly removes targets made by weak old
policies, but concentrates learning on a narrow recent distribution. A large window preserves more openings,
endgames, and policy eras, yet can keep stale targets alive after the acting model has moved on. Allocating room for
twenty million rows does not itself create twenty million distinct positions.

The retained design therefore preallocates the physical memory map once while growing its logical capacity through
0.6, 1.2, 2.0, 2.8, 4, 6, 8, 12, 16, and 20 million rows. Growth avoids pretending that an empty early window offers
diversity and later allows a broader policy history to remain available. A mature earlier campaign showed why
freshness must be measured rather than inferred: strength continued to improve after fixed-dataset policy accuracy
largely saturated, while larger models and deeper search reduced the rate of new positions. That observation
motivates the wider window but does not isolate this exact schedule as a strength improvement.

Replay reuse introduces a second tradeoff. The configured ratio is the number of optimizer presentations funded by
each newly admitted row. In the retained setting, four presentations are credited per row; a 500-step quantum at a
global batch of 2,048 therefore requires 256,000 newly appended rows. Credits are issued only after append and flush,
and a persistent ledger prevents a restart from spending them twice.

This ratio is not merely a sample-efficiency knob. Because training, evaluation, model publication, visit schedules,
and replay growth advance at quantum boundaries, changing reuse also changes their wall-clock pace. Higher reuse
funds more optimizer work from the same generated data; lower reuse exposes each update to more fresh positions if
the actors can supply them. Short controls at ratios four, 6.25, and eight remained matched at their shared
evaluation boundaries despite different update rates, but all ended within roughly 90 minutes and their full raw
curves are not preserved in the benchmark archive. The completed campaign also changed replay capacity, optimizer,
objective weighting, and inference alongside reuse. Ratio four thus favors freshness, although its isolated effect
on final strength is unknown. Effective presentations per distinct admitted row describe exposure more fully than
the configured ratio alone.

## Choosing information without inventing it

Within the live window, the sampler favors rows where search most revised the raw policy. This is related to
prioritized replay [4], but uses a different signal and no importance correction. Policy surprise is the divergence
between the visit distribution and the network prior. Seventy percent of draws are allocated in proportion to this
signal, capped at 2.0, while 30% remain uniform. The cap limits domination by extreme rows and the uniform
component preserves broad coverage. The same row may reappear across optimizer steps, but sampling is without
replacement within one global batch.

There is no inverse-probability correction: the optimizer learns from the prioritized distribution. Surprise may
reflect search noise, target age, phase of play, or the number of legal moves as well as genuine difficulty. The
70/30 mix and cap have no isolated online strength estimate.

Loss weighting is separate. Each row has a positive sample weight which, after batch-mean normalization, multiplies
the primary and eligible auxiliary losses. Rows in the selected recipe use weight 1.0. Earlier replay could
aggregate duplicate positions and encode multiplicity through weights; that established the mechanism, not a
playing-strength gain from arbitrary weights. Thus the final system sees policy-surprising positions more *often*;
it does not increase their per-draw loss weight. TD-error priority, recency weighting, and global deduplication
remained proposals.

Admission happens earlier still. Random opening-prefix moves and the reconstructed prefix of a restart game have no
search observation and create no replay row. A normal searched move does. A final search performed only to value a
cut position can also become a row even though it selects no played action. Sparse policy storage retains at most 60
actions and records how much visit mass was discarded. These rules define the data set before sampling or weighting
can act on it.

## Starting games where information is likely

The configured start draw attacks two different coverage problems. Half of games are assigned a uniformly selected
zero to eight random legal plies. These prefixes cheaply diversify the opening without pretending to be a balanced
opening book. They are reconstructed in the recorded history but not searched or trained directly; their value is
the downstream positions they expose. Drawing zero plies naturally retains some games from the ordinary initial
position.

The other half are assigned archived self-play states, falling back to random openings when a worker's restart
archive is empty. A position is eligible only if at least 15 plies remained in its
source game, its absolute root value was at most 0.8, and two or three leading actions covered 85% of visit mass. The
branch actually played is marked used. A later restarted game reserves one untried plausible alternative and forces
that action after reconstructing the prefix, so new compute explores a branch the source game did not.
This differs from the learned targeted search control [5] explored in prior work:
the archive here uses observed search disagreement and untried branches, not a trained restart policy.

Selection from this archive gives 30% probability to uniform choice. The remaining reservations favor the square
root of *value correction*: half the absolute difference between the searched root value and the raw network value.
The square root softens the priority so one extreme correction cannot dominate. Age and capacity bounds remove old
states, reservations prevent duplicate claims within a worker, and exhausted positions leave the archive. Here,
“difficult” means that search substantially corrected the network, not that difficulty was independently labeled.
Archives are local to workers rather than globally deduplicated.

Restart priority generates new trajectories; replay surprise revisits stored positions. Neither individual Elo
effect, nor that of the start mixture and filters, was isolated. Together they broaden branch coverage and return
the learner to positions where search changed its prior.

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
journaling, persistence, and exclusion of a capped continuation, but did not establish a safe production threshold. The defensible conclusion is that
resignation is calibrated and continuously audited—not that it has zero error or a measured independent Elo gain.

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

Three value mechanisms act at different boundaries and should not be collapsed into “value discount.” During
materialization, the terminal WDL target is blurred toward uniform by 0.998 for each remaining game ply. During
optimization, a scheduled coefficient eventually blends up to 10% of the stored search-root scalar into that
discounted outcome. Inside search, a separate 0.99 per-tree-ply discount changes backups, selection, and the root
value before replay is written. The cut-position benchmark supports searched values as better bootstraps than the
tested material heuristic; it does not isolate the ordinary 10% blend or either discount as a strength gain.

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
