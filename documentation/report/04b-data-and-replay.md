# 4B. Getting more learning from each game

Self-play positions are not interchangeable units of data. Some contain a large correction from search; some repeat
an opening the model already understands; some carry a reliable terminal result; and some end only because the
runtime reached a wall-clock-motivated cap. The replay system decides which of those positions survive, how often
they return to the optimizer, and which parts of each target are actually known. Those decisions shape the learning
problem as directly as the network or optimizer does.

The project gradually separated four operations that are easy to confuse. *Generation* chooses the trajectories to
search. *Admission* turns eligible observations into durable rows. *Selection* chooses rows for a batch. *Weighting*
changes their contribution after selection. A fifth mechanism, presentation credit, governs when training may
advance. This vocabulary matters because a technique that prioritizes future games is not the same as prioritized
replay, and drawing a row more often is not the same as increasing its loss weight. The underlying investigations
are indexed in the [data and replay experiment ledger](../experiments/data-and-replay.md).

## Replay is both memory and clock

A replay window mediates between freshness and breadth. A small FIFO quickly removes targets made by weak old
policies, but concentrates learning on a narrow recent distribution. A large window preserves more openings,
endgames, and policy eras, yet can keep stale targets alive after the acting model has moved on. Merely allocating a
large capacity does not create information: early in training, a twenty-million-row store may contain only a small
fraction of that amount.

The retained design therefore preallocates the physical memory map once while growing its logical capacity through
0.6, 1.2, 2.0, 2.8, 4, 6, 8, 12, 16, and 20 million rows. Growth avoids pretending that an empty early window offers
diversity and later allows a broader policy history to remain available. A mature earlier campaign showed why
freshness must be measured rather than inferred: strength continued to improve after fixed-dataset policy accuracy
largely saturated, while larger models and deeper search reduced the rate of new positions. That observation
motivates the wider window but does not isolate this exact schedule as a strength improvement
([training-dynamics audit](../benchmarks/chess-v34-training-dynamics-rtx4070s-20260912/README.md)).

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
objective weighting, and inference alongside reuse. Ratio four is consequently a deliberate freshness bias, not a
measured standalone explanation for final strength. Both configured reuse and effective presentations per distinct
admitted row are needed to describe what the learner actually saw.

## Choosing information without inventing it

Within the live window, the sampler asks where search most revised the raw policy. This is related to
[prioritized replay](https://arxiv.org/abs/1511.05952), but the priority signal and absence of importance correction
are specific to this project. Policy surprise is the divergence
between the visit distribution and the network prior. Seventy percent of draws are allocated in proportion to this
signal, capped at 2.0, while 30% remain uniform. The cap keeps a few extreme rows from monopolizing training and the
uniform component preserves broad coverage. The same row may reappear across optimizer steps, but sampling is
without replacement within one global batch.

This is prioritization, not an unbiased estimate of the uniform-replay objective. The implementation deliberately
does not apply inverse-probability correction, so the optimizer learns from the prioritized distribution. Surprise
can also reflect search noise, target age, phase of play, or the number of legal moves—not only genuine difficulty.
The sampler is part of the successful retained bundle, but there is no isolated online ablation for its 70/30 mix or
cap.

Loss weighting is a separate path. Each row has a typed positive sample weight which, after batch-mean
normalization, multiplies the primary and eligible auxiliary losses. Ordinary rows in the selected recipe use weight
1.0. Historical replay could aggregate duplicate positions and encode multiplicity through weights; that verifies
that the mechanism worked, not that arbitrary weighting improved play. Claims that the final system “weights hard
positions more” should therefore refer to draw probability unless a non-unit loss-weight schedule is explicitly
being discussed. TD-error priority, recency weighting, and global deduplication remained proposals rather than
completed current experiments ([sample-stream audit](../analysis/v8-training-data-comparison-20260826.md)).

Admission happens earlier still. Random opening-prefix moves and the reconstructed prefix of a restart game have no
search observation and create no replay row. A normal searched move does. A final search performed only to value a
cut position can also become a row even though it selects no played action. Sparse policy storage retains at most 60
actions and records how much visit mass was discarded. These rules define the data set before sampling or weighting
can act on it ([replay system](../system/replay-and-data.md)).

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
This differs from the learned [targeted search control](https://arxiv.org/abs/2302.12359) explored in prior work:
the archive here uses observed search disagreement and untried branches, not a trained restart policy.

Selection from this archive gives 30% probability to uniform choice. The remaining reservations favor the square
root of *value correction*: half the absolute difference between the searched root value and the raw network value.
The square root softens the priority so one extreme correction cannot dominate. Age and capacity bounds remove old
states, reservations prevent duplicate claims within a worker, and exhausted positions leave the archive. This is
the project's practical difficult-state curriculum, but “difficult” is an interpretation of disagreement, not a
ground-truth label. The mechanism does not train a regret network, and archives are local to workers rather than
globally deduplicated.

Restart priority and replay surprise are related only in motivation. One changes which future trajectories are
generated; the other changes which already stored rows are selected for optimization. Their individual Elo effects,
the 50% start mixture, and the exact filters were not isolated. They should be presented as a coherent curriculum
design grounded in branch coverage rather than as independent measured gains
([reference-recipe analysis](../analysis/reference-recipes-for-a-compute-poor-run.md)).

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
journaling, persistence, and exclusion of a capped continuation, but did not establish a safe production threshold
([resignation canary](../benchmarks/resignation-audit-canary-20260723/README.md)). The defensible conclusion is that
resignation is calibrated and continuously audited—not that it has zero error or a measured independent Elo gain.

A ply cap creates a harder target problem because there is no observed result at all. The project compared material
heuristics, raw network value, values from adjacent searches, and a fresh search at the cut position. Playing games
beyond the normal cap supplied later outcomes for evaluation. At the earlier measured checkpoint, the cut-position
search achieved Brier score 0.444 and cross-entropy 0.756, compared with 0.491 and 0.851 for material divided by 39.
At the later checkpoint the corresponding scores were 0.193 and 0.374 versus 0.388 and 0.707. Search-root sign
accuracy exceeded 98% in both cohorts; calibration, not merely sign, distinguished the targets. The retained worker
therefore performs one full search at the actual cut position and uses its root value as the bootstrap
([cut-position benchmark](../benchmarks/cut-game-value-target-rtx4070super-20260825/README.md)).
The scalar root value `v` becomes a soft WDL target: with `r = 1 - |v|`, the win, draw, and loss components are
`max(v, 0) + r/3`, `r/3`, and `max(-v, 0) + r/3`. Materialization then applies the configured per-ply blur.

The cut policy also closes the most damaging data failure found in the project. A former cheap-search tail removed
searched endgame rows while broadcasting one shallow cutoff estimate back through each affected game. Weak endgame
targets then produced weak conversion, more capped games, and more weak targets; replay eviction could remove the
bad rows while their effect remained in the weights. The full reconstruction, measured incidence, and two-stage
repair belong to [the late-game target-poisoning study](05a-three-failures.md#late-game-target-poisoning). Here the
relevant data rule is simply that cut-value provenance and policy-row eligibility must be audited together.

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
rows rather than treating missing labels as zeros. Overfit tests established that the supplied multi-head targets
were learnable, while replay audits established that censoring and offsets were wired correctly. Neither establishes
an independent playing-strength gain ([multi-head overfit study](../benchmarks/chess-overfit-rtx3090-20260819/README.md)).

Broader auxiliary bundles were removed during a period with several simultaneous training problems. That was a
precautionary simplification, not a controlled negative ablation. Retaining next-policy and remaining length later
does not prove that either head helped, just as removing the others does not prove that they harmed. The firm result
is semantic: labels derived from the future must be materialized from complete trajectories and masked when the
future is unknown.

## Fresh targets versus fresh positions

An older replay design synchronously re-searched some recent positions and stored source-bound target overrides.
No controlled study established that this used search more effectively than generating fresh states and outcomes,
and the mechanism disappeared with its replay schema. Reanalysis is therefore superseded infrastructure, not a
negative efficacy result ([historical implementation record](../history/v10-training-quality-implementation.md)).

Model publication addresses freshness from the other side. Publishing every 100 optimizer steps was implemented
historically and rejected because repeated serialization, deployment preparation, validation, synchronization, and
activation outweighed the observed benefit. The raw timing bundle and matched strength curve were not preserved, so
this remains a qualitative decision record, not an effect-size result. The retained 500-step boundary is a
compromise, not a universal optimum; its cost and freshness also depend on replay reuse
([publication-cadence record](../history/historical-research-backlog-20260822.md),
[current decomposition](../benchmarks/v39-selfplay-throughput-rtx4070s-20260913/README.md)).

## Overlap without full asynchrony

Training proceeds in coordinated credit-funded quanta while half the actor processes continue self-play. A pause
sweep found only a narrow, workload-dependent difference between balanced choices, so the retained fraction is an
operational setting rather than a universal optimum. Publication, replay snapshots, and optimizer commits remain
synchronized; fully asynchronous learning was proposed but not implemented. The throughput measurements and a
corrected device-placement artifact are covered in [Chapter 5](05-systems-optimization.md#overlapping-self-play-and-training)
and the [pause benchmark](../benchmarks/selfplay-pause-tradeoff-rtx4070s-20260902/README.md).

## Making the data trustworthy

Completed games pass through bounded, worker-owned materialization into a typed circular store. Atomic writes,
quarantine with a fatal rejection ceiling, immutable trainer snapshots, deterministic cross-rank sampling, and
credit only after durable append preserve the meaning of the experiments under load. These are correctness and
throughput properties, not direct evidence of stronger chess; the implementation and benchmarks belong to
[Chapter 5](05-systems-optimization.md#replay-materialization-and-training-supply), the
[pipeline design](../architecture/replay-pipeline-rework.md), and the
[materialization design](../architecture/replay-materialization-rework.md).

## What the retained curriculum establishes

The final data recipe combines staged replay growth, reuse four, policy-surprise sampling with a uniform component,
unit row weights, shallow random openings, branch-reserved restart states, calibrated resignation, direct
cut-position bootstraps, explicit target eligibility, coordinated publication, and partial actor/trainer overlap.
Most of these choices coexist in one successful system and do not have one-variable online Elo estimates.

The strongest direct comparison favors a searched value over the tested material heuristic at a cut position. The
strongest incident evidence is the late-game poisoning chain. The clearest systems lesson is that more presentations
are not necessarily more information: replay reuse, target freshness, state diversity, and schedule pace move
together. The retained design spends compute on a broad, inspectable stream of searched positions and preserves the
meaning of every admitted target. It should be understood as an integrated curriculum, not a collection of additive
strength bonuses.
