# Search and self-play

## One native search implementation

Chess, Go 7×7, and Go 9×9 instantiate the same game-parameterized native search. The core owners are
[`SearchEngine.hpp`](../../cpp/src/search/SearchEngine.hpp),
[`SearchExecutor.hpp`](../../cpp/src/search/SearchExecutor.hpp),
[`SearchTree.hpp`](../../cpp/src/search/SearchTree.hpp), and
[`SelfPlay.hpp`](../../cpp/src/search/SelfPlay.hpp). Python supplies resolved schedules and owns the game lifecycle;
it does not implement MCTS.

Search uses a tree arena, batched leaf selection and expansion, virtual loss for concurrent traversals, PUCT,
configurable first-play urgency, value discounting, and an asynchronous inference pipeline. Requests from many
active games share inference batches. Trees are retained across moves and are reset when the model changes.

Every current search has a fixed limit. Adaptive budget allocation and learned early stopping were removed after
their Elo evaluations; the retained evidence is summarized in
[`adaptive-search-conclusion-20260904.md`](../analysis/adaptive-search-conclusion-20260904.md). A request may express
an absolute visit limit for evaluation or an additional-visit limit for a retained self-play root, but the budget is
not predicted by the network.

## Final self-play search policy

The final configuration resolves the following policy for each published generation:

- visits per move: 300 initially, 400 from generation 10, 500 from generation 50, 600 from generation 90, and 800
  from generation 1,000;
- PUCT exploration constant `1.5`;
- reduced-parent-value FPU with reduction `0.2`;
- Dirichlet root noise with epsilon `0.25` and alpha `0.3`;
- forced root playouts with coefficient `1.5`;
- value backups discounted by `0.99` per ply;
- 60% of the retained root visits kept before the next move;
- per-position parallel search selected from the fixed budget and capped at four by the current native policy.

Forced playouts intentionally affect exploration but not the stored policy target. After search, the pruning logic
in [`ForcedPlayouts.hpp`](../../cpp/src/search/ForcedPlayouts.hpp) removes visits that the ordinary PUCT score does
not support. This prevents explicitly forced exploration from becoming an equally strong training label.

## Batched workers and model transitions

The final topology runs 32 self-play worker processes, four assigned to each GPU, with 512 active games per process.
Each worker owns one native search and advances its entire active pool one batched move at a time. Its TensorRT
configuration uses one inference worker, batches of 320 positions, and two outstanding batches. Exact topology and
backend paths remain in
[`chess-final-config.yaml`](../../py/configs/production/chess-final-config.yaml).

At a checkpoint transition, a worker finishes its current native batch, refreshes the deployment artifact, applies
the next generation's schedules, and resets its trees. Games may span generations; every search observation records
the generation that produced it. If the required tree capacity or search value discount changes, roots are rebuilt
from their current positions. Worker supervision and pause/resume ownership live in
[`training/self_play_group.py`](../../py/src/training/self_play_group.py).

## Starting positions and difficult-state revisitation

The final recipe draws 50% of games from random openings of zero through eight uniformly random legal plies and
requests 50% from a per-worker SQLite restart archive. There are no ordinary initial-position draws in the configured
mixture. If a restart archive is empty, the worker falls back to the non-restart distribution, which in this recipe
means a random opening.

The archive implementation in
[`restart_archive.py`](../../py/src/self_play/restart_archive.py) extracts positions from completed games only when:

- enough plies remain;
- the absolute root value is below the configured limit;
- the leading actions cover the requested visit mass; and
- the resulting candidate count lies inside the configured range.

For the final recipe those limits are 15 remaining plies, absolute root value at most `0.8`, 85% visit mass, and two
or three candidates. The already played action is marked tried. A restart reserves one untried candidate, so later
self-play deliberately explores a plausible alternative from the archived prefix. Selection is 30% uniform and
otherwise biased toward the square root of the recorded value disagreement. Archives retain at most 50,000
positions and expire entries older than 40 generations.

This mechanism and policy-surprise replay sampling are related but distinct: restart states change which games are
generated, while replay priority changes which already generated positions are shown to the optimizer.

## Move choice and root reuse

Before the greedy cutoff, the worker samples from visit counts after applying a temperature that interpolates from
`1.3` to `0.1`. The cutoff is ply 60 initially and ply 80 from generation 110. At and after the cutoff, the largest
visit count wins with ascending action ID as a deterministic tie break. Random opening plies are not training
observations; searched moves are.

After a move, the chosen child becomes the new root. Before its next search, retained statistics are discounted to
60% and the configured budget is added. The worker implementation and its exact completion semantics are in
[`self_play/worker.py`](../../py/src/self_play/worker.py).

## Maximum length, value targets, and resignation

The maximum game length increases from 150 plies to 160, 180, 200, and finally 250 as training progresses. At the
cap, chess performs one search at the cut position and uses that position's root value as the final soft WDL target;
the remaining-length auxiliary label is censored. No Syzygy tablebase is used in this path.

Calibrated resignation becomes eligible from generation 70. A game resigns only when both the root value and the
highest-visited child's Q are below the published threshold. Twenty percent of games are designated continuation
games and never resign; their eventual outcomes provide counterfactual safety evidence across candidate thresholds.
The coordinator publishes a threshold only after the configured sample count and one-sided confidence bound satisfy
the 2.5% false-nonloss ceiling. Unsafe tightening is immediate, while relaxation is limited to `0.01` per generation.
The durable calibration and idempotent observation journal are implemented in
[`self_play/resignation.py`](../../py/src/self_play/resignation.py).

## Completed-game boundary

Each finished game records its action sequence, per-ply sparse target visits, root/network values, search
diagnostics, model generation, sample weight, final WDL, termination reason, and resignation metadata. The worker
writes a temporary file, flushes it, and atomically renames it into the replay inbox. During an orderly pause it also
persists in-flight games so a restart can resume them instead of inventing terminal values or throwing away their
search work. The schema and publication contract are in
[`self_play/completed_game.py`](../../py/src/self_play/completed_game.py).
