# Handoff: designing the next chess training run

Written 2026-09-06, after the v29 run was stopped, measured and merged. This document is the starting
context for a fresh session whose job is to reason through what the next run should look like.

Read `CLAUDE.md` and `AGENTS.md` first. Then this. Then
`documentation/analysis/reference-recipes-for-a-compute-poor-run.md`.

---

## The goal

**Superhuman chess strength, as fast as possible in wall-clock.** Not beating Stockfish. The
secondary goal is a publishable result. Hardware is 8x RTX 4070 SUPER rented on Vast.ai, self-play
and training sharing the same cards, runs of roughly four days. We are compute-poor by the standards
of every paper we compare against, and the interesting question is what to do about that.

The historical bar is a four-day run that reached ~2,800 ladder Elo at 10k visits.

---

## What v29 achieved, and where it stopped

Final state: generation 1002, 501,000 optimizer steps, 71.3 h. Configuration preserved at
`documentation/benchmarks/ladder-elo-vs-generation-rtx4070s-20260906/reference-config/` (it will not
load against current code -- it carries blocks for machinery since removed -- but it is the exact
resolved config, verified against the run's own archive manifest).

Strength against wall-clock, measured at 10,000 searches/move after the run
(`documentation/benchmarks/ladder-elo-vs-generation-rtx4070s-20260906/`):

| Generation | Hours | Ladder Elo |
|---|---|---|
| 100 | 2.7 | 2034 |
| 200 | 6.0 | 2298 |
| 300 | 9.7 | 2408 |
| 400 | 14.0 | 2434 |
| 500 | 20.0 | 2614 |
| 600 | 28.0 | 2693 |
| 700 | 38.3 | 2693 |
| 800 | 53.3 | 2774 |
| 900 | 62.3 | 2828 |
| 1000 | 71.3 | 2800 |

**First 28 h: +660 Elo at 26.1 Elo/h. Last 43 h: +107 Elo at 2.5 Elo/h.** A 10x collapse, breaking
between generations 600 and 700. The run was ~90% finished, in strength terms, in its first 28 hours.

A 200-game match at generation 936 puts the model at **2844.8 Elo [2806.6, 2882.3]** -- the tightest
number we have (`documentation/benchmarks/deep-match-generation936-50k-nodes-rtx4070s-20260906/`).

**So v29 matched the four-day bar in roughly 72% of the wall-clock, and reached most of it in a day.**
Caveats: this lineage is not from-scratch (checkpoint resumes, a stopping fork, a mid-run visit
change), and we do not know the bar's own measurement conditions.

---

## The diagnosis, and the thing most likely to be wrong about it

Two facts, and they are easy to conflate:

1. **Elo per hour collapsed 10x** after generation 600.
2. **Generations per hour collapsed 4.5x** over the same span -- 118 s/gen at generation 200 rising to
   535 s/gen by generation 800, from the 19x176 model promotion and 1000-visit self-play. The visit
   cut to 600 at generation 797 pulled it back to ~325 s/gen.

The first 28 h bought 600 generations; the next 43 h bought 400. **Elo per generation held up far
better than Elo per hour.** That points at throughput as the primary lever rather than the training
recipe -- which is the opposite of what the loss curves alone suggest, since training loss drifted
*up* through the plateau.

**Do not treat this as settled.** It is one run, the per-generation series is noisy at 40 games per
point, and "throughput is the bottleneck" is a hypothesis that the next run should be designed to
test, not assume.

---

## The data economics, which is the deeper problem

| | AlphaZero chess | v29 |
|---|---|---|
| Optimizer steps | 700,000 | 501,000 |
| Batch | 4096 | 2048 |
| Presentations | 2,867 M | 1,026 M |
| **Distinct positions** | **~4,400 M** | **128 M** |
| **Reuse factor** | **0.65** | **8.0** |
| **Fresh positions per step** | **~6,286** | **256** |
| Simulations/move | 800 | 1000 -> 600 |
| Self-play hardware | 5,000 TPUs | shares our 8 GPUs |

AlphaZero never presented a position twice. We present each eight times, drawn from a 5 M window that
stops growing at generation 700. At AlphaZero's 150k-step mark -- the phase the user cares about,
where most of its gain happened -- it had seen ~940 M distinct positions. We reached 128 M after
501k steps.

**We are not under-trained. We are data-starved, and the extra steps re-fit a small stale window.**

---

## Measurement: what the instruments actually do

This cost us real time during v29 and must not be repeated.

- **The in-run 64-search ladder understates by 550-684 Elo**, and the offset *widens* over a run. It
  also compresses gains: it reported +625 across v29 where the truth was +766. Usable rule: add ~550
  early, ~650 late, multiply observed gains by ~1.2. Fine as a relative regression signal. Never
  quote it as a level.
- **Rung counts change as the model outgrows rungs.** Fits over different rung sets are not
  comparable. Always carry the rung count. This produced several wrong conclusions mid-run.
- **A 40-game ladder resolves roughly 80 Elo.** A 200-game single-rung match gives about +/-38. Most
  changes worth making are smaller than the first number, so **measurement resolution is a binding
  constraint on improving the recipe, not only on reporting it.**
- **Evaluation search budget dominates the number**: the same weights read 2157 at 64 searches, 2565
  at 800, and 2800-2845 at 10,000.
- `parallel_searches` is an upper cap on in-flight leaves per tree and **only binds when games cannot
  fill the inference batch**. At 400 concurrent games with batch 64 it is inert. Its Elo cost is
  somewhere in -6 to -45 and is **not resolved**; separating those needs ~2,000 games per arm. Use
  `parallel_searches` 1 for anything reported; higher is fine for small probe ladders and for
  self-play, where batches genuinely starve.

---

## What the literature says (full detail in `reference-recipes-for-a-compute-poor-run.md`)

- **Playout cap randomization: do not implement.** Only full-search turns are recorded, so at KataGo's
  parameters fast moves burn ~33% of search and produce nothing. The only chess-family evaluation
  (Czech 2019, crazyhouse) rejected it, finding it better suited to less tactical games with longer
  average length. And its payoff is denominated in self-play compute, which our own fork experiment
  priced at ~1/5 of face value.
- **Visits per move: ours are fine.** 300-1200 sits inside every published band. There is a failure
  wall below ~100-200 simulations; above it, Gumbel-style methods buy parity, not gain.
- **The promising cluster is prioritising surprising samples** -- KataGo policy-surprise weighting,
  lc0 value focus, RGSC (ICLR 2026: +77 Elo over AlphaZero). It is the only approach that attacks the
  fresh-data bottleneck without needing more data, and the sample-weight and restart-state plumbing
  already exists.
- **Two config-only changes rank above all code work**: a terminal learning-rate decay (v29 decayed
  3.3x total and took no sharp drop; AlphaZero and lc0 drop ~1000x, KataGo 10x for final tuning), and
  uncapping the replay window past 5 M.
- **Do not bother**: periodic network resets, further adaptive search budgeting, TD-error PER,
  switching to SGD to inherit AlphaZero's schedule.
- **Unverified**: the claim that only AlphaZero chess's first LR drop mattered. The published figure
  is not tabulated and confounds LR drops with an improving data distribution.

---

## Deferred and open

- **Six failing tests** (five in `test_experiment_queue_process.py`, one segfault in
  `test_interactive_engine.py`). Pre-existing, Linux-only, outside the training path. A fix is in
  progress; merge it when it lands.
- **The promotion criterion is `maximum_relative_loss: 1.01`** -- it promotes a candidate slightly
  *worse* than the incumbent. v29 promoted three model tiers; the user's view is that the later
  promotions were not clearly worth it.
- **Self-play `parallel_searches` is derived natively**, not configured: `searchParallelism` gives 4
  at 600 visits and 8 at 1000. Whether that harms *target fidelity* has never been measured -- the
  right metric there is target quality, not playing Elo. Cheap to measure, and worth doing before
  changing it, because reducing it costs throughput and throughput is the suspected bottleneck.
- **A surviving finding from the declined inference-cache work**: `parallel_searches` 2 gave +29.9%
  search throughput over 1 across two matched seeds. Relevant to the same question.

---

## Where things are

- `master` is a clean slate: adaptive search budgeting and learned early stopping removed from both
  the Python and native sides, the inference contract back to two tensors.
- Closed work is preserved as GitHub releases, not branches: `adaptive-stopping-final`,
  `mcgs-rejected`, `inference-cache-declined`.
- All v29 measurement lives under `documentation/benchmarks/*-20260906/`.
- Run evidence is under `.codex-diagnostics/` (gitignored): the node evaluations and the final v29
  archive's tensorboard, logs and config, SHA256-verified.

## Rules that bind the next session

The user owns approvals, launches, stops and phase acceptance. Never start, stop or reconfigure a run
without explicit instruction, and never spend GPU time that was not authorised. Native builds happen
on a node, never locally. Every measurement carries its configuration SHA, and a run with no fetched
archive did not happen.

## The question to answer

Given ~26 Elo/h available in the first day and ~2.5 Elo/h after it, and given that we are data-starved
rather than under-trained: **what configuration reaches a higher strength in the first 24-30 hours,
and what would make the tail worth running at all?**

A defensible answer needs to say how it will be measured to a resolution finer than the effect it
expects.
