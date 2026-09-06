# Reference recipes for a compute-poor AlphaZero run

2026-09-06. Literature review, no new measurements. Written to inform the next four-day chess run on the
8x RTX 4070 SUPER node.

## Evidence grading used throughout

- **(A) controlled ablation** — a paper that ran the arm and the counterfactual and reports a number.
- **(B) practitioner report** — engine documentation, project logs, theses, forum/Discord write-ups. Usually
  uncontrolled, often confounded with other simultaneous changes.
- **(C) my inference** — reasoning from A/B evidence to our situation. Not measured by anyone.

Where I could not verify a claim I say so rather than asserting it.

## Two facts about our system that constrain every recommendation

1. **Fresh data per optimizer step is the stated bottleneck.** ~1.03 B presentations over ~128 M distinct
   positions (`replay_ratio: 8`), against AlphaZero's 2.87 B presentations over ~4.4 B distinct positions
   (reuse below 1).
2. **Cutting self-play search does not buy wall-clock here.** The fork experiment in
   [`adaptive-search-conclusion-20260904.md`](adaptive-search-conclusion-20260904.md) measured a 14% search
   cut converting to a 3% cadence gain, with no Elo difference at ±10 Elo paired resolution, because
   sixteen of thirty-two self-play workers run inside the trainer quantum. This is the single most
   important local result for reading the literature below: **every technique whose payoff is "self-play
   costs less" is worth roughly a fifth of its nominal value in our scheduler.** Techniques whose payoff is
   "the same gradient step is more informative" are unaffected.

A third fact matters for validation rather than design: independent runs have ~41 Elo per-bucket ladder
noise, forked paired arms ~10–24 Elo, and the 64-search ladder understates strength by ~640 Elo and
saturates. Most effects below are 20–90 Elo. **Nothing in this document can be evaluated by an
independent-run comparison on the current instrument.** Fix the instrument, fork every A/B.

---

## Ranked changes

Ranked by expected value per unit of implementation effort in this codebase.

### 1. Add a real end-of-run learning-rate decay

**Claim.** Every successful AZ-style run ends with a large LR reduction, and ours does not. AlphaZero chess:
LR 0.2, "dropped three times (to 0.02, 0.002 and 0.0002)" across 700 k steps — a 1000x total reduction
([Silver et al. 2017, §Methods](https://arxiv.org/abs/1712.01815)). Leela Chess Zero does the same at the end
of every test run: T79 went 0.04 → 0.004 → 0.0004 → 0.00004 within ~1100 steps of the run's end; T77, T76 and
T70 are identical in shape ([lc0 project history](https://lczero.org/dev/wiki/project-history/)). KataGo runs
an essentially flat per-sample LR (6e-5) for 19 days and drops to 6e-6 only "for final tuning"
([Wu 2019, §Training](https://arxiv.org/abs/1902.10565)). Our v28 goes 0.005 → 0.0015 over 1500 generations:
a 3.3x reduction, spread evenly, with nothing at the end.

**Evidence.** (B) for the practice — it is universal across three independent projects. (A) is absent: no
one ablates the final drop, because no one runs without it. Effect size is not published anywhere I could
find; lc0 folklore puts the terminal drop at "around +100 Elo" but I could not source that to a measurement
and do not rely on it.

**On the user's reading of the AlphaZero chess curve.** I could not verify it. The preprint's Figure 1 plots
Elo against thousands of training steps but the underlying values are not tabulated in the text, and text
extraction cannot read the plot. What is verifiable: the schedule is three drops, not four (secondary
sources including ELF OpenGo's description propagate a "100k/300k/500k/700k" four-drop version that the
AlphaZero text does not support), and chess passed Stockfish by ~300 k steps. Attributing upticks in that
curve to LR drops specifically is inference — the curve also reflects a monotonically growing and improving
self-play distribution, so drop-timing and data-quality effects are confounded in the published figure.

**Interaction with our bottleneck.** Neutral-to-positive. A late LR drop converts an already-plateaued
network into a lower-variance one on the same data; it does not need fresh data. It is the classic response
to "flat for the last two days".

**Caveat that is genuinely ours.** All the schedules above are SGD+momentum. AdamW normalises gradient scale,
so the *shape* transfers but the *magnitudes* do not, and the interaction with our `warmup_optimizer_steps:
1000` and `max_grad_norm: 0.5` is untested. I found no AZ-style run that publishes an AdamW LR schedule.
Treat the drop factor as a thing to sweep on a fork, not a thing to copy.

**Cost.** Config only: extra `training.trainer.learning_rate.stages` entries. Under an hour.

### 2. Weight training samples by how surprising the search result was

**Claim.** Not all of our 8 presentations of a position are equally informative, and the ones worth keeping
can be identified cheaply. Three independent projects converged on variants of this:

- **KataGo, policy surprise weighting.** Roughly half the total frequency weight is spread uniformly and
  half is distributed proportionally to the KL divergence from the policy prior to the policy training
  target. lightvector calls it "one of the larger improvements in KataGo's training between its g170 run and
  earlier runs" ([KataGoMethods.md](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md)).
  **(B), no controlled number given.**
- **Leela Chess Zero, value focus.** Positions are included in training with a probability derived from the
  Q-delta (search value minus network value), with `value_focus_min` / `value_focus_max` / `value_focus_slope`
  bounding the tactical bias. Introduced in T60 and used continuously from T76 onwards
  ([project history](https://lczero.org/dev/wiki/project-history/)). A community sweep found
  `value_focus_max ≈ 0.46` best at 1500 nodes but could not distinguish arms at 10 000 nodes
  ([lczero-training wiki](https://github.com/hans-ekbrand/lczero-training/wiki)). **(B), weak.**
- **Regret-Guided Search Control (RGSC), ICLR 2026.** A learned regret network scores states by how far the
  agent's evaluation diverges from the outcome; high-regret states go into a prioritised buffer and are
  reused as self-play start states. **+77 Elo over AlphaZero and +89 Elo over Go-Exploit** averaged over
  9x9 Go, 10x10 Othello and 11x11 Hex; win rate against KataGo on 9x9 Go rises 69.3% → 78.2% where the
  baselines do not move ([Tsai et al. 2026](https://arxiv.org/abs/2602.20809)). **(A).**

**Confidence.** Medium-high that *some* form of surprise/error prioritisation pays; low on which form and on
the size in chess. Three projects, two games families, one controlled ablation.

**Interaction with our bottleneck.** This is the best-matched idea in the review. It does not need more
fresh data; it reallocates gradient budget across data we already have. At `replay_ratio: 8` the marginal
value of the 8th presentation of a quiet, unsurprising position is close to zero, and that is exactly what
the weighting removes.

**Cost.** Moderate. Policy-surprise weighting is the cheapest form: KL(policy target ‖ network prior) is
already computable at materialization, and `duplicate_multiplicity_weight_cap` / `primary_sample_weight`
show the sample-weight plumbing exists. Reusing high-regret states as start positions is nearly free — our
`start_position.kind: restart_state` already implements the Go-Exploit archive; RGSC changes only the
*scoring* used to admit and rank candidates (today: `candidate_visit_mass`, `maximum_absolute_root_value`).
Estimate 1–2 days for the weighting, 1 day for the restart-state scoring change.

### 3. Keep growing the replay window past 5 M

**Claim.** Our window caps at 5 M positions from generation 700 onward. KataGo grows its window sublinearly
without a cap for the whole run: `N_window = c(1 + β((N_total/c)^α − 1)/α)` with `c = 250 000`, `α = 0.75`,
`β = 0.4`, taking the window from 250 k samples to about 22 M by the end of the main run
([Wu 2019](https://arxiv.org/abs/1902.10565)). ELF OpenGo used a flat 500 000-game buffer, matching AlphaGo
Zero ([Tian et al. 2019](https://arxiv.org/abs/1902.04522)).

**Evidence.** (B). The formula is published and is the default in the most sample-efficient known
reimplementation, but no one ablates window size in isolation.

**Interaction with our bottleneck.** A larger window does not create fresh data, but it reduces how
concentrated our 8 presentations are on the most recent generations, which is the mechanism most likely to
produce the late-run flatness we observed. Note the distinction, which is easy to conflate: **`replay_ratio`
sets reuse; window size sets staleness and diversity.** They are independent knobs and only the first is
capped by KataGo's `-max-train-bucket-per-new-data`.

**Risk.** More staleness. KataGo's window is sublinear precisely to bound this. Our window at 5 M against
~128 M total distinct positions is ~4%; KataGo's 22 M against its total is a comparable fraction, so we are
not obviously wrong today — the argument is about the *shape* after generation 700, where ours goes flat and
KataGo's keeps growing.

**Cost.** Config only: extend `lifecycle.replay.capacity.stages` and raise `maximum_capacity`. Bounded by
host RAM, which is the real constraint (`minimum_ram_gib: 200`).

### 4. Reconsider `replay_ratio: 8` — but measure it, do not assume

**Claim.** Our reuse factor is double the reference implementation's default. KataGo's documented default is
`-max-train-bucket-per-new-data 4` — "4 training steps (measured in rows or samples, not batches) per data
row generated by selfplay" — described as "conservative, you can increase it to train more/faster", with the
overall guidance to "spend anywhere from 4x to 40x more GPU power on the selfplay than on the training"
([SelfplayTraining.md](https://github.com/lightvector/KataGo/blob/master/SelfplayTraining.md)). ELF OpenGo
reports a selfplay:training ratio of about 13:1, against AlphaZero's 30:1 and AlphaGo Zero's 7:1
([Tian et al. 2019](https://arxiv.org/abs/1902.04522)). Our 8x self-play/train GPU split is *shared* — the
same cards do both — which is far outside every one of those regimes.

**On the annealed-replay-ratio hypothesis.** The user's hypothesis (high reuse helps early, hurts late) is
plausible and I found **no direct evidence for or against it** in either the AZ literature or the
off-policy RL literature. What exists nearby:

- High replay ratios are known to degrade learning through overfitting and plasticity loss, and periodic
  parameter resets recover the ability to scale them
  ([Nikishin et al. 2022, primacy bias](https://proceedings.mlr.press/v162/nikishin22a/nikishin22a.pdf);
  [D'Oro et al. 2023](https://openreview.net/forum?id=OpC-9aBBVJe)). **(A)**, but on Atari 100k and DM
  Control with tiny interaction budgets and a *fixed* task — not a self-play system whose data distribution
  is generated by the network being trained.
- Zhang and Sutton's observation that raising the replay ratio past a modest threshold degrades performance
  is the standard citation for the downside. **(A)**, again outside self-play.
- KataGo's *growing window* already anneals reuse implicitly: with reuse capped and the window growing, the
  expected number of times any given row is drawn falls over the run. **(C):** this may be the whole
  mechanism the hypothesis is reaching for, obtainable via change 3 without touching `replay_ratio`.

**Recommendation.** Do not change `replay_ratio` blind. Fork an A/B at 8 vs 5 from a shared checkpoint. Note
the coupling that makes this non-obvious: at fixed self-play throughput, lowering `replay_ratio` **lowers**
the permitted optimizer-step rate, so the arm trades steps for distinct data. That is the actual experiment
worth running, and it is the one experiment in this document that directly interrogates our stated
bottleneck.

**Cost.** Config only for the arms; the value is in the run time, not the code.

### 5. Do not bother raising `value_loss_weight`; consider raising the policy weight instead

lc0's T60 raised `value_loss_weight` to 0.8 and then 1.6 (project history, **(B)**, no reported effect).
We sit at 1.0/1.0. **(C):** the argument for raising the *value* weight is a Go argument — one binary label
per ~250-move game. In chess with our 150–250-ply cap, value labels are comparatively abundant while good
policy targets cost ~600 searches each. If either weight moves, the case is stronger for the policy side.
This is low-confidence tuning; it is on the list only because the opposite change is a tempting cargo-cult.

### 6. Gumbel root search — only as a route to much lower visit counts

See the visits section. Summary: at our visit counts it is a wash, and its payoff is in self-play compute
that our scheduler does not convert well. High implementation cost (native search). Recommend as a measured
fork experiment if and only if we decide to attempt a large visit reduction.

### 7. Already implemented — no action

For completeness, since these dominate any "what should an AZ run do" list and v28 already has them:
global pooling (KataGo ablation **1.60x speedup**, the largest single non-Go-specific item in Table 2),
forced playouts and policy target pruning (**1.25x**), auxiliary policy targets (**1.30x**, ours is
`next_policy` at weight 0.15), FPU reduction, calibrated resignation, restart states, tree retention,
staged visit counts, opening diversity via restart states. All figures from
[Wu 2019, Table 2](https://arxiv.org/abs/1902.10565), **(A)**.

Two gaps against that list are Go-specific and not portable: auxiliary ownership and score targets (**1.65x**)
and the Go-specific input features (**1.55x**). Our chess analogue of the latter is the input representation,
where [Czech et al. 2023](https://arxiv.org/abs/2304.14918) report that improved feature representation in
AlphaZero outperforms switching to transformers — I did not verify their effect sizes and flag it only as a
pointer.

---

## The playout-cap-randomization objection

### What the mechanism actually is

From [Wu 2019, §3.1](https://arxiv.org/abs/1902.10565):

- Two caps, a full cap `N` and a fast cap `n`. With probability `p` a move gets the full search, otherwise
  the fast one. KataGo used `p = 0.25`, `(N, n) = (600, 100)` initially, annealing to `(1000, 200)` after
  about two days.
- **Only turns with a full search are recorded for training.** Fast-search moves contribute *no* policy
  target and *no* value-head training row. They contribute only by being moves in a game that eventually
  produces an outcome, and by being cheap.
- On fast searches, Dirichlet noise and other explorative settings are **disabled**, maximising playing
  strength on those moves.
- Motivation, in the paper's own framing: "the game outcome value target is highly data-limited, with only
  one noisy binary result per entire game", while good policy targets need many playouts. Fast moves buy
  more *games* per unit compute (hence more value labels) at a small cost in policy samples per unit
  compute, because `n ≪ N`.
- Ablation: removing it drops the run from Elo 1329 to 1242, a **1.37x** compute speedup attributed to the
  technique, and Figure 5 shows it beating a sweep of fixed playout counts N ∈ {100, 150, 200, 250, 600}.
  **(A)**, single seed per arm as far as the paper states, on 19x19 Go.

Arithmetic for our numbers: at `p = 0.25`, `(N, n) = (600, 100)`, mean visits per move is
`0.25·600 + 0.75·100 = 225` — a **62% cut in search** and a **75% cut in policy targets per game**.

### Is the objection right?

**Yes, and there is a second, stronger reason it fails for us that the objection does not mention.**

**On the stated objection (game length / value-target abundance): supported, and by a chess source.**
Johannes Czech implemented and evaluated playout cap randomization for crazyhouse in his 2019 TU Darmstadt
master's thesis and rejected it. Verbatim, §5.4: "reaching a high amount of games seems not be an issue for
crazyhouse. Playout cap random both reduced the generation speed for training samples and likely increased
the noise for the value target due to an increasing blunder rate. **It appears to be more beneficial in less
tactical game types with a higher average game length.**"
([Czech 2019](https://ml-research.github.io/papers/czech2019deep.pdf), p. 35). This is **(B)** — a brief,
uncontrolled evaluation in a chess *variant*, not chess, and it is the only chess-family evaluation of PCR I
could find. It states exactly the user's objection, plus a second failure mode the objection does not
anticipate: fast searches play weaker moves, so the *value* labels PCR is supposed to improve get noisier
from the increased blunder rate.

I searched for a Leela Chess Zero evaluation of PCR specifically and found none. lc0 solved the same
"spend less search on obvious moves" problem differently, with KLD-gain-based adaptive visits (below), and
their public logs never mention playout cap randomization. Absence of evidence, but the absence is
informative given how thoroughly lc0 mines KataGo's other ideas.

**The second reason, which is ours alone.** PCR's entire payoff is denominated in self-play compute. Our
fork experiment measured that currency at roughly a fifth of face value: a 14% search cut yielded a 3%
cadence gain and no Elo movement at ±10 Elo. Linearly extrapolated (**(C)**, and the extrapolation is
generous), PCR's 62% search cut buys ~13% cadence — against a 75% reduction in policy targets per game and
Czech's noisier value labels. That is a bad trade at any plausible effect size.

**Where the objection is *not* quite right.** The mechanism is not "we waste 15–20% of search on moves that
produce nothing" — at KataGo's parameters the fast moves consume `0.75 × 100 / 225 = 33%` of search and
produce nothing. The waste is larger than the objection assumes, and the technique's defence is that those
moves buy game *count*, which is genuinely valuable in Go where a game is 250+ moves and one label. Our
150–250-ply cap and abundant early terminations mean we already get a value label per ~100–150 positions,
not per 250+, and our early-termination path already forces a full search at the cut position to produce
the value.

**Recommendation: do not implement playout cap randomization.** If we ever want the compute back, take it
where our own evidence says it is cheap — the paused-worker structure — not by degrading policy targets.

**One partial version worth keeping in mind.** KataGo has an experimental "reanalyze" option: after a game
finishes, a random subset of the cheap-search positions is re-searched fully and recorded as training data,
favouring positions where the cheap search was surprising
([KataGo docs](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md)). That is PCR without
the policy-target loss, and it composes with change 2. I found no reported effect size for it; it is listed
as experimental. **(B), unquantified.** Not recommended for the next run, but it is the shape of the idea
that survives the objection.

---

## Visits per move in self-play

### What the field actually uses

| System | Game | Self-play visits/move | Source |
|---|---|---|---|
| AlphaZero | chess | 800 | [Silver et al. 2017](https://arxiv.org/abs/1712.01815) |
| AlphaGo Zero / ELF OpenGo | 19x19 Go | 1600 | [Tian et al. 2019](https://arxiv.org/abs/1902.04522) |
| KataGo | 19x19 Go | 600→1000 full, 100→200 fast, p=0.25 | [Wu 2019](https://arxiv.org/abs/1902.10565) |
| Leela Chess Zero | chess | ~800 average, KLD-adaptive, 1 to 10 000 | [lc0 project history](https://lczero.org/dev/wiki/project-history/) |
| MiniZero (AlphaZero arm) | 9x9 Go, Othello | 200 | [Wu et al. 2023](https://arxiv.org/abs/2310.11305) |
| Gumbel AlphaZero | chess | 400 | [Danihelka et al. 2022](https://iclr.cc/virtual/2022/spotlight/6419) |

Our staged 300→1200 sits inside this band throughout. There is no source that says 600 is wrong.

### What the controlled evidence says about the trade

- **Below ~100–200 visits, standard PUCT breaks.** Danihelka's Figure 5.2 (9x9 Go, Elo against a Pachi-10k
  anchor, 800-simulation evaluation for all arms): "MuZero fails to learn from 16 or fewer simulations",
  while Gumbel MuZero "learns reliably even with 2 simulations". On ms_pacman, "MuZero fails to learn from
  4 or fewer simulations". **(A)**, 2 seeds on Go, 10 on Atari. An independent reproduction on Tablut
  reports the same wall from the other side: reducing simulation count below 128 "degraded both search
  quality and the resulting policy targets"
  ([Reproducing AlphaZero on Tablut](https://arxiv.org/abs/2604.05476), **(B)**, single project).
- **Above that wall, Gumbel buys nothing.** On the two large-scale experiments — 19x19 Go and **chess**, both
  at n = 400 — the claim is parity, not improvement: for Go the text says Gumbel MuZero "reaches or exceeds"
  MuZero's performance, and the chess panel (Figure 5.4b) is presented the same way, with no separate claim
  of a gain. The paper's own framing is that Gumbel "matches" the state of the art on chess and Go.
  On 9x9 Go at n = 200 the curves overlap. **(A).** The Gumbel result is a fix for a low-simulation
  failure mode, not a general improvement.
- **Equal-time comparisons favour fewer, cheaper searches — in one game.** MiniZero trained Othello arms for
  ~5 hours each: AlphaZero n=200 at ~378 s/iteration, Gumbel AlphaZero n=16 at ~41 s, n=2 at ~23 s, and
  found "g-α₀ n=2 and α₀ n=200 achieve similar playing strength". On 9x9 Go the same paper finds AlphaZero
  n=200 ahead of the Gumbel n=2 and n=16 arms for most of training. **(A)**, and note the two games
  disagree — MiniZero explicitly flags that Othello's n=2 ≈ n=16 result "contrasts sharply with Go".
- **Doubling rollouts is worth real strength at fixed weights.** ELF OpenGo reports approximately **200 Elo**
  from doubling rollouts, which they read as evidence that model capacity, not search, was their limit.
  **(A)**, but this is *evaluation-time* rollouts, and does not license a claim about training-time visits.
- **lc0's answer is per-position, not global.** KLD-gain adaptive visits, introduced in T50, aborts a search
  when the root visit distribution has stopped moving; the threshold was tuned run by run (T76 stepped
  340 → 170 → 80 → 60 → 40 micronats). Average stays around 800 with a 10 000 ceiling. **(B).** We already
  tried and retired the equivalent idea twice, on our own Elo evidence.

### Reading our own 600-vs-1000 result

Correctly treated as unreliable — a 64-search ladder that understates by ~640 Elo and saturates cannot
resolve a difference this size. But note what the literature predicts: everything above says 600 and 1000 are
both comfortably above the failure wall and on a flat part of the curve, and the equal-time argument for
cutting visits is game-dependent and unproven in chess. **The most likely truth is that our result is
correct for the wrong reasons.**

### Recommendation

Do not move visits in the next production run. If it is measured, measure it properly: fork three arms
(400 / 600 / 900) from a shared checkpoint, on a ladder that does not saturate, and read the *cadence* as
well as the Elo — our scheduler's poor conversion of self-play savings means a visit cut may not even
produce the throughput it appears to. Only if that experiment shows a large win from fewer visits does the
Gumbel root become worth its native-code cost, and only then in the n ≤ 100 regime where it has an edge.

---

## Do not bother

- **Playout cap randomization.** Section above.
- **Any further adaptive per-position search budget.** Two of our own attempts are closed negative, and the
  external record does not contradict that: nobody outside lc0 reports Elo from KLD-style stopping, and
  lc0's own tuning history is a sequence of threshold adjustments with no published effect. Our scheduler
  structurally cannot pay for it.
- **Periodic network resets / plasticity interventions (primacy bias, SR-SPR, BBF).** Strong **(A)** results
  — but on Atari 100k and DM Control, at replay ratios of 8–32 against a *stationary* task with a tiny fixed
  interaction budget. In self-play the network being reset is also the data generator, so a reset destroys
  the self-play distribution for as long as it takes to recover, and the reset literature's own known side
  effect is "periodic collapses in performance immediately after resets". Neither KataGo nor lc0 nor ELF
  does this. High risk, no in-domain evidence.
- **Prioritized experience replay by TD error.** [Fedus et al. 2020](https://arxiv.org/abs/2007.06700) is the
  standard citation for uniform sampling from a large enough buffer matching or beating PER on large-scale
  Atari at lower complexity; I took this from secondary descriptions and did not read the paper's tables, so
  treat the direction as reliable and the size as unverified. This does **not** argue against change 2: policy-surprise weighting is a fixed weight computed
  once from the search that produced the row, not a moving priority recomputed from a changing network.
- **Switching the optimizer to SGD+momentum to inherit AlphaZero's schedule.** The published schedules do not
  transfer in magnitude anyway, and we would give up bfloat16-friendly AdamW behaviour to chase a shape we
  can implement directly.
- **Auxiliary ownership / score targets.** KataGo's largest ablation number (1.65x) and entirely Go-specific.
- **MuZero.** Danihelka is explicit: "We train AlphaZero on chess, because AlphaZero learns faster than
  MuZero on chess." Learning a model buys nothing where the rules are free.
- **Copying lc0's `value_loss_weight: 1.6`.** See change 5.

---

## Proposed configuration delta for the next run

Ordered by expected value per unit of implementation effort. Every one of these should be validated by a
**forked paired A/B**, never by an independent run — 41 Elo of independent-run noise swallows all of them.

| # | Change | Concrete parameters | Expected effect | Risk |
|---|---|---|---|---|
| 1 | Terminal LR decay | Append `learning_rate.stages` at ~90% and ~97% of planned generations, dropping 5x then 5x again (e.g. 0.0015 → 0.0003 → 0.00006). Sweep the factor on a fork. | Recovers some of the last-two-days flatness. Unquantified but universal practice. | Low. Worst case the run stops improving slightly earlier. AdamW magnitudes are untested — this is why it is sweep-first. |
| 2 | Uncap the replay window | Extend `lifecycle.replay.capacity.stages` past generation 700 on a sublinear curve; raise `maximum_capacity` to whatever 200 GiB of host RAM permits. | Less concentration of the 8 presentations on recent generations; the mechanism most likely behind late flatness. | Low-moderate. Staleness, and host RAM is the hard limit. |
| 3 | Policy-surprise sample weighting | New per-sample weight at materialization: `w = 0.5 + 0.5·normalised KL(policy_target ‖ network prior)`. Reuse the existing sample-weight plumbing. | Best match to our bottleneck: reallocates gradient budget away from the uninformative 8th presentation. KataGo calls it one of its larger gains. | Moderate. Uncontrolled source; a badly normalised KL could bias training toward noisy positions — cap the weight. |
| 4 | Regret-scored restart states | Change the admission/ranking in `start_position.restart_state` from visit-mass to \|search value − network value\|; keep `maximum_absolute_root_value` as the sanity bound. | RGSC reports +77 Elo over AlphaZero on three games; we already have the archive machinery. | Moderate. Their result is 9x9 Go / Othello / Hex; the tactical-position bias that lc0's `value_focus_max` exists to bound is a real failure mode — bound it the same way. |
| 5 | Fork A/B: `replay_ratio` 8 vs 5 | Two arms from one checkpoint, ≥60 generations each. | Directly interrogates the stated bottleneck. KataGo's default is 4. | Low as an experiment; the arms differ in optimizer-step rate as well as reuse, so read cadence and Elo together. |
| 6 | Fork A/B: visits 400 / 600 / 900 | Three arms, shared checkpoint, non-saturating ladder. | Replaces an unreliable prior result with a real one. | Low as an experiment. Expect a null. |
| 7 | Gumbel root search | Only if 6 shows a large win from fewer visits, and then targeting n ≤ 100. Native search change. | The only known way to run below ~128 visits without breaking policy targets. | High cost, and at our current visit counts the published chess result at n=400 is a wash. |

Prerequisite to all of it: **a ladder instrument that does not saturate and does not understate by 640 Elo.**
Six of the seven changes have plausible effect sizes below the current instrument's resolution.

---

## Sources

- Silver et al., *A general reinforcement learning algorithm that masters chess, shogi and Go through
  self-play* (AlphaZero) — https://arxiv.org/abs/1712.01815
- Wu, *Accelerating Self-Play Learning in Go* (KataGo) — https://arxiv.org/abs/1902.10565 ·
  HTML: https://arxiv.org/html/1902.10565v5
- KataGo methods documentation —
  https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md
- KataGo self-play training guide —
  https://github.com/lightvector/KataGo/blob/master/SelfplayTraining.md
- Tian et al., *ELF OpenGo: An Analysis and Open Reimplementation of AlphaZero* —
  https://arxiv.org/abs/1902.04522
- Danihelka, Guez, Schrittwieser, Silver, *Policy improvement by planning with Gumbel*, ICLR 2022 —
  https://iclr.cc/virtual/2022/spotlight/6419 · full text and figures read from the author's thesis,
  *Planning and Policy Improvement*, ch. 5 — https://discovery.ucl.ac.uk/id/eprint/10167022/2/ivo_danihelka_thesis.pdf
  · code: https://github.com/google-deepmind/mctx
- Wu et al., *MiniZero: Comparative Analysis of AlphaZero and MuZero on Go, Othello, and Atari Games* —
  https://arxiv.org/abs/2310.11305
- Trudeau & Bowling (Go-Exploit), *Targeted Search Control in AlphaZero for Effective Policy Improvement* —
  https://arxiv.org/abs/2302.12359
- Tsai et al., *Regret-Guided Search Control for Efficient Learning in AlphaZero*, ICLR 2026 —
  https://arxiv.org/abs/2602.20809
- Czech, *Deep Reinforcement Learning for Crazyhouse*, MSc thesis, TU Darmstadt 2019 —
  https://ml-research.github.io/papers/czech2019deep.pdf (playout cap randomization: §5.4, p. 35)
- Leela Chess Zero project history (per-run training changes) — https://lczero.org/dev/wiki/project-history/
- lczero-training value-focus experiments — https://github.com/hans-ekbrand/lczero-training/wiki
- Nikishin et al., *The Primacy Bias in Deep Reinforcement Learning*, ICML 2022 —
  https://proceedings.mlr.press/v162/nikishin22a/nikishin22a.pdf
- D'Oro et al., *Sample-Efficient Reinforcement Learning by Breaking the Replay Ratio Barrier*, ICLR 2023 —
  https://openreview.net/forum?id=OpC-9aBBVJe
- Fedus et al., *Revisiting Fundamentals of Experience Replay*, ICML 2020 — https://arxiv.org/abs/2007.06700
- Czech et al., *Representation Matters for Mastering Chess: Improved Feature Representation in AlphaZero
  Outperforms Switching to Transformers* — https://arxiv.org/abs/2304.14918 (pointer only, not verified)
- *Reproducing AlphaZero on Tablut: Self-Play RL for an Asymmetric Board Game* —
  https://arxiv.org/abs/2604.05476

### Sources I could not read

- The ICLR OpenReview PDFs for Danihelka et al. 2022 and D'Oro et al. 2023 are behind a browser check; the
  Gumbel numbers above come from the author's thesis chapter, which is the same work, and the D'Oro claims
  are taken from secondary descriptions and are flagged as such.
- No Leela Chess Zero evaluation of playout cap randomization exists in any public source I could reach.
  The lc0 Discord `test-results` channel is where such a result would live and is not fetchable.
