# Plateau investigation from V100 onwards

As of **2026-09-28**. One question runs through everything here: why does self-play training stop near 2,360
ladder Elo at 64 searches (checkpoint 1026, 14x160, 6.3M parameters), and what would lift it? This page
summarises every run since V100, what each one established, and what is still open. Details and raw numbers
live in the linked records.

## Short answer

Several candidate causes are now **ruled out**, one is **confirmed**, and the main question is **still open**.

- **Ruled out:** the search and the evaluation harness; the learning-rate floor; self-play label quality at
  1,600 visits; replay diversity; model size alone (as far as V101 and the grown 19x176 can show).
- **Confirmed:** the network is the limit. A stronger network (Lc0's 20M-parameter T1) in our unchanged search
  plays at least 360 Elo above checkpoint 1026.
- **Open:** why our network family does not get there. Distilling the Lc0 network into it gained at most about
  60 Elo at 64 searches (not significant at 1,000 searches), and the architecture ablation so far only measured
  early learning speed, not the level each variant can reach.

## Timeline

| # | Run | Dates | What it tested | Result |
|---|---|---|---|---|
| 1 | V100 | 09-23 | Learning-rate floor 0.01 -> 0.02 and 1,600 self-play visits, resumed from checkpoint 1080 | Flat: 25 ladder points 2,280-2,372, mean 2,332, below the ~2,360 plateau |
| 2 | V101 | 09-24/25 | 19x176 from scratch, value head 32/128, FP16, `auto` exploration, extra auxiliary targets | Reached within 13 Elo of V89 by hour 17, then flat while V89 kept climbing |
| 3 | Lc0 Phase A | 09-26 | Lc0's network inside our unchanged search | Teacher far stronger at every budget |
| 4 | Distillation pilot and anchored run | 09-26 | Fine-tune 1026 on 2M search-free teacher positions | No gain; searched play slightly worse |
| 5 | From scratch on 2M | 09-26 | Train 1026's architecture on teacher labels only | Very weak (0.065) |
| 6 | Data generation | 09-26 | 47M teacher-labelled positions, half from Stockfish-played games | 54.9 GB dataset |
| 7 | From scratch on 47M | 09-26/27 | Same architecture, fresh weights, 130,000 steps | Levelled off about 90 Elo below 1026 |
| 8 | Fine-tune 1026 on 47M | 09-27 | 1026 plus teacher labels, 60,000 steps | First student level with or above 1026 at 64 searches |
| 9 | Fine-tune continuation | 09-27 | 40,000 more steps at a re-raised learning rate | Same level; one 1,000-search point at 0.6125 |
| 10 | Grown 19x176 | 09-27 | Run 8 grown function-preserving to 10.2M parameters, 80,000 steps | Better fit to the teacher, same strength |
| 11 | Architecture ablation | 09-27/28 | Eight from-scratch variants, SGD, 30,000 steps each | Only the policy head matters for early learning |

## Results

All matches are against Stockfish 13 with this project's search and settings, served in float32 TorchScript,
100 games (40 at 1,000 searches). A 100-game score carries a 95% interval of about ±0.08, roughly ±55 Elo near
0.5; a 40-game score about ±0.1. Held-out gap = policy cross-entropy above the teacher's own entropy on the
same 470,000 held-out positions (lower means closer to the teacher).

### Reference points

| Network | Policy only, 2,000 nodes | 64 searches, 10,000 nodes | 1,000 searches, 50,000 nodes | Held-out gap |
|---|---|---|---|---:|
| Lc0 T1 teacher (20.2M, attention) | 0.855 | 0.885 | 0.8625 | 0 by definition |
| Checkpoint 1026 (6.3M, self-play) | 0.395 | 0.485 | 0.5625 | 0.2260 |

The teacher's scores sit close to the ceiling, so its 360 Elo lead at 64 searches is a lower bound.

### Distilled students

| Student | Steps | Policy only | 64 searches | 1,000 searches | Held-out gap |
|---|---:|---|---|---|---:|
| Pilot (1026 + 2M, policy and value) | 3,000 | 0.405 | 0.38 | 0.35 | — |
| Anchored (1026 + 2M, value held to 1026) | 10,000 | 0.325 | 0.42 | — | — |
| From scratch, 2M | 10,000 | 0.045 | 0.065 | — | — |
| From scratch, 47M | 30,000 | 0.16 | 0.275 | — | 0.1527 |
| From scratch, 47M | 130,000 | 0.315 | 0.36 | — | 0.1159 |
| 1026 fine-tuned, 47M | 60,000 | 0.445 | **0.57** | 0.425 | 0.0879 |
| ... continued (re-raised learning rate) | +40,000 | — | 0.48 | **0.6125** | 0.0912 |
| Grown 19x176 from the 60,000-step fine-tune | +80,000 | 0.46 | 0.56 | 0.5125 | 0.0806 |

- **Distillation onto 1026 helps a little at low search.** The best students score 0.56-0.57 at 64 searches
  against 1026's 0.485, about +50 to +60 Elo; the intervals touch. At 1,000 searches the students range from
  0.425 to 0.6125 against 1026's 0.5625: no demonstrated gain.
- **Imitation loss and strength are loosely coupled.** Closing the gap from 0.226 to 0.088 was worth about 60 Elo,
  yet the teacher's raw policy alone is about 290 Elo stronger than the students'. Most of the teacher's
  strength sits in the part of the loss the students never close.
- **Teacher labels alone do not reach self-play strength.** From scratch, 47M teacher-labelled positions trained
  the same architecture to 0.36, about 90 Elo below 1026, with training and held-out loss 0.04 apart (no
  overfitting).
- **More size fits the teacher better but does not play better.** The 10.2M 19x176 reached a gap of 0.0806 against
  the 14x160's 0.0879, with no measurable change in strength. That agrees with V101.
- **Top-move agreement** with the teacher on match positions: 1026 0.595, anchored student 0.640, 2M scratch
  student 0.514.

### Architecture ablation (early learning)

From scratch on the 47M dataset, production's optimizer (SGD, Nesterov 0.9, weight decay 1e-4, clip 1.0),
learning rate 0.05 falling linearly to 0.005 at batch 1,024, 30,000 steps. One change per variant against the
14x160 baseline.

| Variant | Parameters | Held-out gap | 64 searches |
|---|---:|---:|---|
| 1 Baseline (scaled post-activation, ReLU6, global pooling, from-to policy, value 2/48) | 6.26M | 0.2123 | 0.11 |
| 6 Value head 32/128 | 6.52M | 0.2116 | 0.11 |
| 7 Dense policy head | 6.69M | **0.2665** | **0.03** |
| 5 Scaled pre-activation | 6.26M | 0.2141 | 0.09 |
| 4 Plain post-activation (AlphaZero block) | 6.26M | 0.2113 | 0.10 |
| 2 10x192 (wide, shallow) | 6.45M | 0.2092 | 0.115 |
| 3 20x128 (deep, narrow) | 5.72M | running | — |
| 8 Global pooling off | 6.60M | running | — |

- Only the policy head changes early learning: the from-to attention head production uses is clearly better
  than a dense one. Value-head width, pre- versus post-activation, the plain AlphaZero block and width versus
  depth make no difference.
- **This does not measure the level each variant can reach.** At 30,000 steps the baseline plays at 0.11, where
  the same architecture reaches 0.36 from scratch with AdamW at 130,000 steps. SGD also learned much more slowly
  than AdamW here (gap 0.212 against 0.153 at the same step).

### Other measurements

- **Trunk activations** (2,048 positions): the ReLU6 cap is never reached in 1026 or the students, so it does not
  clip anything. 1026's trunk is sparse and very small in scale (mean 0.008-0.115 across blocks, 16 of 160
  channels dead after the first block) compared with a from-scratch student (0.22-0.32).
- **Teacher fidelity:** the wrapped Lc0 network reproduces real Lc0 to a worst prior difference of 0.00066, 0 of 50
  top-move disagreements, WDL within 0.0026.

## What was built

On branch `worktree-lc0-teacher`:

- Lc0 input encoding (112 planes, eight real board states) beside the project encoding, selected at build time;
  the Lc0 policy map; a TorchScript wrapper of the Lc0 network with a fidelity gate against real Lc0.
- A dataset builder labelling positions with the teacher, including a mode where Stockfish chooses the moves; a
  merger; memory-mapped datasets larger than RAM (`MADV_RANDOM`).
- Student trainer options: continue from a checkpoint, value anchoring, architecture from a checkpoint, residual
  block and pooling choice, a linear learning-rate schedule.
- `grow_student_checkpoint.py`: function-preserving growth of a distilled student.
- A fix that lets TorchScript-served checkpoints be matched even when their architecture is not a configured model.

## What is still open

1. **Can any variant of our network family reach a higher level?** Needs long runs, not 30,000-step ones. The
   14x160 from-scratch AdamW run (130,000 steps, gap 0.116, 0.36) is a converged baseline, so a variant trained
   on the identical schedule and compared at the same checkpoints answers this without re-running the baseline.
   About 14 hours for two variants in parallel on the current node.
2. **Does self-play hold a distilled start?** Seeding self-play with the fine-tuned 1026 is the only test of the
   training loop itself. It needs the production setup.
3. **Why does the Lc0 network get so much more out of 20M parameters?** Untested. Its body is attention-based;
   our attention student has only a simple input embedding and was not trained here.

## Evidence

Under `.codex-diagnostics/lc0-teacher-diagnostic-20260926/` in the main checkout: `evidence-final.tgz`
(Phases A and B), `evidence-scratch.tgz`, `evidence-large.tgz` (47M from scratch, 30,000 steps),
`evidence-finetune.tgz` (continuation to 130,000 and the 60,000-step fine-tune), `evidence-grown-partial.tgz`
(fine-tune continuation, grown run to step 50,000), `student-weights-20260927.tgz`. The grown run's completion
and the ablation are still on node B and must be fetched before it is released. V100 and V101 evidence is under
`.codex-diagnostics/final-2026-09-23/`.

Detailed records: [Lc0 teacher diagnostic](../benchmarks/lc0-teacher-diagnostic-rtx3070-20260926/README.md),
[V101 capacity run](../benchmarks/chess-v101-capacity-rtx4070s-20260925/README.md),
[final chess run](../results/final-chess-run.md).
