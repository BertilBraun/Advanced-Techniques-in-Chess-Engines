# Plateau investigation from V100 onwards

As of **2026-09-28**. One question runs through everything here: why does self-play training stop near 2,360
ladder Elo at 64 searches (checkpoint 1026, 14x160, 6.3M parameters), and what would lift it? This page
summarises every run since V100, what each one established, and what is still open. Details and raw numbers
live in the linked records.

## Short answer

Several candidate causes are now **ruled out**, two are **confirmed**, and what remains **open** is whether
self-play reaches the same level.

- **Ruled out:** the search and the evaluation harness; the learning-rate floor raised to 0.02; self-play label
  quality at 1,600 visits; replay diversity; model size alone (as far as V101 and the grown 19x176 can show).
- **Confirmed: the network is the limit.** A stronger network (Lc0's 20M-parameter T1) in our unchanged search
  plays at least 360 Elo above checkpoint 1026.
- **Confirmed: the architecture is the lever (update, 2026-09-28).** A student built like T1 at our compute
  (10x192 attention with smolgen, 366M MAC per position against 396M) trained from scratch on the 47M teacher
  labels beats the convolutional student trained identically by about 200 Elo, and passes checkpoint 1026 by
  about 150 Elo at 64 searches (0.68-0.695 at steps 90,000-110,000). No convolutional variant came close.
- **Cost:** served through TensorRT float16 it runs at 0.61x the CNN's positions per second after fusing its
  squared ReLU (0.55x before; 0.71x in self-play on TorchScript). INT8 buys the CNN 1.70x but the attention
  network only 1.12x (weight matmuls only), so with both in INT8 it serves at 0.40x the CNN's rate. At equal
  time that is about 2.5x fewer searches, worth roughly 75-185 Elo, against about 150 gained at equal searches.
- **Open:** whether self-play with this architecture, at its lower throughput, climbs past the convolutional
  plateau without a teacher. T1 itself is distilled from much larger networks, so its level is not a self-play
  target at this size. Our self-play also never anneals below a 0.01 learning rate, where Lc0 ends at 0.0005.

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
| 12 | T1-shaped student | 09-28 | T1's construction at 10x192, from scratch, the convolutional student's exact schedule, 110,000 steps | **About +200 Elo over the CNN student, +150 over checkpoint 1026** |
| 13 | Inference cost | 09-28 | Self-play and forward throughput, CNN against T1-shaped, float16 | 0.71x in self-play (TorchScript), 0.55x forward (TensorRT) |

Full numbers for 12 and 13: [Lc0 teacher diagnostic, architecture section](../benchmarks/lc0-teacher-diagnostic-rtx3070-20260926/README.md#architecture-a-student-built-like-t1-against-the-convolutional-student).

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

## Next experiment

[`vast-chess-8gpu-final-attention.yaml`](../../py/configs/production/vast-chess-8gpu-final-attention.yaml)
(sha256 `d039bf49777f4be12112feae0cbb03ac4d1ce4bde92f549e6d507df46db7fb61`) is the final recipe with:
- a two-stage ladder of T1-shaped networks, 8x160 (207M MAC per position) then 10x192 (366M); the second size is
  to be settled by a throughput comparison on the rented node (12x192 at 437M and 10x224 at 494M are the
  candidates);
- float16 TensorRT, INT8 QAT off;
- SGD as in the final recipe, at AlphaZero's 0.2 at batch 4,096 scaled to 0.1 at 2,048, held to generation 100
  and then decayed geometrically to 0.0005 by generation 1000 instead of holding a 0.01 floor;
- a generation every 402 optimizer steps with replay reuse 3 (was 500 and 4), generation-keyed schedules left as
  they are;
- the candidate trained from scratch and started at 20 Elo/h, with V98's catch-up floor (0.1 -> 0.03), and the
  progressive-candidate promotion match restored: the final recipe's evaluation list lacks it, so as written
  its promotion gate never receives a result.
Everything else is the final recipe, so the CNN lineage's ladder curve on the same hardware is the baseline.

Before it can start, on a node: build and refit-test the four float16 templates (both sizes at batch 320 and
64); the attention engines are Myelin-compiled and their refit has not been exercised. Build the production
network with its seven auxiliary heads and export it once. Rebuild the native extension with TensorRT.

## What is still open

1. **Does self-play with the T1-shaped network beat the convolutional plateau at equal time?** Answered only in
   distillation so far; the next experiment above is the test. It needs the network kind in the production configuration, a TensorRT float16 serving
   path for it (INT8 QAT does not exist for attention) and either a small attention-against-CNN A/B or a rerun
   of the final training run. The 0.55x serving rate costs about 1.8x fewer searches at equal time, worth roughly
   50-120 Elo against a gain of about 150 at equal searches.
2. **Does annealing the learning rate lift the convolutional plateau?** Lc0 steps down to 0.0005; our self-play
   holds 0.01, and every distillation run gained most while annealing. Testable by resuming checkpoint 1026 with
   a stepped-down rate.
3. **Can the attention network be served faster?** TensorRT gains 2.56x on the CNN but 1.49x on the attention
   network; the generated attention bias probably blocks fused attention kernels.
4. **Does the report's conclusion hold?** The attention-viability decision (8,000-step comparison plus
   throughput), "capacity was not the binding constraint" and the final recipe's convolutional trunk need
   revisiting once 1 has a first answer.

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
