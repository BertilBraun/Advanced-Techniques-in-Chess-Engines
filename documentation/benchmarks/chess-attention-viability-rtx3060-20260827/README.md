# Chess attention viability — rtx3060, 2026-08-27

Can an attention trunk be made competitive with the convolutional one in this engine, now that the
generation-0 bootstrap defect that invalidated the two earlier attention runs is fixed?

**Yes, but the win is the policy head, not the trunk.** The best cell is an attention trunk with a
generated attention bias and a from-to policy head, 0.0406 nats better than the convolutional baseline and
comfortably past the 0.03-nat bar. But putting the same head on the *unchanged* convolutional trunk
recovers 0.0298 of that 0.0406 — with 11% fewer parameters than the baseline and 1.9% of its forward
throughput — while a bare attention trunk is 0.0060 nats **worse** than convolution at matched head.

**The actionable result is a policy head, not an architecture change.** `cnn-from-to-narrow` is production's
own 12x128 trunk with the dense head replaced: 0.0298 nats (+33 to +42 Elo), 3,451,655 parameters against
3,885,287, 51,072 head parameters against 483,680. Attention costs 64% of the forward throughput to add a
further 0.0108 nats on top, which is a much harder trade.

| | |
| --- | --- |
| `experiment_configuration_sha256` | `c1ac342fbb33e959258814474f9709d33752f3833c99981e590837a8fd910e2d` (`py/configs/validation/distillation-probe-chess-single-gpu.yaml`, file sha256 `8caaca0ff5e4fbd90fcb8d566b53d5ebb7d630166666599751dbc463f8c45f11`) — the throughput measurement only |
| Cell definitions | `py/tools/attention_viability_cells.py`; resolved architectures sha256 `9a5dc37a54791f35a41bcb03a4b29f62f5f2955d60789b63d96d9edb1f9b0f0c` |
| Source revision | branch `attention-viability`, based on `distillation-probe` `6c62e605`; training ran at `fb2a6bc0`, throughput at `bfaaeaaa` |
| Training node | Vast.ai `50.120.65.61:41841` — 1× RTX 3060 12 GiB, driver 595.84, 56 effective CPUs, 62 GiB RAM |
| Throughput node | `154.64.230.50:50623` — 1× **RTX 4070 SUPER** 12 GiB, driver 580.159.03, 80 effective CPUs, 188 GiB RAM |
| Runtime | Python 3.12.3, torch 2.12.1+cu126 |
| Dataset | `/workspace/distill/attention-6m.bin` — the distillation probe's 6,000,000-position chess set, teacher `vast-chess-4day-production-v8` generation 322, weights sha256 `dd514db199186f6e657593210751b7f6b1dce9b4e172129c9d2944e38b86f3df` |
| Openings (throughput) | `chess-elite-2025-11-balanced-4moves-200-v1`, sha256 `ab15a513135c7aec3e19c0ce3b845e03c2f47a9a73f07a72f5a9ab4842470c1e` |
| Date | 2026-08-27 |

## What bounds every number here

**Training ran for 8,000 steps, not the 80,000 the protocol specifies.** The RTX 3060 was shared with a
`measure_policy_target_fidelity` job from separate `search-evaluations` work from about 14:33 UTC onward,
and the specified budget was over thirty hours of wall clock. Every cell got the identical shortened
budget, so the comparison stays matched, but it measures quality at 8.2M training samples rather than at
convergence. For scale, the distillation probe's best student on this same dataset reached 0.1584 after
80,000 steps; `cnn-A` here reaches 0.1143 after 8,000, so the cells are not in a degenerate
under-trained regime — they are past the previous best on this data.

**Training throughput on the RTX 3060 is worthless and is not reported.** Between one and three cells
plus a foreign job shared the card at any moment, so the samples/s figures in `cells.json` vary by more
than 4× for reasons that have nothing to do with the architectures. Inference throughput was measured
separately on an idle RTX 4070 SUPER, which is the production card, and those numbers do transfer.

**One seed per cell.** The intervals below are the sampling uncertainty of the held-out set, not seed
variance. The distillation-probe branch's rule is that gaps under ~0.005 nats should not be ranked
without a second seed; the 0.0090 attn-C-versus-cnn-from-to margin is only modestly above that line,
while the 0.0406 headline is eight times it.

**The dataset carries two known defects**, both inherited from the probe that generated it and both
applying identically to every cell: every recorded position has the same side to move (an even ply
stride, fixed in the builder since), and the teacher is the v8 run with the documented endgame-conversion
defect, so absolute quality does not transfer — only the relative architecture comparison does.

## The cells

Matched on **total** parameter count including output heads, not on trunk parameters, because the dense
policy head is a flat ~483,680 parameters regardless of trunk and a from-to head is far smaller. Each
attention cell is sized by depth at a fixed embedding of 176 with head dimension 16, so the head and the
bias are the only things that change and the head savings buy trunk depth.

| cell | trunk | policy head | attention bias |
| --- | --- | --- | --- |
| `cnn-A` | 12x128 global-pooling convolutional | dense, 4 channels, unbottlenecked | — |
| `attn-A` | 14x176 attention, 11 heads, FFN 352 | dense, 4 channels, unbottlenecked | none |
| `attn-B` | 15x176 attention, 11 heads, FFN 352 | from-to attention, key size 128 | none |
| `attn-C` | 13x176 attention, 11 heads, FFN 352 | from-to attention, key size 128 | smolgen (8, 32, 32) |
| `cnn-from-to` | 12x136 global-pooling convolutional | from-to attention, key size 128 | — |

`cnn-from-to` was added mid-experiment. Once `attn-B` beat both `cnn-A` and `attn-A`, the question stopped
being "is attention weaker" and became "is the win the trunk or the head", and only a convolutional trunk
carrying the same head answers it.

## Results

### Held-out policy cross-entropy

Gap above the target-entropy floor. The training-log column is the trainer's own 8×1,024-row evaluation
slice; the paired column is a larger 32,768-row slice of the held-out tail, evaluated from the saved
checkpoints, and is the one the verdict uses. Intervals are 95% from a 10,000-sample paired bootstrap over
positions.

| cell | total params | trunk | policy head | trunk MAC | total MAC | gap (log) | gap (paired) | vs `cnn-A` | vs `cnn-from-to` | held-out WDL CE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `attn-A` | 3,995,783 | 3,505,216 | 483,872 | 242,547,712 | 243,102,864 | 0.2522 | 0.2387 | **+0.1318** [+0.1289, +0.1348] | +0.1634 | 0.4506 |
| `cnn-A` | 3,885,287 | 3,395,008 | 483,680 | 214,511,616 | 215,048,336 | 0.1143 | 0.1069 | — | +0.0316 | 0.4387 |
| `attn-B` | 3,817,847 | 3,754,960 | 56,192 | 259,849,216 | 263,945,360 | 0.0842 | 0.0814 | **−0.0255** [−0.0269, −0.0242] | +0.0060 [+0.0051, +0.0069] | 0.4356 |
| `cnn-from-to` | 3,887,651 | 3,829,964 | 51,072 | 242,021,520 | 245,784,864 | 0.0793 | 0.0753 | **−0.0316** [−0.0328, −0.0303] | — | 0.4368 |
| `attn-C` | 3,842,135 | 3,779,248 | 56,192 | 245,520,384 | 249,616,528 | **0.0689** | **0.0663** | **−0.0406** [−0.0420, −0.0392] | **−0.0090** [−0.0098, −0.0082] | 0.4342 |

Parameter spread across the five cells is 3,817,847 to 3,995,783, a 4.7% band. In a regime where 2.8× the
parameters bought ~0.05 nats, 4.7% is worth roughly 0.002 nats — two orders below the effects measured.

### Verdict against the 0.03-nat rule

| cell | margin over `cnn-A` | rule |
| --- | --- | --- |
| `attn-C` | 0.0406 [0.0392, 0.0420] | **passes**, whole interval above 0.03 |
| `cnn-from-to` | 0.0316 [0.0303, 0.0328] | **passes**, whole interval above 0.03 |
| `attn-B` | 0.0255 [0.0242, 0.0269] | **fails**, whole interval below 0.03 |
| `attn-A` | −0.1318 | fails catastrophically |

At roughly 1,100–1,400 Elo per nat, `attn-C` is +45 to +57 Elo over the baseline and `cnn-from-to` is +35
to +44. The dataset was 6,000,000 positions, which meets the stated minimum, so the verdict is not
disqualified for want of data — and the train/held-out separation of 0.018 on `cnn-A` confirms data was
not the binding constraint at this budget.

### Where the win actually comes from

Decomposing the 0.0406:

| mechanism | isolated by | worth |
| --- | --- | --- |
| from-to policy head | `cnn-A` → `cnn-from-to` (trunk held convolutional) | 0.0316 |
| from-to policy head | `attn-A` → `attn-B` (trunk held attention) | 0.1573 |
| generated attention bias | `attn-B` → `attn-C` | 0.0151 |
| attention trunk itself | `cnn-from-to` → `attn-B`, head held constant | **−0.0060** (convolution ahead) |

The head is the finding. It is worth 0.0316 nats on a convolutional trunk and 0.1573 on an attention one —
five times more — because the dense head compresses 64 squares × 176 dims through 256 floats before
predicting 1,880 moves, and a transformer whose entire representation is those per-square vectors has far
more to lose than a CNN whose features are already spatially arranged.

The two head gains also have opposite shapes over training. On the CNN the gain decays (0.2557 at step
500, 0.0371 at 7,500) — the dense head is a slow-start handicap the CNN largely trains out of. On
attention it grows and holds (−0.0432, then 0.1973 at 5,000, 0.1889 at 6,000) — the dense head is a
ceiling attn-A never escapes. **The convolutional head gain may therefore shrink further at the specified
80,000-step budget; the attention one probably will not.**

### The generation-0 bootstrap prior

Measured on 256 real encoded positions from the dataset, on each cell's generation-0 inference export.
Uniform over the 32-action probe subset is top-1 0.031 and top-3 0.094.

| cell | policy logit std | top-1 mass | top-3 mass | calibration scale | top-3 after |
| --- | --- | --- | --- | --- | --- |
| `cnn-A` | **1234** | 0.993 | 1.0000 | ×0.0354 | 0.950 |
| `attn-A` | **0.103** | 0.038 | 0.110 | ×94.5 | 0.950 |
| `attn-B` | 0.138 | 0.041 | 0.115 | ×54.5 | 0.950 |
| `attn-C` | 0.126 | 0.040 | 0.114 | ×68.7 | 0.950 |

This confirms the `33b9e115` diagnosis on real positions rather than on noise: the pre-fix attention
export was within 17% of literally uniform, so MCTS had nothing to break symmetry with, while the CNN's
accidental prior was a near-one-hot. `calibrate_bootstrap_policy_prior` brings all four to the same 0.95
target and `checkpoint_0.json` records it for every architecture — including *dampening* the CNN by 28×,
so the fix standardises both trunks rather than only sharpening attention.

**But the plateau was not purely that bug.** With the bug fixed and everything else held constant,
`attn-A` — the architecture family the two plateaued runs used — is still 0.1318 nats worse than the
convolutional baseline. The bootstrap defect was real and is fixed; it was not the only thing wrong.

### Learning rate and optimiser

All five arms use the `cnn-A` architecture and differ only in schedule and optimiser.

| arm | optimiser | peak | shape | gap | held-out WDL CE |
| --- | --- | --- | --- | --- | --- |
| `cnn-A` (reference) | AdamW | 0.002 | plateau, cosine over final 20% | **0.1143** | 0.4387 |
| `lr-production-flat` | AdamW | 0.005 | 0.005 → 0.004 → 0.003, 1.67× (production v9's) | 0.1228 | 0.4414 |
| `lr-cosine-floor` | AdamW | 0.005 | cosine to a 10× floor | 0.1300 | 0.4410 |
| `lr-staged-decay` | AdamW | 0.005 | staged 10× (the proposed v10) | 0.1463 | 0.4427 |
| `lr-sgd-alphazero` | SGD + momentum | 0.05 | AlphaZero's 1000× step shape | 0.5134 | 0.4789 |

AdamW trains cleanly at a 5×10⁻⁴ floor — no divergence, no instability, in any arm. At this horizon more
decay is monotonically worse, and the proposed v10 10× staged decay is 0.032 nats behind the near-flat
schedule production already runs.

**This experiment cannot tell you where the optimal decay point is in a self-play run, and the result
above must not be read as saying the v10 proposal is wrong.** Supervised distillation has a *stationary*
target distribution. Self-play does not: the thing that makes a decaying schedule pay — the target
distribution still moving under the learner — is absent here by construction. What this does establish is
that AdamW is numerically well-behaved at the proposed floor, and that on a stationary target, decay buys
nothing at this horizon.

Two further caveats on the SGD arm specifically. Its peak was never tuned — 0.05 is AlphaZero's 0.2
linearly scaled from batch 4096 to batch 1024 — and compressing a 700,000-step shape into 8,000 steps puts
its first tenfold drop at step 1,143, before the model has learned much. The arm faithfully renders
AlphaZero's schedule *shape* at a horizon it was never designed for, and its 0.5134 should be read as
"this shape does not survive compression", not as "SGD cannot train this network".

### The head against the trunk it was paid for with

`cnn-from-to` changed two things: the head, and a trunk widened 128 → 136 to spend the head's parameter
saving. `cnn-from-to-narrow` keeps cnn-A's exact trunk and changes only the head.

| cell | trunk | head | trunk params | total params | gap (paired) | vs `cnn-A` |
| --- | --- | --- | --- | --- | --- | --- |
| `cnn-A` | 12x128 | dense | 3,395,008 | 3,885,287 | 0.1069 | — |
| `cnn-from-to-narrow` | **12x128** | from-to | **3,395,008** | 3,451,655 | 0.0771 | **−0.0298** [−0.0311, −0.0285] |
| `cnn-from-to` | 12x136 | from-to | 3,829,964 | 3,887,651 | 0.0753 | −0.0316 [−0.0328, −0.0303] |

`cnn-dense-wide` (12x136 with the dense head) would complete the square and was deliberately not run;
the widening's contribution is therefore inferred from the other three corners, not measured directly.

**The head is 94% of the effect.** Holding the trunk at cnn-A's exact 12x128, the head alone is worth
0.0298 nats; widening to 136 adds only 0.0018 more, for 42% of the forward throughput. The head is not
buying its gain with the 435,000 parameters it freed — `cnn-from-to-narrow` is 11% *smaller* than cnn-A
in total and still 0.0298 ahead.

### Native inference and search throughput — RTX 4070 SUPER

**These are the production card, so unlike everything above they transfer.** Measured through the
evaluation search path with `parallel_searches: 1` and the memory-efficient SDPA backend, at 64 roots
(so the inference batch is 64) with 64 searches per move. `cnn-A` is the reference in every pair and is
therefore measured once per pair.

| cell | positions/s | searches/s | ratio to `cnn-A` | total MAC | MAC-implied ratio |
| --- | --- | --- | --- | --- | --- |
| `cnn-A` | **26,056** (25,316–26,701, n=12) | 25,655 | 1.000 | 215,048,336 | 1.000 |
| `cnn-from-to` | 20,537 (19,798–21,311) | 20,221 | 0.788 | 245,784,864 | 0.875 |
| `attn-A` | 20,155 (19,730–20,703) | 19,845 | 0.774 | 243,102,864 | 0.885 |
| `attn-B` | 18,943 (18,844–19,060) | 18,651 | 0.727 | 263,945,360 | 0.815 |
| `attn-C` | 18,532 (18,409–18,613) | 18,255 | 0.711 | 249,616,528 | 0.862 |

Three repeats per cell; `cnn-A` is the reference in every pair and so has twelve, spanning 5.5%. Every
cell is slower than the baseline, and every one is slower than its multiply-accumulate count alone
predicts — `attn-C` runs 17% below MAC-proportional and `cnn-from-to` 10% below, which is the extra
kernel launches the from-to head's four small matmuls cost at batch 64.

**This does not reproduce the +40% figure for attention at native batch-64 inference that
`chess-attention-sdpa-backends-rtx4070s-20260818` records.** Here attention is 23–29% *slower* than the
convolutional baseline on the same card. The two are not measuring the same thing — that benchmark
compared a particular attention configuration against a particular CNN, while these cells are matched at
~3.9M total parameters, which is precisely the constraint that leaves attention carrying 13–23% more
multiply-accumulates. Both numbers can be right; only this one answers "at matched parameters".

**The search-path harness cannot price an architecture on its own.** It measures the whole search, and the
number of network evaluations per search depends on the policy the weights produce. Every trained cell
sits at 1.016 positions per search, so the trained comparisons above are sound — but an *untrained* model
of the same architecture reported 0.41 positions per search, a 2.4× different workload. Architecture-only
questions therefore need a forward-pass measurement.

#### Forward-pass throughput, no search

Batch 512 bf16, models interleaved with the reference so that ordering, clock drift and kernel autotuning
cannot favour one, seven timed repeats per entry, two passes.

| model | batch 512 | vs `cnn-A` | batch 64 | vs `cnn-A` |
| --- | --- | --- | --- | --- |
| `cnn-A` (12x128 + dense) | 103,150 | 1.000 | 17,575 | 1.000 |
| `cnn-from-to-narrow` (12x128 + from-to) | 101,210 | **0.981** | 15,947 | 0.907 |
| `cnn-from-to` (12x136 + from-to) | 57,130 | 0.554 | 16,444 | 0.936 |
| `attn-C` | 37,240 | 0.361 | 8,044 | 0.458 |

**The from-to head costs 1.9% at the production trunk width**, not the 21% the search harness suggested
for `cnn-from-to`. That 21% was the trunk widening, not the head.

#### Trunk width against throughput, and why 136 was a bad choice

12-layer global-pooling trunks with the dense head, batch 512 bf16, torch 2.12.1+cu126 on one RTX 4070
SUPER. **One process per width**, each measuring only `{128, width}` interleaved twice, so cuDNN never
sees more than two distinct shapes. The 128 reference reproduced across twelve independent processes at
103,022-104,014 positions/s, a spread of 0.5%.

| channels | positions/s | ratio to 128 | width-squared predicts | efficiency |
| --- | --- | --- | --- | --- |
| 96 | 132,841 | 1.251 | 1.778 | 0.70 |
| 112 | 95,854 | 0.904 | 1.306 | 0.69 |
| 120 | 93,448 | 0.881 | 1.138 | 0.77 |
| **128** | **103,490** | **1.000** | **1.000** | **1.00** |
| **136** | **59,678** | **0.565** | 0.886 | **0.64** |
| 144 | 63,808 | 0.605 | 0.790 | 0.77 |
| 152 | 53,807 | 0.510 | 0.709 | 0.72 |
| 160 | 61,801 | 0.587 | 0.640 | 0.92 |
| 176 | 52,443 | 0.499 | 0.529 | 0.94 |
| 192 | 41,530 | 0.394 | 0.444 | 0.89 |
| 224 | 35,473 | 0.336 | 0.327 | 1.03 |
| 256 | 31,148 | 0.295 | 0.250 | 1.18 |

**128 is a sharp isolated optimum and the curve is not monotone.** 112 and 120 are slower in absolute
terms than 128 despite doing less arithmetic, and 136 is 42% slower than 128 for 13% more arithmetic.
Efficiency against width-squared then recovers to 0.89-1.18 from 160 upward, which is the ordinary
picture of a small model becoming compute-bound as it grows.

**It is not an alignment rule.** 96, 112 and 120 are inefficient and 176, 224 and 256 are efficient, so
divisibility by 32 predicts none of it. Two candidate mechanisms were tested directly and neither
accounts for it:

| | default | cuDNN autotune | channels-last |
| --- | --- | --- | --- |
| 128 | 103,490 | 105,075 | **124,829** |
| 136 | 59,678 | 64,057 | 66,572 |
| 160 | 61,801 | 65,193 | 72,905 |

Autotuning recovers 7% of 136's deficit and the NHWC layout 12%; the gap survives both. The mechanism is
unidentified and this note does not claim one. What is established is the shape of the curve and that it
is neither alignment, nor autotune, nor memory layout.

Two consequences beyond this experiment, both needing their own validation before being acted on:

1. **Progressive-sizing rungs should avoid the 132-152 band**, which costs 25-35% beyond its arithmetic.
   The production rungs of 96 -> 160 -> 192 sit outside it already.
2. **`channels_last` is worth 21% at 128 and 18% at 160** and is independent of everything else measured
   here. The `cnn-inference-throughput` work stream is already measuring `bf16-channels-last` on this same
   node, so this corroborates that line rather than duplicating it.

#### Rung shapes at the batch sizes production runs

> **Provenance: transcribed from run output, raw JSON not preserved.** These measurements were made on
> the RTX 4070 SUPER on 2026-08-27 and the node was released before they were fetched. Every other table
> in this note is backed by a file in this directory; this one is not. The figures are two interleaved
> passes per shape against the reference, the reference reproduced within 0.5% across every process, and
> the `C(width)/layers` model below reproduced all seven shortlist shapes to within 1.4% — but they are
> transcriptions and should be re-measured before anything depends on them alone.

Self-play runs `inference_batch_size: 320` and evaluation runs 64. Batch 512, which the width study
above uses, is neither, and the ranking is not the same at all three.

| shape | vs reference | batch 512 | batch 320 | batch 64 |
| --- | --- | --- | --- | --- |
| `20x128` | vs `14x152` | 1.356 | 1.203 | **0.730** |
| `10x176` | vs `14x152` | 1.352 | 1.237 | 1.339 |
| `13x160` | vs `14x152` | 1.234 | 1.128 | 1.100 |
| `14x160` | vs `14x152` | — | **1.048** | **1.013** |
| `15x160` | vs `14x152` | — | 0.980 | 0.930 |
| `16x160` | vs `14x152` | — | 0.918 | 0.960 |
| `14x176` | vs `14x152` | — | 0.888 | 0.956 |
| `11x224` | vs `18x176` | 1.099 | 1.241 | **1.660** |
| `34x128` | vs `18x176` | 1.057 | 1.161 | **0.556** |
| `19x176` | vs `18x176` | — | 0.948 | 0.970 |
| `20x176` | vs `18x176` | — | 0.899 | 0.912 |
| `22x160` | vs `18x176` | — | 0.995 | 0.838 |
| `4x224` | vs `12x128` | 0.978 | 1.087 | **2.356** |
| `6x176` | vs `12x128` | 0.977 | 1.010 | **1.741** |

**Depth is the expensive axis at production batch sizes, and the ranking inverts against batch 512.**
`20x128` is 1.36 of `14x152` at batch 512 and 0.73 at 64; `34x128` goes 1.06 to 0.56. At batch 64 an 8x8
board makes every convolution tiny, so time tracks kernel launches: throughput is close to
**213,000 / layers positions per second regardless of width**, which is why `4x224` beats `12x128` by
2.36 there. Any future rung change must be measured at 320 and 64. Measuring at 512 alone would have
put a rung into v10 that is 27% slower in self-play.

A `C(width) / layers` model fits the measurements: C is about 1,150,000 at width 128, 690,000 at 160,
575,000 at 176 and 428,000 at 224 for batch 320, and 213,000 at any width for batch 64. It reproduced
`14x160`, `15x160`, `16x160`, `14x176`, `19x176`, `20x176` and `22x160` to within 1.4%.

**Open question.** `16x192` and `19x176` are the same model to within 28 training parameters
(10,251,274 against 10,251,246) and 0.4M multiply-accumulates. At batch 512, both measured, `16x192` is
0.940 of `19x176`; at batch 64 it should be 1.19 on the layer-count rule. Batch 320 was never measured
for width 192, and extrapolating the measured C320/C512 ratios of neighbouring widths puts `16x192` at
about 0.98. `19x176` was chosen because its 320 and 64 figures are measured rather than extrapolated and
because self-play dominates the time budget, but if evaluation throughput ever becomes the constraint,
`16x192` is the shape to measure first.

#### The earlier reading of this effect was wrong twice

Sweeping many distinct widths in one process depressed every entry after the first by up to 40%, which
looks like autotune-cache thrashing: 160 read 52,000 positions/s in an eight-width sweep and 62,000 when
measured in isolation against 128. Only one process per width reproduced. Two readings were published to
the session before that was controlled for — first that the from-to head cost 21% of throughput, when it
was the trunk widening that did, and then that 128 was 2x faster than every other width, when relative to
its own arithmetic it is 160 and above that are normal and 132-152 that are not. Both are corrected above;
the raw per-width JSON under `widths/` is the record.

**Equal-compute implication.** The recommendation below costs 1.9% of forward throughput at batch 512 and
9% at batch 64, against 0.0298 nats (+33 to +42 Elo). That trade needs no equal-compute arithmetic to
justify. `attn-C`'s does: it buys 0.0406 nats at 0.361 of the forward rate, and must be settled by an
actual equal-compute match rather than by a conversion.

### Peak training memory

Measured on the RTX 3060 at batch 1024; memory is not meaningfully affected by the contention.

| cell | peak MiB | ratio to `cnn-A` |
| --- | --- | --- |
| `cnn-A` | 1,181 | 1.00 |
| `cnn-from-to` | 1,342 | 1.14 |
| `attn-A` | 4,853 | 4.11 |
| `attn-B` | 5,230 | 4.43 |
| `attn-C` | 6,104 | 5.17 |

Attention's peak is 4.1–5.2× the convolutional baseline's, well above the 1.94–2.43× this repository had
previously recorded. Batch 1024 still fits on a 12 GiB card for every cell, but two attention cells do not
fit together, which constrained how the eight cells were scheduled.

## Two defects found along the way

**The multiply-accumulate counter saw no attention cost at all.** `FlopCounterMode` returns zero for the
fused SDPA backends on some devices, so the same attention model reported 222.4M MAC per position on CPU
and 242.5M on CUDA. Forcing the decomposed backend for the measurement makes the two agree. This changes a
premise of the comparison: at matched parameters the attention trunks are **not** FLOP-neutral against the
convolutional one — the 64×64 score matrix costs `2 × 64² × E` per layer that no parameter pays for, so
`attn-A` spends 13% more multiply-accumulates than `cnn-A` and `attn-B` 23% more.

**The native inference pipeline corrupted the from-to head's index buffers.**
`torch::jit::script::Module::to(dtype)` casts every parameter and buffer with no floating-point guard,
unlike Python's `nn.Module.to`. On CUDA the pipeline runs bfloat16, where 4,094 is not representable, so
the head's int64 gather indices were destroyed and the search died with `gather(): Expected dtype
int32/int64 for index`. Every existing test missed it because CPU inference runs float32, where those
indices round-trip exactly. Fixed by converting only floating-point state; regression-tested by exporting
the from-to head on both trunks through the native pipeline, plus a CUDA-gated case on the bfloat16 path.

## Validation

`ruff format` and `ruff check` are clean on every touched Python file. The suite runs from `py/`:

- Workstation, CPU only: `python -m pytest --import-mode=importlib ./test -q` → **806 passed, 56 skipped**.
- RTX 4070 SUPER, native extension and CUDA present: **967 passed, 4 skipped, 13 failed**.

The 13 failures are pre-existing and unrelated: `test_experiment_queue_process` (5),
`test_trainer_group` (4), `test_game_contracts` (2), `test_interactive_engine` (1) and
`test_experiment_configuration` (1). The identical 13 were reproduced on a detached worktree at the base
commit `6c62e605` before any change on this branch, and `documentation/CURRENT-STATE.md` already lists
them as a known open item.

The native from-to head is exercised on both trunks through the real inference pipeline, plus a
CUDA-gated case on the bfloat16 path. That case is the regression test for the integer-buffer defect
below; the defect's empirical proof is that the same five checkpoints failed the throughput measurement
before the fix and completed after it, on the same card and the same binary otherwise.

## Reproduce

From `py/` on the node, with the locked virtual environment:

```
# One cell. tools.attention_viability_cells prints the exact command line for any of them.
python -m tools.attention_viability_cells --cell attn-C --output-root /workspace/attention-cells \
  --python /workspace/alphazero-engine-venv/bin/python --steps 8000

# The generation-0 bootstrap prior of every architecture cell, on real positions.
python -m tools.measure_bootstrap_policy_prior --dataset /workspace/distill/attention-6m.bin \
  --output bootstrap-prior.json --scratch /tmp/bootstrap-scratch

# The results table, joined to the parameter and multiply-accumulate split of the same cell definitions.
python -m tools.attention_viability_report --log-root /workspace/attention-cells --output cells.json

# Paired held-out cross-entropy with a bootstrap interval, against either reference.
python -m tools.attention_viability_paired_ce --dataset /workspace/distill/attention-6m.bin \
  --run-state-root /workspace/attention-cells --output paired-cross-entropy.json \
  --generation 322 --positions 32768 --bootstrap-samples 10000 [--reference cnn-from-to]

# Native inference and search throughput at the batch size the comparison is stated at.
python -m tools.distill_match --mode throughput-only --throughput-position-count 64 \
  --parallel-searches 1 --searches-per-move 64 --opening-pair-count 100 \
  --teacher-run-state <cnn-A> --teacher-generation 322 \
  --student-run-state <cell> --student-generation 322 \
  --openings-manifest <openings-200.json> \
  --experiment-config configs/validation/distillation-probe-chess-single-gpu.yaml --output <json>
```

## Files

| file | contents |
| --- | --- |
| `bootstrap-prior.json` | generation-0 exported policy prior of each architecture cell on 256 real positions |
| `cells.json` | every cell's full evaluation curve joined to its parameter and multiply-accumulate split |
| `paired-cross-entropy.json` | paired held-out differences against `cnn-A`, 32,768 positions |
| `paired-vs-control.json` | the same against `cnn-from-to`, the head control |
| `throughput-4070super.json` | native search throughput on the production card |
| `paired-final.json` | the paired differences including the head control, 32,768 positions |
| `forward-interleaved.json` | forward-pass throughput, models interleaved with the reference |
| `channel-136-vs-160.json` | the 128 / 136 / 160 channel comparison, interleaved |
| `channel-alignment-interleaved.json` | the 128 / 160 alternation that ruled out ordering effects |
| `widths/w<N>.json` | one process per trunk width, each interleaved against 128 |
| `widths/cudnn-benchmark.json`, `widths/channels-last.json` | the two candidate mechanisms, neither of which explains the 136 deficit |

Training logs for all ten cells were fetched off the ephemeral node to
`.codex-diagnostics/chess-attention-viability-20260827/logs/`.

**One set of measurements was not preserved.** The batch-320 and batch-64 rung comparisons in "Rung
shapes at the batch sizes production runs" were made on the RTX 4070 SUPER and the node was released
before their JSON was fetched; that table is transcribed from run output and is labelled as such. Every
other table here is backed by a file in this directory. The lesson is the one the repository already
states: fetch each result as it completes, not at the end of the session.
