# Chess attention viability — rtx3060, 2026-08-27

Can an attention trunk be made competitive with the convolutional one in this engine, now that the
generation-0 bootstrap defect that invalidated the two earlier attention runs is fixed?

**Yes, but almost none of the win is the trunk.** The best cell is an attention trunk with a generated
attention bias and a from-to policy head, at 0.0406 nats better held-out policy cross-entropy than the
convolutional baseline — comfortably past the 0.03-nat bar. But a *convolutional* trunk carrying the same
from-to head recovers 0.0316 of that 0.0406 on its own, and a bare attention trunk is 0.0060 nats
**worse** than convolution. The policy head is the finding; the trunk is close to a tie.

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

### Native inference and search throughput — RTX 4070 SUPER

**These are the production card, so unlike everything above they transfer.** Measured through the
evaluation search path with `parallel_searches: 1` and the memory-efficient SDPA backend, at 64 roots
(so the inference batch is 64) with 64 searches per move. `cnn-A` is the reference in every pair and is
therefore measured once per pair.

RESULTS PENDING.

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
| `throughput-4070super.json` | native inference and search throughput on the production card |

Training logs for all nine cells were fetched off the ephemeral node to
`.codex-diagnostics/chess-attention-viability-20260827/logs/`.
