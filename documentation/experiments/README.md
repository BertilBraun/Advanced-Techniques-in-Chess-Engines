# Experiments — what worked, what did not

The interesting part of this project is not the pipeline, it is the ledger of things that were tried against a
fixed reference and either survived or did not. This page is that ledger in narrative form. Numbers here are
summaries; the measured artifacts live in [`benchmarks/`](../benchmarks/README.md), the run-level records in
[`evidence/`](../evidence/README.md), and the current authority is the
[chess recovery plan](../plan/chess-recovery-plan-20260820.md).

## How a candidate is judged

Two rules decide everything below.

1. **Slope, not level.** A configuration is judged on the slope of fixed-dataset policy accuracy and fitted
   Stockfish-ladder Elo across at least four evaluation boundaries (1,200–4,800 s). A steep slope from a low
   start beats a shallow slope from a higher start; single points are noise at 50–100 game samples.
2. **Against the yardstick, not against each other.** The four-day r3/r4 run (≈2,800 ladder Elo at 10k visits;
   generation 445 scored 66.0 % vs Stockfish 13 at 6,500 nodes) is the absolute reference. Its per-generation
   and per-wall-clock-hour table is
   [`yardstick_wall_h.csv`](../evidence/chess-four-day-freeze-20260817/yardstick_wall_h.csv), and every recovery
   run is read against it.

Rule 2 was learned the hard way: the 2026-08-23 component ladder validated four configurations against each
other, all four passed, and the winner still never matched the yardstick's pace. Comparative screening is cheap
and answers "which of these", never "is this good enough".

## Architecture: attention lost to the CNN, but not for the reason it looked like

An attention trunk plateaued for seven hours where the CNN baseline climbed. The root cause was not the
architecture: the generation-0 TorchScript export runs BatchNorm in eval mode, which collapsed the exported
policy logits to a standard deviation of 0.065 — a near-uniform prior. Self-play search then spread its visits
across all legal moves, produced flat training targets (top-1 target mass 0.12 against the CNN's 0.39), hit the
ply cap in 61 % of games, and learned almost nothing.

Scaling the policy prior on the gen-0 export (logit std 5.45) fixed the bootstrap outright: target top-1 mass
0.52, target entropy 1.62 down from 3.2, no ply-capped games, and the run reached in ~100 minutes what the
plateaued attention run needed 7 h for.

With the bootstrap unblocked, the architecture question got a clean answer and attention still lost: over the
matched window its accuracy slope was ~75 % of the CNN's (Δ+0.109 vs Δ+0.146), ladder Elo ~100 lower, and no
level-0 win uptick. The standing rule was "attention if within ~5 % of the CNN"; production runs the CNN.
Attention remains viable and untuned — prior scale, learning rate and warmup are its open candidates.

Related measurements: [contended architecture comparison](../benchmarks/chess-architecture-contended-rtx3060-20260817/README.md),
[packed QKV](../benchmarks/chess-attention-packed-qkv-rtx3060-20260818/README.md),
[SDPA backends](../benchmarks/chess-attention-sdpa-backends-rtx4070s-20260818/README.md),
[attention training throughput](../benchmarks/chess-attention-training-rtx4070s-20260818/README.md).

> Caveat on the fix: the export-time prior scale is a constant chosen per architecture, and the resulting logit
> standard deviation varies by seed and by head shape. It has to be re-measured whenever the policy head changes
> — the rank-96 head below is exactly such a case.

## Replay ratio: nearly free, and mostly a scheduling knob

Replay ratio (samples ingested per sample trained on) was A/B-tested at 2 and 16, 65 minutes each, same recipe.
Wall-clock strength came out near-identical — level-0 W/D/L 3/20/77 vs 3/19/78, fixed-dataset accuracy 0.198 vs
0.227, ladder 755 rising vs 683 flat. Sample reuse simply is not the binding constraint at this scale.

What the ratio actually changes is *pace*: ratio 16 raced through 48 generations per hour, so every
generation-indexed schedule (visits, learning rate, temperature, ply caps, capacity) advanced four times faster
in wall-clock terms than at ratio 4. Production therefore runs ratio 4 with every generation schedule scaled to
⅔ and floored at 5 seconds — the ratio was chosen for schedule pacing, not for data efficiency.

## Policy head: 208k parameters tie 484k

Seven dense policy-head variants were trained for 4,000 steps each on the same frozen replay store (12×128
trunk, 1,880 actions, batch 1,024), scored by holdout policy cross-entropy:

| Variant | Head parameters | Holdout policy CE |
| --- | ---: | ---: |
| channels 4, bottleneck rank 96 | 207,552 | **2.0812** |
| channels 4, full rank (baseline) | 483,680 | 2.0825 |
| channels 8, 1 spatial reduction, rank 96 | 358,856 | 2.0885 |
| channels 8, 1 spatial reduction, rank 64 | 289,448 | 2.1022 |
| channels 4, 1 spatial reduction | 420,832 | 2.1125 |
| channels 4, bottleneck rank 64 | 139,168 | 2.1179 |
| channels 8, 2 spatial reductions | 538,984 | 2.1757 |

Reading: a rank-96 bottleneck removes 57 % of the head's parameters at no measurable cost, rank 64 costs
+0.035 CE, and reducing spatial resolution before the projection hurts in every pairing — the head needs the
board, not the channels. Production uses dense channels 4 with bottleneck rank 96. Evidence bundle:
`.codex-diagnostics/policy-head-bakeoff-20260824/` (run-local, not tracked).

The small-initialisation of that bottleneck is not free: two factors multiply, so Kaiming-scaled first factors
plus the BatchNorm blowup put the exported gen-0 prior at logit std 256, five times beyond the validated band.
Small-initialising both factors and halving the first factor's standard deviation brings it back to 13–32
across seeds.

## Encoding: back to 1,880 actions

The reduced chess action encoding (`reducedActionCount = 1880` in
[`ChessEncoding.hpp`](../../cpp/src/games/chess/encoding/ChessEncoding.hpp), `CHESS_ACTION_SIZE` in
[`contract.py`](../../py/src/games/chess/contract.py)) was restored in place of the 4,864-action scheme the
rework had introduced — a 2.6× smaller policy target, which is most of why a 208k-parameter head suffices.
The colour-symmetry flip harness re-validated the change over 83,651 moves with zero failures.

## The rediscovered discount

Forensics on the four-day run found that r3 carried a hard-coded 0.99-per-ply discount on the MCTS backup value
that was deleted at the r3→r4 boundary and never re-introduced. It is now an explicit optional configuration key
(`objective.search_value_discount_per_ply`,
[`training/configuration.py`](../../py/src/training/configuration.py)) rather than a constant in the search, and
production sets 0.99 per ply with target 1.0.

Two other differences the same forensics ruled *out*: r3's ingestion was exactly-once at effective ratio 8.0
(today measures 7.94), and r3's sample-time mirror carried the castling-plane bug that has since been fixed —
it hurt r3, it did not help it.

## Throughput: the GPU is rarely the problem

- **Inference batch size.** Sweeping the per-worker inference batch at realistic budgets (4 processes per GPU,
  trained 12×128) gave 84.7k searches/s at batch 64, 100.9k at 128 (+19 %) and 106.9k at 256 (+26 %).
  Production runs batch 256 at 512 parallel games per process. Raising games per process to 768 added 4 %
  throughput but slowed each individual game by ~30 % through staleness, and was rejected.
- **Model size dominates early self-play.** At generation-0 conditions a 1.12M-parameter rung produces 1.86×
  the searches per second of a 3.07M rung in the production 4-process topology, because the small model leaves
  GPU headroom that process-level fill exploits while the big one saturates compute. That ratio is the entire
  rationale for progressive sizing — see the
  [progressive-sizing throughput benchmark](../benchmarks/progressive-sizing-throughput-rtx4070super-20260823/README.md)
  and the [accepted policy](../architecture/progressive-model-sizing.md).
- **Generation cadence has not moved.** ~15 generations/hour in both the 2026-08 era and the four-day era,
  on faster GPUs: self-play is bound by batching and host CPU, not by GPU FLOPs. The wall-clock deficit against
  the yardstick is therefore a learning-efficiency problem, not a throughput problem — which is why the ladder
  above is about targets, priors and schedules rather than about kernels.
- **Search itself.** `parallel_searches` 1→4 buys +35–38 % searches/s in a single process by filling inference
  batches ([search throughput](../benchmarks/search-throughput-rtx4070-20260821/README.md)); a deliberately
  unoptimised Python PUCT reference manages ~81 sims/s
  ([naive Python MCTS](../benchmarks/naive-python-mcts-rtx3060-20260816/README.md)).

## The open question

A zero-deviation replica of the r3 recipe was the decisive diagnostic: it became the best recovery run so far
(level-0 0.685 at 5 h, ladder 663→1,101 monotone) and closed the wall-clock gap from 5+ hours to ~1.5–2 hours,
but it still needed ~1.6–1.7× r3's generations to reach the same strength. Configuration alone therefore does
not explain the gap; the residue is platform-level — search semantics, replay mixing and target composition are
the candidates. `vast-chess-4day-production-v2` is the run testing the assembled answer.

Go (7×7 and 9×9) screening is implemented and paused for the duration of the chess recovery.
