# 7. Final-run results

> **Status: complete.** The run stopped cleanly on 2026-09-23 and its evidence is fetched and verified. Quantitative
> authority is [`documentation/results/final-chess-run.md`](../results/final-chess-run.md); this chapter interprets
> it and does not restate every artifact.

The project-level result record is [`documentation/results/final-chess-run.md`](../results/final-chess-run.md). Once
the archive and evaluations are complete, that page is the quantitative authority; this chapter should interpret it
rather than duplicate every artifact.

## Frozen run identity

| Field | Final value |
| --- | --- |
| Public run name | `vast-chess-8gpu-v89-v35-progressive-int8-sgd-reuse4-plateau` |
| Segment producing the selected checkpoint | `vast-chess-8gpu-v97-revert-promotion` |
| Training lineage and resume boundaries | V89 → V91 → V92 → V93 → V94 → V95 → V97 → V99 in one run directory; V90 and V96 reverted and excluded |
| Source revision | `f9f8cee4c59403befdf55d34d3d7dffc3671a97a` |
| Resolved configuration SHA-256 | `498e7687f68eb48da87212a2a6e43b5fc84b903983c23c9ab1af7cd46a60d243` |
| Archive SHA-256 and location | eight verified archives under `.codex-diagnostics/final-2026-09-23/`, listed in the [result record](../results/final-chess-run.md#run-identity) |
| Start, stop, and effective training duration | stopped 2026-09-23; 2.5 days on the stitched evaluation axis, 3.32 days of raw boundary seconds before excising V90 and V96 |
| Hardware/runtime identity | 8x RTX 4070 SUPER, Vast.ai offer 48571853, driver 595.71.05; PyTorch 2.12.1+cu126, CUDA 12.6, cuDNN 9.10.2 |
| Training-node cost and exclusions | **$43.20** (60 h x $0.72/h); excludes evaluation compute, the distillation runs, and the V100/V101 follow-ups |

The lineage audit must distinguish continuous learning state from operational run identifiers V89–V93. Downtime,
failed exports, discarded segments, evaluation compute, and node rental should be reported consistently rather than
compressed into one ambiguous “training time” number.

## Selected model

| Field | Final value |
| --- | --- |
| Checkpoint generation and optimizer step | **1026**, 513,000 run-level steps (408,500 on the selected stage) |
| Active progressive stage | second of three, 14x160 scaled post-activation |
| Training/inference parameter counts | **6,315,378** trainable / **6,261,007** inference |
| Checkpoint hash | `c92a363b041a18d0ef93b852ac1c6d58716ae9a22b4e62d543de297c4ec5f904` |
| ONNX artifact hash | `d634abacae3c874eac6ded89f6af861eb81b509da638b5ad710587b1a08be658` |
| Precision and serving backend | bfloat16 training; TensorRT INT8 QAT pre-fold for self-play and evaluation |

Selection should be declared before terminal match interpretation or governed by a pre-stated rule. If a non-terminal
checkpoint is selected, document why and preserve the terminal checkpoint too.

## Training volume

| Measure | Final value |
| --- | --- |
| Optimizer steps | **513,000** |
| Training presentations | **1,050,624,000** |
| Fresh materialized positions | **≥ 261,016,277** at generation 990, the nearest recorded ledger point |
| Replay capacity and configured reuse | 20,000,000 rows, reuse 4 |
| Time and volume per model stage | 12x128 to generation 481; 14x160 from 481; 19x176 from 1081, after the selected checkpoint |
| Time and volume per visit stage | 300/400/500/600 visits from generations 0/10/50/90; 800 from 1000 |
| Replay unique-row content | **98.21%** of 20,000,000 live rows are distinct positions; 19,477,090 seen exactly once, one position 15,387 times |
| Completed self-play games, replay occupancy, age percentiles, rejection counts | not retained — the replay store and run directory were deliberately left on the node |
| Resignation | calibrated from generation 70 with 20% continuation games; per-generation counts not retained |

Counts must come from the fetched archive and reconcile coordinator, replay, and trainer accounting. If restart or
resume boundaries create duplicate counters, report the reconciliation method.

## Terminal evaluation matrix

The final protocol should include policy-only, the production-scale 64-search condition, an intermediate/deep budget
such as 10,000 searches, and a high-search condition chosen to represent roughly five seconds per move under a
documented saturated workload. The exact high budget must be fixed by measurement, not assumed from v34.

The matrix ran at policy-only, 100, 1,000, 10,000 and 100,000 searches, bracketed by two Stockfish rungs each. The
full table with both rungs, parallelism and confidence intervals is in the
[result record](../results/final-chess-run.md#terminal-strength); the reported figure per budget is the rung scoring
nearest 0.500.

| Candidate search | Opponent (anchor) | Games | W/D/L | Score | Benchmark Elo (95% CI) |
| ---: | --- | ---: | --- | ---: | ---: |
| Policy only | 1,000 nodes (1700) | 100 | 32/24/44 | 0.440 | **1658** (1597–1717) |
| 100 | 10,000 nodes (2470) | 100 | 39/18/43 | 0.480 | **2456** (2393–2518) |
| 1,000 | 50,000 nodes (2960) | 100 | 21/48/31 | 0.450 | **2925** (2875–2974) |
| 10,000 | 100,000 nodes (3100) | 100 | 30/44/26 | 0.520 | **3114** (3063–3166) |
| 100,000 | 200,000 nodes (3230) | 100 | 25/56/19 | 0.530 | **3251** (3206–3297) |

The 64-search condition is not in this table by design: it is the *training* ladder budget, reported continuously
through `evaluation/ladder_elo_64` rather than as a terminal match, and its plateau value is 2358 on the three-rung
bracketed estimator. A time-based high-search headline was not produced; the deepest condition is specified in
searches, not seconds, because search counts and elapsed time are not interchangeable and the serving topology was
not held fixed across budgets.

Every row requires balanced paired openings, exact Stockfish identity and fixed-node limit, search parallelism,
inference batch/concurrency, artifact hashes, raw games, and confidence intervals. The comparison with v34 must use
matched protocols or explicitly identify differences.

## Figures

Nine figures are rendered in [`figures/`](figures/) by
[`py/tools/render_report_figures.py`](../../py/tools/render_report_figures.py) from the archived TensorBoard bundle
`.codex-diagnostics/final-2026-09-23/evidence-tensorboard.tgz`. No curve is hand-entered. Every figure records the
runs and tags it was built from in [`figures/figures-manifest.json`](figures/figures-manifest.json), so a reader can
tie a line back to a logged series; series the figure asked for but the runs never logged are listed there as
`missing_tags` rather than approximated.

| Figure | Content |
| --- | --- |
| `01-cross-lineage-ladder-elo` | 64-search ladder Elo across v9, v29, v34, v46 and the stitched final lineage |
| `02-training-losses` | total, policy and WDL losses beside the two auxiliary losses, against optimizer steps |
| `03-optimization` | learning-rate schedule and gradient norm against the configured 1.0 clip |
| `04-final-lineage-ladder` | final-lineage ladder Elo at 64 searches and policy-only, both estimators, resume boundaries marked |
| `05-training-volume` | completed games, materialized positions, training presentations and optimizer steps |
| `06-throughput` | trainer throughput and the self-play visit budget that governs it |
| `07-replay` | replay occupancy against capacity, and mean generation age of sampled rows |
| `08-promotion` | progressive stage against the plateau signal that triggers candidate starts |
| `09-resignation` | resignation threshold, false non-loss rate against its bound, trigger volume and mean saved plies |

The lineage figures stitch V89 → V91 → V92 → V93 → V94 → V95 → V97 → V99; V90 and V96 are excluded because both
were reverted, so their boundaries describe discarded work.

Three of the originally requested figures are not rendered, because the series behind them do not exist in the
archive and estimating them would defeat the point:

- **INT8 legal-policy fidelity over time.** No fidelity scalar was ever logged. `08-promotion` occupies its slot.
- **Backend usage over time** (TensorRT INT8, FP16 fallback, bootstrap TorchScript). Backend selection was logged as
  run-log text, not as a scalar series; recovering it would mean parsing logs rather than reading archived evidence.
- **Cost/strength against v34 under matched definitions.** v34's node price was never recorded, so a dollar axis
  cannot be drawn without inventing one. Both runs are 8-GPU and bill by wall-clock, so `01-cross-lineage-ladder-elo`
  already carries the matched-estimator time comparison; the cost statement stays in the identity table above.

One further gap is internal to a rendered figure: `03-optimization` shows the gradient norm against the 1.0 cap
because the clipped-step fraction the chapter asked for was never logged. The manifest records it as missing.

## Result interpretation

**Against v34 under a matched estimator: +74 ± ~15 Elo**, not the ~+100 the raw series suggests. v34 logs only a
single-rung fit; converting both onto the three-rung estimator gives 2283.9 against 2358.0. The decomposition matters
more than the total: at 1 search the gap is **+30**, at 64 searches **+74**, so most of the gain appears only once the
tree runs rather than in the raw network.

**Progress did continue past v34 strength, and roughly twice as fast.** The final lineage reached each level in about
half the time: 2000 at 0.25 d against 0.92 d, 2200 at 1.01 d against 1.98 d, 2350 at 2.17 d against 4.02 d. Fitted
against log time the final lineage's slope is *lower* (341.6 against 469.9 Elo per decade), which is the correct
reading: the curve is shifted left by roughly 2x with a modest ceiling lift, not made steeper.

**Deeper search keeps adding strength, with clean deceleration**: +798, +469, +189, +137 Elo per decade from
policy-only to 100,000 searches. The two independent opponents at the top agree within 4 Elo.

**The 19x176 stage never became the reported model.** It was promoted twice and reverted twice — once from scratch
on a training-loss comparison (−270 Elo) and once grown function-preserving, which reached parity with its parent and
then sat 36–42 Elo below it while flat. A later dedicated run (V101) trained a 19x176 from scratch for 24 hours and
[did not beat the 12x128 baseline](../benchmarks/chess-v101-capacity-rtx4070s-20260925/README.md). Capacity was not
the binding constraint.

**Operational losses were material and are not excluded from the cost.** The lineage lost one branch to an INT8
template-staleness collapse (−450 Elo, reverted to checkpoint 480), one to the loss-gated promotion (reverted to
checkpoint 990, with 4,040,112 replay rows pruned), and 2.5 hours to a full disk. The stitched axis removes the
discarded branches so they do not consume the reported training time; the $43.20 is computed on that stitched axis
and therefore understates rental.

**Replay diversity was not a limitation**: 98.21% of the 20M live rows are distinct positions. Nor was label quality
or the learning-rate floor — both were tested directly in V100 and returned negative.

The figures are rendered and recorded, so no publication item remains outstanding in this chapter. The three
unrendered items above are unavailable in the archive rather than pending work, and are stated as such instead of
being estimated.
rather than a speculative live number.
