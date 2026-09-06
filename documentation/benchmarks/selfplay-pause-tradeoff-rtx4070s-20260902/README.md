# How many self-play workers should keep running during training

2026-09-02. Node `38.49.42.120:53893`, 8x RTX 4070 SUPER, 200 W host power cap, two NUMA islands of
four GPUs, PCIe gen3 x16, no P2P mesh.

`node_ids_to_pause_during_training` decides how many of the 32 self-play workers keep running while
the trainer holds the GPUs. Production ran 24 paused / 8 running. This measures the trade-off directly
against an existing replay store and checkpoint, without generating any training data.

## Method

`py/tools/benchmark_training_throughput.py --self-play-workers N`. The tool stages a real checkpoint,
opens an existing replay store read-only, spawns the full self-play group, applies a fail-closed
(flat-to-baseline) stop policy, then pauses all but N workers — mirroring the production
pause-during-training knob. Two quanta run; the first is discarded because it absorbs compilation and
the self-play model load. GPU utilisation and power are sampled once a second across the measured
quantum via `nvidia-smi`.

| | |
| --- | --- |
| source configuration | `py/configs/validation/vast-chess-4day-production-v27.yaml` |
| `experiment_configuration_sha256` | `1f110e1733c20eb2595d450daf4796fa147aba0ec489fb8e06d12105e0ae97e8` |
| run state | `vast-chess-4day-production-v27`, checkpoint generation 152 |
| replay rows | 4,000,000 |
| `baseline_visits` at generation 152 | 600 (the late regime, reached for free from the schedule) |
| trainer | 8-way DDP, global batch 2048, 500 optimizer steps, bfloat16, `torch.compile` default |
| source revision (corrected arms) | `6fa851df` |

## Result

| workers running | trainer quantum | searches | searches/s | trainer samples/s | mean GPU util | mean GPU power |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 40.5 s | — | — | 25,275 | 42.2% | 55.9 W |
| 8 | 47.6 s | 24,087,000 | 505,509 | 21,492 | 93.8% | 130.9 W |
| 16 | 59.8 s | 36,195,000 | 605,762 | 17,139 | 98.9% | 148.4 W |
| 32 | 111.2 s | 82,443,000 | 741,508 | 9,210 | 99.8% | 170.3 W |

**Cross-check against production.** The 8-worker arm reports 21,492 samples/s; live v27 at generations
130–143 reported 22,014 samples/s at the same setting. That agreement is what makes the rest of the
table trustworthy, and its earlier absence is what exposed the bug below.

## Converting to generation time

During training N workers run; outside it all 32 run. Calibrating searches-needed-per-generation
against v27's observed ~117 s/generation at N=8:

| workers running | training window | self-play window | generation |
| --- | --- | --- | --- |
| 8 | 47.6 s | 69.5 s | 117.1 s |
| 16 | 59.8 s | 53.1 s | 112.9 s |
| 32 | 111.2 s | 0 s | 111.2 s |

A ~5% spread, mildly favouring more concurrent workers. Two assumptions carry real weight at that
margin: that all 32 workers run flat out in the non-training window (backpressure sometimes prevents
it), and the searches-per-generation constant calibrated from v27. Treat the ranking as the result,
not the seconds.

This is regime-dependent. An earlier sweep over live generations 3–14 at `baseline_visits` 300 found
pausing nearly irrelevant — 8 running 95.5 s/gen (v23), 16 running 96.7 s/gen (v26), 0 running
97.7 s/gen (v24), all at `excess_cost_ceiling` 0.15, a 2.3% spread. Self-play is cheap early and the
trainer dominates; at 600 visits self-play dominates and the balance tips toward unpausing.

## The bug this measurement first produced

`SelfPlayGroup` assigns worker *i* to `device_ids[i]`, and the production `device_ids` list places four
consecutive workers on each GPU. The benchmark originally kept the **first N** worker IDs running, so
the 8-worker arm placed all eight on GPUs 0 and 1 and left GPUs 2–7 with no self-play at all. DDP runs
at the speed of its slowest rank, so the trainer measured that imbalance rather than the self-play
load. Superseded numbers, recorded so they are not mistaken for a second data point:

| workers running | trainer quantum | searches/s | trainer samples/s |
| --- | --- | --- | --- |
| 8 (consecutive IDs) | 85.6 s | 292,967 | 11,977 |
| 16 (consecutive IDs) | 100.9 s | 411,546 | 10,151 |

Those figures suggested unpausing was worth 2.53x more search per second of training. It is worth
about 5% of generation time. The arms at 0 and 32 workers are symmetric and were unaffected.

The tool now selects running workers round-robin across devices, reproducing the production pause list
exactly at N=8 (`[0, 4, 8, 12, 16, 20, 24, 28]`).

## Incidental finding

Loading an optimizer from a checkpoint places AdamW's step counters on the GPU via `map_location`,
where foreach AdamW synchronises once per parameter per optimizer step; a freshly created optimizer
keeps them on the CPU. Fixed in `c13b749c` with a CUDA-gated test. The 0-worker arm reached
25,275 samples/s *without* the fix, above production's 22,014, so the cost is not material at this
model size — the fix is correct but explains nothing here.

## Applied to

`vast-chess-4day-production-v28` runs 16 paused / 16 running, two per GPU.
