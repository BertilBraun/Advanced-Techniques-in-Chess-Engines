# Attention run throughput, precision and refit — 8x RTX 4070 SUPER, 2026-09-29

Pre-start measurements for the [attention run](../../plan/chess-attention-final-run-plan-20260929.md): which
T1-shaped size to use as the medium model, whether FP8 or weight-only INT8 is worth building into production,
and whether the Myelin-compiled attention engines refit correctly.

## Setup

| | |
|---|---|
| Node | Vast.ai instance `53401154`, 8x RTX 4070 SUPER 12 GiB at a **160 W** power limit, driver 580.159.03, Xeon E5-2673 v4, 76.8 granted CPUs, 120 GiB granted RAM ([node note](../../operations/current-node.md)) |
| Runtime | PyTorch 2.12.1+cu126, cuDNN 9.10.2, TensorRT 10.14.1.48 (Python and native), ModelOpt 0.46.1 |
| Source | `ce04f219` |
| Configuration | `vast-chess-8gpu-final-attention.yaml`, SHA-256 `be317a46fbcc3ab42dadfaf164adbc0cbe4ac6c0baf60ea7d0fc45928870dbf7` (search and inference settings are unchanged in later revisions) |
| Models | Random-initialised inference exports of each architecture ([export](raw/scripts/export_matrix_models.py)); throughput depends only on the architecture. The trained 10x192 is `lc0arch-r2/step_80000` from the Lc0 diagnostic (`evidence-lc0arch-and-throughput.tgz`, SHA-256 `20495142...`) |
| Evaluation positions | `chess-stockfish-evaluation-v33.bin` (516 positions): 196 for fidelity, the other 320 for calibration, no overlap |

Parameters: attention 8x160 7.22M, 10x192 11.91M, 12x192 14.05M, 10x224 15.67M (heads = width / 32, FFN 4x
width, smolgen 32/width/width); CNN 14x160 6.52M.

## Forward throughput at batch 320

[`tools/compare_tensorrt_low_precision.py`](../../../py/tools/compare_tensorrt_low_precision.py), one GPU per
model, CUDA-graph replay, median of 15 x 100 batches. FP8 and INT8 quantize only activation-by-weight products
(max calibration); the policy and value heads stay float16.

| Model | Float16 positions/s | FP8 | Weight-product INT8 |
|---|---:|---:|---:|
| Attention 8x160 | 92,119 | 95,347 (1.04x) | 103,796 (1.13x) |
| Attention 10x192 | 56,510 | 62,871 (1.11x) | 71,018 (1.26x) |
| Attention 12x192 | 49,539 | 54,504 (1.10x) | 62,743 (1.27x) |
| Attention 10x224 | 46,153 | 48,186 (1.04x) | 54,168 (1.17x) |
| CNN 14x160 | 77,687 | 75,036 (0.97x) | 134,814 (1.74x) |

The CNN's 1.74x from INT8 matches the 1.70x measured on the previous 4070 SUPER node; the attention networks
gain 1.04-1.27x.

## FP8 and INT8 fidelity on the trained 10x192

Against its own float16 engine on 196 evaluation positions:

| Precision | Speed | Top-move agreement | Mean policy KL | WDL MAE | Expected-value MAE |
|---|---:|---:|---:|---:|---:|
| FP8 | 1.11x | 0.842 | 0.0350 | 0.0174 | 0.0349 |
| Weight-product INT8 | 1.23x | 0.832 | 0.0693 | 0.0271 | 0.0559 |

**FP8 is rejected** by the plan's criterion (1.2x or better with no measurable loss): it is barely faster and
changes the top move on 16% of positions. Post-training INT8 is faster but less faithful still. The fidelity
figures of the random-weight models are not meaningful (their policies are near uniform) and are omitted.

## Self-play at the live settings

[`tools/run_self_play_search_benchmark.sh`](../../../py/tools/run_self_play_search_benchmark.sh), one GPU per
model, 4 workers x 512 games, float16 TensorRT refit templates at batch 320, generation 1026's search budget,
60 s. Five models ran concurrently on GPUs 0-4, so each had about 15 CPUs, more than the live run's 2.4 per
worker; every model was GPU-bound regardless (1.1-2.4 CPUs used per GPU, batches 319.8 / 320).

| Model | Searches/s per GPU | Relative to 8x160 | Summed peak RSS (4 workers) |
|---|---:|---:|---:|
| Attention 8x160 | 80,539 | 1.00 | 10.6 GiB |
| Attention 10x192 | 50,999 | 0.63 | 10.5 GiB |
| Attention 12x192 | 44,112 | 0.55 | 10.5 GiB |
| Attention 10x224 | 41,211 | 0.51 | 10.5 GiB |
| CNN 14x160, float16 | 64,847 | 0.81 | 10.6 GiB |

The production CNN served INT8, which the forward matrix puts at 1.74x its float16 rate.

**Medium model: 10x192 stays.** 12x192 and 10x224 search 13% and 19% slower, and 10x192 is the only size whose
strength was measured (the distillation arms).

## Refit

The attention engines had never been refit. A template built from random weights
([`build_tensorrt_refit_template.py`](../../../py/tools/build_tensorrt_refit_template.py), refit mode `all`,
optimization level 3) was refit through the production `refit_engine` with the same architecture's weights
perturbed by x1.01 + 0.001, then compared with an engine built directly from those weights on 320 evaluation
positions: **top-move agreement 1.000, policy KL 0.0, WDL MAE 4.7e-5** ([log](raw/logs/refit-fidelity.log)).

Refitting it with the trained 10x192 failed (96 of 126 weights), because that model was exported by an older
source revision whose graph has 232 initializers against the current export's 94. Production publications are
exported by the running source, so every publication matches its template.

## Raw data

[`raw/results`](raw/results) (forward matrix reports), [`raw/self-play`](raw/self-play) (benchmark manifests and
summaries), [`raw/logs`](raw/logs), [`raw/scripts`](raw/scripts) (the scripts as run on the node).
