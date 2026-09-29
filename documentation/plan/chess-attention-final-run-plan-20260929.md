# Final run with a T1-shaped network — plan

As of **2026-09-29**. Configuration:
[`py/configs/production/vast-chess-8gpu-final-attention.yaml`](../../py/configs/production/vast-chess-8gpu-final-attention.yaml)
(`experiment_configuration_sha256` `b4fa2bc453b80e56664a2450995d4e0c2f36b0337a4766b9682944ce0794c2b7`). Not started;
waiting for a node.

## Why this run

The convolutional lineage (V89 to V101) stopped near 2,360 ladder Elo at 64 searches, and every cause tested
against it came back negative: the learning-rate floor, self-play label quality, replay diversity and model size.
The Lc0 teacher diagnostic then found the lever
([summary](../analysis/plateau-investigation-v100-onwards-20260928.md),
[measurements](../benchmarks/lc0-teacher-diagnostic-rtx3070-20260926/README.md)):

- **The network is the limit, not the search.** Lc0's T1 (20.2M parameters) run inside this project's unchanged
  search plays at least 360 Elo above checkpoint 1026 at 64 searches, and about 275 above it at 1,000.
- **The architecture is what differs.** At equal compute per position and on identical data and schedule, a
  10x192 student built like T1 beat the 14x160 convolutional student by about 200 Elo and checkpoint 1026 by
  about 150 (0.68-0.695 against 0.485 at 64 searches), trained from scratch without self-play. No convolutional
  change moved: size, residual block, pre- or post-activation, value head width, depth against width.
- **It costs serving speed.** TensorRT float16 serves it at 0.61x the CNN's positions per second; with both in
  INT8 at 0.40x (the CNN gains 1.70x from INT8, the attention network 1.12x). At equal time that is roughly
  75-185 Elo of fewer searches against about 150 gained at equal searches.
- **Our self-play never annealed.** Its learning rate held a 0.01 floor for 130+ generations while the ladder was
  flat; every distillation run gained most of its strength while its rate fell.

Distillation shows the architecture can represent a much stronger function at our compute. Only self-play shows
whether this project's loop reaches it without a teacher, at the attention network's lower throughput.

## Question and pass criterion

Does self-play with the T1-shaped ladder beat the convolutional lineage at equal wall-clock time on the same
hardware?

- **Baseline:** the convolutional lineage's `evaluation/ladder_elo_64` curve on 8x RTX 4070 SUPER, stitched
  plateau about 2,358 ([final run record](../results/final-chess-run.md)). Compared by elapsed hours, not
  generations: a generation is now 360 optimizer steps, not 500.
- **Pass:** the attention run's ladder Elo exceeds the convolutional curve at the same elapsed hour after both have
  left their steep phase, and rises above the ~2,360 plateau.
- **Proposed stop** (the decision stays with the run's owner): more than 100 Elo below the baseline at the same hour
  after 24 hours, with a flat slope.

## Configuration

The final recipe (`chess-final-config.yaml`) with only these changes:

| Setting | Final recipe | This run | Why |
|---|---|---|---|
| Model ladder | 12x128 -> 14x160 -> 19x176 CNN | 8x160 (208M MAC per position) -> medium T1-shaped | T1's construction; the same step factor as the CNN's first promotion |
| Medium model | — | 10x192 (366M), 12x192 (437M) or 10x224 (494M), chosen on the node by throughput | |
| Quantization | INT8 QAT | off, TensorRT float16 templates | QAT exists only for post-activation CNNs |
| Optimizer | SGD, Nesterov 0.9, weight decay 1e-4 | same | as AlphaZero, KataGo and Lc0 |
| Learning rate | linear 0.2 -> 0.01 floor | 0.1 to generation 100, then geometric to 0.002 by generation 1000 | AlphaZero's 0.2 at batch 4,096 scaled to 2,048; annealed 50x, near KataGo's automatic schedules at our batch |
| Generation | 500 steps, replay reuse 4 | 360 steps, reuse 3 | fewer positions an hour from the slower network; 245,760 new positions per generation against 256,000; generation-keyed schedules unchanged, so they arrive after 0.72x the steps |
| Value discount | 0.998 per remaining ply on targets, 0.99 per ply in search | none in either | AlphaZero, KataGo and Lc0 do not discount; evaluation search changes with it |
| Replay window | 600K -> 20M rows in ten stages by generation 1000 | the same endpoints along ln(1 + generation / 50) | continuous growth; within 12% of the stages at their generations up to 100, then above the flat 8M stretch: 14.6M against 12M at 400, 17.9M against 16M at 700 |
| Candidate | — | from scratch, starts at 20 Elo/h, catch-up 0.1 -> 0.03 (V98), promoted after two consecutive candidate matches at 0.48 or better | |
| Dependency lock | `bffc5dad...` | same, inherited | the lock with the lc0 and publication extras and ModelOpt's ONNX dependencies; the run checks it at start |

The medium model trains from scratch on the shared replay, on its own generation clock, with 1.5x the active
model's steps. After promotion it follows the main learning rate at the run's generation.

## Hardware

**8x RTX 4070 SUPER**, 70-85 cents an hour: the baseline curve's hardware, so curves compare by time, and Ada, so
FP8 can be tested. Needs at least 80 effective CPUs and 200 GiB RAM **as granted by the cgroup quota** (check
`/sys/fs/cgroup`, not `nproc` or `free`), 150 GB disk, and a TensorRT-enabled native build. Update
`run.hardware.offer_id` to the rented offer.

8x RTX 3060 is cheaper but slower, cannot run FP8, and would need its own CNN baseline.

## Before the start, on the node

1. Provision with `deployment/setup_remote.sh`; record the cgroup CPU and memory grant and confirm the native
   extension has TensorRT.
2. **Throughput matrix**, forward-only engines at batch 320 and native self-play at the live settings for the
   leaders: 8x160, 10x192, 12x192 and 10x224 in float16 and FP8, against the 14x160 CNN in INT8. The standard
   build no longer records Lc0's board history (it is behind `-DCHESS_LC0_HISTORY=ON`), so no search-speed check
   of it is needed.
3. **FP8 fidelity** on the trained 10x192 (`evidence-lc0arch-and-throughput.tgz`) with
   `tools/compare_tensorrt_low_precision.py --precision fp8 --precision int8`: float16, FP8 and weight-only INT8
   engines from one TorchScript model, forward throughput at batch 320, and policy KL, top-move agreement and WDL
   difference against float16 on evaluation positions. Only activation-by-weight products are quantized, so the
   fused attention kernel survives; the heads stay float16. Its FP8 path has not yet run on hardware. A 100-game
   FP8-against-float16 match at 64 searches follows if the metrics are close. FP8 is worth building into
   production (a new template kind, per-publication scale calibration and refit) only at 1.2x or better with no
   measurable strength loss.
4. **Choose the medium size and precision**; update the configuration and record its new hash.
5. **Engines:** build the four templates (both sizes at batch 320 and 64) with
   `tools/build_tensorrt_refit_template.py` and test a refit with different weights; the attention engines are
   Myelin-compiled and their refit has never been exercised.
6. **Smoke:** one publication through the production trainer with the seven auxiliary heads, exported and served.
7. Start through `deployment/run_control.sh` after approval.

## Risks

- **The ladder is not measured under the baseline's search.** Evaluation matches inherit the self-play search
  discount, so removing it changes the yardstick's search as well as the network. The effect of a 0.99 backup
  discount at 64 searches is unmeasured; the comparison with the convolutional curve carries it.
- **SGD on a post-norm transformer is untested here**; every attention network in this project trained with AdamW.
  The 1,000-step warm-up and gradient clipping at 1.0 mitigate it. Watch the first 20 generations' losses; AdamW is
  the fallback.
- **The candidate-start latch can fire on noise** and is irreversible; check the gain rate before trusting it.
- **Promotion drops the learning rate** from the candidate's catch-up value (0.03 at its floor) to the main schedule
  at the run's generation.
- **Throughput:** at 0.4-0.6x the CNN's serving rate the architecture may not win at equal time even though it wins
  at equal searches. That is what the run measures.

## Also fixed on the way

The final recipe's evaluation list lacked the `progressive_candidate` match its promotion gate names, so as written
no candidate could be promoted (the lineage that ran carried it from V98). It is restored on master and in this
configuration.
