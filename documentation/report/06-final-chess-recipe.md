# 6. Final chess recipe

## Canonical entry point

The reproduction entry point is
[`py/configs/production/chess-final-config.yaml`](../../py/configs/production/chess-final-config.yaml). It is fully
expanded and inherits from no other YAML file. Documentation should link to that file instead of copying every field,
because the living recipe may be revised if the project resumes.

The current final training run was developed iteratively across V89–V93. Those versions include operational resumes
and fixes; they do not redefine the intended recipe as separate scientific runs. The terminal report must freeze the
actual source revision and resolved configuration used by the reported checkpoint.

## Recipe summary

| Component | Settled choice |
| --- | --- |
| Hardware | 8x RTX 4070 SUPER, shared by training, self-play, and evaluation |
| Trainer | 8-rank NCCL DDP; global batch 2,048; bfloat16 |
| Optimizer | SGD, momentum 0.9, Nesterov, weight decay 0.0001, gradient norm cap 1.0 |
| Learning rate | 1,000-step warmup to 0.1; linear 0.1 to 0.01 over generations 0–1,000 |
| Quantization | TensorRT INT8 QAT, pre-fold deployment copy, per-generation recalibration |
| Model ladder | 12x128 → 14x160 → 19x176 convolutional networks |
| Context/head | Global pooling every second residual block; from-to attention policy head |
| Candidate timing | Searched-Elo plateau thresholds; loss-based promotion after catch-up |
| Replay | Staged 0.6M → 20M positions; reuse 4; eight materializers |
| Sampling | 30% uniform plus capped policy-surprise weighting |
| Search | 300 → 400 → 500 → 600 → 800 visits; reduced-parent FPU; forced playouts |
| Starts | 50% shallow random openings; 50% recent restart states |
| Resignation | Calibrated after generation 70, with 20% continuation games |
| Objectives | Policy 1.0, value 1.0, root-value blend, discounted value targets |
| Auxiliary targets | Next policy at 0.15; remaining game length at 0.1 |

This table is explanatory, not executable. The YAML remains authoritative.

## Progressive stages

All stages share the same semantic heads and objective, making loss comparison meaningful during catch-up:

1. **12 blocks × 128 channels.** The high-throughput bootstrap and early-data stage.
2. **14 blocks × 160 channels.** The medium stage used once early Elo gain per hour no longer justifies staying small.
3. **19 blocks × 176 channels.** The maximum configured stage, started under a lower plateau threshold.

Each block family uses scaled post-activation with an activation cap of 6. The branch scale decreases with depth.
The from-to policy key size is 128 in every stage, and the value head shape remains fixed.

## Data curriculum

The recipe increases three resources over time:

- search visits increase as the policy becomes capable of using deeper search;
- replay capacity grows from 600,000 to 20 million positions, limiting early stale-data dilution while preserving
  broader later experience;
- the network grows only when the current stage's measured improvement per hour falls below its threshold.

Meanwhile, game length caps increase from 150 to 250 plies, greedy move selection is delayed later in mature games,
and a root-value blend rises from zero to 0.1. These schedules are part of the data curriculum, not incidental knobs.

## Search and diversity

Every self-play position uses the generation's fixed visit cap; the rejected learned adaptive systems are absent.
Forced playouts broaden root exploration, while reduced-parent-value FPU discourages an unexplored move without
treating it as maximally bad. Dirichlet noise and temperature provide early exploration, with lower temperature after
the configured greedy threshold.

Random openings cover up to eight legal plies. Restart states must pass value, branch-mass, age, and remaining-length
filters, and selection retains a uniform component. This combination seeks variety without turning the entire start
distribution into a narrow hard-position curriculum.

## What the recipe does not prove

Presence in the final configuration does not mean every component has an isolated Elo estimate. Progressive sizing,
the from-to head, inference backends, and several search decisions have focused evidence. Restart states, the two
retained auxiliary heads, and interactions among replay growth, surprise weighting, and resignation are principally
supported as an assembled recipe. The final run evaluates the bundle.
