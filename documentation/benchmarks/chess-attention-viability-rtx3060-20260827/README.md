# Chess attention viability — rtx3060, 2026-08-27

Can an attention trunk be made competitive with the convolutional one in this engine, now that the
generation-0 bootstrap defect that invalidated the two earlier attention runs is fixed?

**RESULTS PENDING — this file is written as the runs complete.**

| | |
| --- | --- |
| `experiment_configuration_sha256` | `c1ac342fbb33e959258814474f9709d33752f3833c99981e590837a8fd910e2d` (`py/configs/validation/distillation-probe-chess-single-gpu.yaml`, file sha256 `8caaca0ff5e4fbd90fcb8d566b53d5ebb7d630166666599751dbc463f8c45f11`) — used by the native inference measurement only |
| Cell definitions | `py/tools/attention_viability_cells.py` sha256 `0d13c3505d6189e285adae591fd07eb51844deb322b12745d367197b92f3b916`; resolved architectures sha256 `9a5dc37a54791f35a41bcb03a4b29f62f5f2955d60789b63d96d9edb1f9b0f0c` |
| Source revision | `<filled at completion>` (branch `attention-viability`, based on `distillation-probe` `6c62e605`) |
| Node | Vast.ai `50.120.65.61:41841` — 1× RTX 3060 12 GiB, driver 595.84, 56 effective CPUs, 62 GiB RAM |
| Runtime | Python 3.12.3, torch 2.12.1+cu126, Release native extension built at `e8bec367` |
| Dataset | `/workspace/distill/attention-6m.bin` — the distillation probe's 6,000,000-position chess set, teacher `vast-chess-4day-production-v8` generation 322, weights sha256 `dd514db199186f6e657593210751b7f6b1dce9b4e172129c9d2944e38b86f3df` |
| Date | 2026-08-27 |

## The two things that make every number here conditional

**Throughput measured on this node does not transfer to production.** This repository has already
measured that the attention-versus-convolution throughput verdict *inverts* between an RTX 3060 and the
RTX 4070 SUPER production hardware: 29% slower on a contended 3060, but 37.5% faster in compiled training
and +40% in native batch-64 inference on a 4070 SUPER, and 40% slower at single-GPU batch 2048
(`chess-attention-sdpa-backends-rtx3060-20260818`, `chess-attention-sdpa-backends-rtx4070s-20260818`,
`chess-architecture-contended-rtx3060-20260817`). Every samples/s, positions/s and MiB figure below is
labelled rtx3060 and **must not be compared against any 4070 SUPER number**. The axis that does transfer
is quality at matched parameters — held-out policy cross-entropy — and that is what the verdict rests on.

**The GPU was shared.** It was idle at 14:02 UTC. At about 14:33 UTC a `measure_policy_target_fidelity`
job belonging to separate `search-evaluations` work started on the same card and held it for the rest of
the session, driving it to 58% utilisation on its own. Every throughput number below is therefore
contended, on top of already being unusable for a production decision.

## Method

Supervised distillation from the frozen 6M-position dataset, one student per cell, using the production
`ResolvedTrainingObjective` unchanged: its policy term is soft-target cross-entropy against the teacher's
masked-softmax policy, and `root_value_blend` 0 makes the value term cross-entropy against the teacher's
WDL. No auxiliary heads. No MCTS anywhere in the loop.

The headline metric is **held-out policy cross-entropy minus the target-entropy floor of the held-out
set** ("gap above floor"). Raw cross-entropy is not comparable across datasets because the floor moves
with the generator; the gap is. The held-out split is the contiguous last 2% (120,000 positions) of the
same generator, so it measures generalisation to new positions from the same distribution.

Every cell shares one random seed, so initialisation and batch order are held fixed and a difference is
attributable to the architecture. The cost is that there is no estimate of seed variance.

### Deviations from the specified protocol, and why

| specified | run | reason |
| --- | --- | --- |
| 80,000 steps | **8,000 steps** | The GPU is shared with another workload and the eight cells needed to finish inside one session. At the contended rate the specified budget was 30+ hours. |
| — | evaluate every 500, checkpoint every 2,000 | a denser curve, so the ordering can be checked for stability rather than read off one endpoint |

Everything else is the probe-settled protocol: batch 1024, peak learning rate 2e-3, 200-step warm-up,
plateau schedule with the cosine anneal over the final 20%, gradient-norm bound 0.5. The step count is
identical across cells, so the comparison stays matched; what it no longer measures is *converged*
quality. Read every architecture number below as "at 8.2M training samples", not "at convergence".

### The dataset

The 6M set is the distillation probe's merged `chess-v8-g322-1m` + `chess-v8-g322-5m`. Two properties of
it are carried over from that probe and stated again here because they bound what these numbers mean:

1. **Every recorded position has the same side to move.** The builder recorded when
   `ply % recorded_ply_interval == 0` with an even interval, so only even plies were ever sampled. Chess
   inputs are canonical, which bounds the harm, but for any exchange sequence only one side of it was
   sampled. Fixed in the builder since; this file predates the fix.
2. **The teacher is weak.** v8 generation 322 is the run with the documented endgame-conversion defect,
   so absolute quality does not transfer. Only the relative architecture comparison does.

The file was written before `60bc8612` added the auxiliary-head fields, so its records are 1,019 bytes
rather than 1,409. The reader now names the record layout in the manifest; the core layout is a strict
prefix of the auxiliary one, so no field moved.

## The cells

All eight cells train on the same dataset with the same protocol. The four architecture cells are matched
on **total** parameter count including output heads, not on trunk parameters, because the buildable dense
policy head is a flat ~483,680 parameters regardless of trunk — 12% of a 3.9M model — and a from-to
attention head is far smaller. Each attention cell is therefore sized by depth at a fixed embedding of
176 with head dimension 16, so the head and the bias are the only things that change and the head savings
are spent on trunk depth.

| cell | trunk | policy head | attention bias | question |
| --- | --- | --- | --- | --- |
| `cnn-A` | 12x128 global-pooling convolutional | dense, 4 channels, no bottleneck | — | the baseline |
| `attn-A` | 14x176 attention, 11 heads, FFN 352 | dense, 4 channels, no bottleneck | none | was the ~650 plateau purely the bootstrap bug? |
| `attn-B` | 15x176 attention, 11 heads, FFN 352 | from-to attention, key size 128 | none | does the head that keeps all 64 square representations pay? |
| `attn-C` | 13x176 attention, 11 heads, FFN 352 | from-to attention, key size 128 | smolgen (8, 32, 32) | does a generated attention bias pay? |

The four learning-rate cells all use the `cnn-A` architecture and differ only in schedule and optimiser.

## Reproduce

From `py/` on the node, with the locked virtual environment:

```
# One cell. tools.attention_viability_cells prints the exact command line for any of the eight.
python -m tools.attention_viability_cells --cell attn-C --output-root /workspace/attention-cells \
  --python /workspace/alphazero-engine-venv/bin/python --steps 8000

# The generation-0 bootstrap prior of every architecture cell, on real positions.
python -m tools.measure_bootstrap_policy_prior --dataset /workspace/distill/attention-6m.bin \
  --output bootstrap-prior.json --scratch /tmp/bootstrap-scratch

# The results table, joined to the parameter and multiply-accumulate split of the same cell definitions.
python -m tools.attention_viability_report --log-root /workspace/attention-cells --output cells.json

# Native inference throughput at the batch size the comparison is stated at.
python -m tools.distill_match --mode throughput-only --throughput-position-count 64 \
  --parallel-searches 1 --teacher-run-state <cnn-A> --teacher-generation 322 \
  --student-run-state <attn-X> --student-generation 322 \
  --openings-manifest /workspace/evaluation-artifacts/chess/chess-elite-2025-11-balanced-4moves-200-v1-openings.json \
  --experiment-config configs/validation/distillation-probe-chess-single-gpu.yaml --output <json>
```

## Files

| file | contents |
| --- | --- |
| `bootstrap-prior.json` | generation-0 exported policy prior of each architecture cell on 256 real positions |
