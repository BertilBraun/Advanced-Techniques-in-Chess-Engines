# Chess SGD QAT replay screen on RTX 4070 SUPER

This bounded screen compares three from-scratch SGD+Nesterov schedules for the 12x128 scaled-post QAT network.
It is a frozen-replay screening proxy, not a self-play strength result.

## Controlled protocol

- Hardware: three concurrent two-GPU DDP arms on RTX 4070 SUPER pairs 0-1, 2-3, and 4-5. GPUs 6-7 were not used.
- Source revision: `46ab25b507481d1f0c8cfccbb1b15084f5be1d93`.
- Replay: the terminal v34 replay store, SHA-256
  `d668a18e07be099538a79a70b2c65c275a724b7dba56eddeb5629d0a7df52a83`.
- Split: the final 2% of logical replay rows is held out; 4,096 fixed rows at the start of that tail are evaluated.
- Initialization and sampling seed: `20260913`. Each arm draws the same 2,048 global replay indices per step and
  partitions them between ranks.
- Model: 12 residual blocks, width 128, scaled post-activation residual branch, global context every second block,
  chess from-to attention policy head with key size 128, and TensorRT INT8 QAT fake quantization.
- Optimizer: SGD, momentum 0.9, Nesterov enabled, weight decay 0.0001, global batch 2,048, and gradient norm clip 1.0.
- Common pre-fold phase: linear warmup from 0 to 0.1 over 1,000 optimizer steps. Batch normalization is folded at
  step 1,000, QAT ranges are immediately recalibrated on the same 256 replay positions, and the optimizer is reset.
- Runtime bound: 1,200 seconds per arm, with a maximum of 12,000 optimizer steps.

Only the deployment learning-rate schedule differs:

| Arm | GPU pair | Deployment schedule |
|---|---:|---|
| `v35_control` | 0-1 | 0.02 immediately after the fold |
| `lr_004_warm_2000` | 2-3 | 0.001 to 0.04 over 2,000 deployment steps |
| `lr_006_warm_3000` | 4-5 | 0.001 to 0.06 over 3,000 deployment steps |

The first arm reproduces v35's deployment LR behavior. The other arms test whether a post-fold warmup permits a
higher useful SGD rate without the immediate gradient shock.

## Results

All arms reached the 12,000-step cap before the 1,200-second wall-time limit. No arm produced a NaN or other
divergence. The initial held-out result was identical in all three arms (total loss 4.30639 and target top-action
agreement 6.2988%), and the sampled global-index sequence digest was identical:
`19b5fbfdba3500fb5b440fb96d17cb96c310e09781693db153906d89a35fb4e2`. A tensor-by-tensor comparison of the
three serialized initial states also found exact equality; the `.pt` file hashes themselves are not canonical
because Torch assigns different serialization storage identifiers.

| Deployment schedule | Final policy | Final WDL | Final total | Top-action agreement | Final-window mean / max gradient norm | Final-window clipped steps | Final-window samples/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.02 immediate | 2.01819 | 0.78178 | 2.79997 | 47.1191% | 1.8389 / 3.0225 | 100.00% | 25,318 |
| 0.001 to 0.04 over 2,000 steps | 1.98276 | **0.77034** | 2.75310 | 47.6807% | 1.3073 / 1.9609 | 100.00% | 25,344 |
| 0.001 to 0.06 over 3,000 steps | **1.95197** | 0.77093 | **2.72291** | **49.3408%** | **1.0197 / 1.4800** | **53.25%** | **26,333** |

The 0.06 arm beat the v35 control by 0.07706 total loss (2.75% relative), 0.06622 policy loss (3.28%), and 2.22
percentage points of top-action agreement. It beat the 0.04 arm by 0.03019 total loss and 1.66 agreement points.
The advantage was not an early transient: total losses for 0.02 / 0.04 / 0.06 were 2.8815 / 2.8664 / 2.8550 at
step 6,000, 2.8381 / 2.8164 / 2.7935 at step 8,000, and 2.8197 / 2.7846 / 2.7506 at step 10,000.

All three schedules experienced a large gradient shock after BatchNorm folding and optimizer reset. Every step in
the first deployment intervals was clipped even though the warmed arms started near 0.001. The shock therefore is
not caused by the selected deployment LR alone. The 0.06 arm recovered most strongly: its clipped fraction fell to
93.4% over steps 6,001-8,000, 74.9% over 8,001-10,000, and 53.25% over 10,001-12,000. The other arms remained at or
near 100% clipping.

## Recommendation

Use the 0.001 to 0.06 deployment warmup over 3,000 optimizer steps as the next self-play experiment recipe. Keep
the common 1,000-step pre-fold warmup to 0.1 and fold boundary for the cleanest translation of this result. This
screen gives no support for v39's 0.1 deployment target or its second 5,000-step warmup; it also shows that v35's
0.02 deployment rate leaves measurable replay-fitting speed unused.

Treat 0.06 as a candidate rather than a settled production optimum. The screen used one seed, a stationary terminal
v34 replay distribution, only the primary policy and WDL objectives, and 12,000 optimizer steps. It does not measure
on-policy feedback, chess Elo, replay freshness, auxiliary-head behavior, or longer-horizon stability. The arms ran
concurrently on different GPU pairs; their common pre-fold training losses agreed closely but were not forced into
deterministic CUDA kernels. The next run should watch post-fold clipping, policy-only Elo, searched Elo, and whether
fresh self-play data changes the ordering.

Raw reports, CSV observations, stdout/stderr logs, hardware inventory, and the independently recomputed replay hash
are under [`raw`](raw). The replay hash recomputation matched the previously recorded v34 digest.
