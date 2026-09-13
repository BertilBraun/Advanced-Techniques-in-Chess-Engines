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

Results are pending.
