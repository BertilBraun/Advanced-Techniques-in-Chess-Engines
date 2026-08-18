# Direct-policy progressive Chess inference on RTX 4070 SUPER

> This eager/compiled diagnostic is not the production self-play comparison. Production exports fused TorchScript
> and uses automatic SDPA dispatch. The corrected controls are recorded in
> `../chess-direct-policy-kernel-controls-rtx4070s-20260818/README.md`.

This run measures the three clean-run attention stages at exact source revision
`5095255a8e9211e1a6ed4b041911406fe050fa56`. The checkout was clean. GPU 0 was an idle NVIDIA GeForce RTX 4070
SUPER under PyTorch 2.12.1+cu126 and CUDA 12.6. Each point used ten warmups and a calibrated five-second BF16
measurement with the memory-efficient SDPA backend.

The network emits 4,864 raw `chess_76_plane_direct_v2` logits. Native search applies softmax only to legal action
IDs, which is outside this model-only measurement.

## Results

| Model | Training parameters | Batch | Eager positions/s | Compiled positions/s |
| --- | ---: | ---: | ---: | ---: |
| 6x96 | 474,754 | 1 | 532 | 695 |
| 6x96 | 474,754 | 16 | 8,410 | 10,677 |
| 6x96 | 474,754 | 64 | 33,456 | 42,573 |
| 6x96 | 474,754 | 256 | 134,368 | 171,538 |
| 10x160 | 2,104,642 | 1 | 329 | 452 |
| 10x160 | 2,104,642 | 16 | 5,196 | 7,178 |
| 10x160 | 2,104,642 | 64 | 21,278 | 28,093 |
| 10x160 | 2,104,642 | 256 | 76,300 | 104,214 |
| 16x192 | 4,797,922 | 1 | 219 | 300 |
| 16x192 | 4,797,922 | 16 | 3,611 | 4,762 |
| 16x192 | 4,797,922 | 64 | 13,790 | 18,356 |
| 16x192 | 4,797,922 | 256 | 40,849 | 54,572 |

At the production batch of 64, compilation improves throughput by 27.3%, 32.0%, and 33.1% for the small, medium,
and large stages respectively.

## Interpretation

The 16x192 large stage places 4,761,600 of its 4,797,922 training parameters in the shared backbone and keeps all
learned heads to 36,322 parameters, but its depth is costly. Its compiled batch-64 rate is 18,356 positions/s. The
earlier 5x256, 3.62M attention control measured approximately 59,800 positions/s at batch 64 on the same GPU and
framework. That historical control used the obsolete 1,880-action dense-head ABI and a 100-batch scripted protocol,
so the rates are not a strict isolated head comparison; they nevertheless show that the current 16-layer stage is
not an acceptable throughput-neutral replacement. A shallower large-stage layout needs a direct-policy benchmark
before the production configuration is accepted.

The machine-readable report and its complete parameter, runtime, timing, and memory fields are in `results.json`.
Its SHA-256 is `d191a715ffbcab82deb5c1b97909581ea70287f89d12302dfd1ee149cb2b60f1`.
