# Final progressive Chess inference on RTX 4070 SUPER

This is the acceptance measurement for the clean-run `6x96 -> 10x160 -> 15x192` attention progression at exact
source revision `401463904bad682667c3d267db5ad7ec8baf37f6`. The checkout was clean. GPU 0 was an idle NVIDIA
GeForce RTX 4070 SUPER under PyTorch 2.12.1+cu126 and CUDA 12.6.

The measurement matches configured production model execution: fused TorchScript, BF16, memory-efficient SDPA,
batch 64, twenty warmups, and a calibrated five-second interval. The network emits 4,864 raw direct-policy logits;
native search performs legal-action selection and normalization outside this model-only timing.

| Stage | Architecture | Training parameters | Inference parameters | Batch-64 positions/s | Batch latency |
| --- | --- | ---: | ---: | ---: | ---: |
| Initial | 6x96, 3 heads, FFN 192 | 474,754 | 467,219 | **53,747** | 1.191 ms |
| Intermediate | 10x160, 5 heads, FFN 320 | 2,104,642 | 2,092,179 | **36,706** | 1.744 ms |
| Final | 15x192, 6 heads, FFN 384 | 4,500,898 | 4,485,971 | **26,127** | 2.450 ms |

The intermediate model is approximately level with the former 10x128 CNN direct-policy control at 35,647
positions/s. The final model trades about 29% of that throughput for 15 attention layers and a 4.465M-parameter
shared backbone. Learned training heads occupy only 36,322 parameters, or 0.81% of the final model.

The machine-readable report is `results.json`. Its SHA-256 is
`878145cd67d7f566353a938c1191b542723861daf8bd06f9bd9bfa0fa69ae5d5`.
