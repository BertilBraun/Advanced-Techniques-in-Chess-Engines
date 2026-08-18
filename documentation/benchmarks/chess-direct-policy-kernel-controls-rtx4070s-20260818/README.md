# Production Chess inference and SDPA controls on RTX 4070 SUPER

This experiment corrects the earlier eager/compiled comparison by measuring the actual fused TorchScript form
exported for self-play. It also reconstructs the former 10x128 CNN and 5x256 attention backbones with the new
direct 76-plane policy head, isolating the policy cleanup from architecture depth. All measurements used BF16,
batch 64, twenty warmups, and an idle NVIDIA GeForce RTX 4070 SUPER under PyTorch 2.12.1+cu126 and CUDA 12.6.

The automatic-dispatch measurements ran for five calibrated seconds. Forced-kernel measurements ran for three
seconds. Exact source revision: `08cff24d19cfe582e7588cd5ad1a8c60cbea822d`.

## Batch-64 positions per second

| Model | Training parameters | Automatic | Memory-efficient | Flash | cuDNN |
| --- | ---: | ---: | ---: | ---: | ---: |
| Former 10x128 CNN, direct head | 2,861,186 | 36,195 | 35,647 | 36,456 | 36,682 |
| Former 5x256 attention, direct head | 2,694,050 | 61,032 | 63,460 | 61,424 | 58,562 |
| Current 6x96 attention | 474,754 | 51,755 | 53,058 | 51,436 | 49,348 |
| Current 10x160 attention | 2,104,642 | 35,896 | 36,762 | 35,805 | 33,031 |
| Current 16x192 attention | 4,797,922 | 23,625 | 24,330 | 23,437 | 22,509 |

The former dense-head benchmarks on this same node reported approximately 35,900 positions/s for the 10x128 CNN
and 59,800 positions/s for the 5x256 attention model. Their direct-head controls now reach 36,195 and 61,032
positions/s respectively. The 4,864-logit direct policy ABI therefore did not cause the reported slowdown.

Memory-efficient SDPA is the best tested forced backend for every attention layout, improving automatic dispatch
by 2.4-4.0%. Flash is approximately level with automatic, and cuDNN is consistently slower. Kernel selection can
recover only a few percent; it cannot compensate for sequential depth.

The depth explanation is nearly exact. Historical 5x256 latency was about 1.067 ms. Scaling it by `16 / 5`
predicts 3.414 ms for sixteen layers, while the corrected current 16x192 automatic measurement is 2.709 ms and
the earlier compiled measurement was 3.487 ms. The production TorchScript path is faster than the earlier
compiled path, but the large model still executes sixteen normalization, attention, projection, residual, and
feed-forward sequences. Parameter count is not a latency-equivalent measure when width is exchanged for depth.

## Conclusion

Nothing in the direct policy implementation or CNN path shows a throughput regression. The bad comparison mixed
the wrong runtime with a model that was 3.2 times deeper. The production configuration retains TorchScript inference
and selects memory-efficient SDPA. After this control, the final architecture decision deliberately retained greater
depth for learning capacity: the selected 15x192 final stage measures 26.1k positions/s at batch 64. That final decision
and its exact three-stage measurements are recorded in
[the final progressive benchmark](../chess-direct-policy-final-progressive-rtx4070s-20260818/README.md).

Artifact SHA-256 values:

- `automatic.json`: `31ae37859a0bfa78f8be6bd59f159c1571092ad2229eb464af5aae4039fdabaf`
- `memory-efficient.json`: `20581ed7ee4e22ef05591ebde14fdba87ef6c89c2eebf6d64c24c13dfbdae29b`
- `flash.json`: `f9ff93047b0c7529108ac93f143bacc660a866880be79dd4c65f35a97de59704`
- `cudnn.json`: `4563b16ba89a6b56daa6c4489d725c747767d70a70e37eab3e0b307ca97c2b0c`
