# Chess attention training at batch 256 on RTX 4070 SUPER

## Question

Determine whether the attention model's poor single-GPU training result at local batch 2,048 also applies to the
intended local batch of 256, and measure two relevant optimizations: explicit SDPA backend selection and fixed-shape
`torch.compile` training.

## Hardware and protocol

- GPU 0 only: NVIDIA GeForce RTX 4070 SUPER 12 GiB.
- PyTorch 2.12.1+cu126; CUDA 12.6.
- Source revision `769ca4e907ec2fb20bc1e688f49382bf425064f2`.
- BF16 mixed precision, AdamW, one process, global/local batch 256.
- Ten warmup optimizer steps followed by 15 seconds of equal-wall-time measurement.
- Frozen replay SHA-256: `6ff14bf9f605132c788e325351ea423f9315d96881c0bab8551af4aeb3f897f2`.
- GPU 1 was not used or controlled. An unrelated benchmark was active there and shared the node's CPU quota.

The 4M labels identify the existing catalog controls: the CNN has 3,864,208 parameters and attention has 3,624,336.

## Results

| 4M model | Training mode | SDPA backend | Samples/s | Relative to eager CNN |
| --- | --- | --- | ---: | ---: |
| CNN | Eager | Automatic | 5,628 | 100.0% |
| Attention | Eager | Automatic | 6,081 | 108.0% |
| Attention | Eager | cuDNN | 6,142 | 109.1% |
| Attention | Eager | Memory-efficient | 6,269 | 111.4% |
| CNN | Compiled, mean of two | Automatic | 5,944 | 105.6% |
| Attention | Compiled, mean of two | Automatic | 8,171 | 145.2% |
| Attention | Compiled | Memory-efficient | 7,675 | 136.4% |

The two compiled automatic replicates were 5,849 and 6,038 samples/s for CNN, and 8,147 and 8,194 samples/s for
attention. The mean compiled attention rate is 37.5% above compiled CNN and 34.4% above eager automatic attention.

## Interpretation

The batch-2,048 slowdown is not representative of the intended single-GPU training shape. At batch 256, attention
is already 8.0% faster in eager automatic mode. Memory-efficient SDPA adds 3.1% to eager attention, while fixed-shape
compilation provides the decisive improvement.

Automatic SDPA dispatch is preferable once the training network is compiled. Forcing memory-efficient SDPA reduces
compiled attention throughput by 6.1% in this measurement, despite being the best eager backend. Backend choices
therefore need to remain independent for compiled training and native inference.

Both compiled models emitted a DDP warning about gradient strides in a small convolutional head. It is not
attention-specific and did not prevent execution, but correcting that layout may offer a smaller additional gain.
Compilation startup and graph-cache behavior were outside the timed region; production adoption still needs a
startup/checkpoint-resume smoke and numerical-equivalence validation.

No production configuration or running service was changed.

## Artifacts

Each JSON file records the exact plan, model parameter count, runtime identity, backend, compilation state, frozen
replay hash, training rate, allocation watermark, and scripted inference control.
