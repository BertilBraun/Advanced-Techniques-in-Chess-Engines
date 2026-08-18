# Two-GPU DDP attention training on RTX 4070 SUPER

## Question

Measure whether `static_graph=True` together with `gradient_as_bucket_view=True` improves compiled BF16 attention
training and removes the shared 1x1-head gradient-stride warning.

## Protocol

- Two NVIDIA GeForce RTX 4070 SUPER GPUs, one DDP rank per GPU.
- Local batch 256 and global batch 512.
- Compiled 3,624,336-parameter attention model with automatic SDPA dispatch.
- Ten warmup optimizer steps followed by 15 seconds of equal-wall-time measurement.
- Frozen replay SHA-256: `6ff14bf9f605132c788e325351ea423f9315d96881c0bab8551af4aeb3f897f2`.
- Two paired replicates in reversed order; both GPUs were confirmed idle before each pair.

## Results

| DDP mode | Replicate 1 samples/s | Replicate 2 samples/s | Mean samples/s |
| --- | ---: | ---: | ---: |
| Standard | 16,238 | 15,740 | 15,989 |
| Static graph and gradient bucket views | 15,270 | 15,942 | 15,606 |

The combined flags were 2.4% slower by the paired mean, with run-to-run variation larger than any plausible gain.
They also emitted the same gradient-stride warning on both ranks. The production trainer should retain ordinary DDP.

The warning identifies a compiled gradient for a shared 1x1 projection with shape `[4, 256, 1, 1]`. Its stride is
`[256, 1, 256, 256]`, while DDP's bucket view uses `[256, 1, 1, 1]`. Because both spatial dimensions are singleton,
the layouts have identical element ordering; the mismatch is metadata rather than a numerical or architectural
error.

No production configuration or service was changed.
