# Chess attention throughput on RTX 4070 SUPER

## Question

Repeat the 4M CNN versus packed-attention comparison on Ada hardware, identify the best SDPA backend for the fixed
64-square workload, and measure the native batch-64 path used by self-play rather than only a GPU-resident Python
forward.

## Hardware and runtime

- GPU 0 only: NVIDIA GeForce RTX 4070 SUPER 12 GiB, compute capability 8.9, 220 W limit.
- Driver 595.71.05; PyTorch 2.12.1+cu126; cuDNN 9.10.2.
- Source revision `f77645fa58e4148fb3c03f6fac98bc009696eb82`.
- Release native benchmark compiled for `sm_89`.
- GPU 1 was not used or controlled. An unrelated MCGS benchmark was active there during at least the later runs and
  shared the node's 23.04-CPU quota. The long decisive comparison therefore alternated CNN and attention on GPU 0.

The exported models use deterministic random seed 20260818 and the production inference path: auxiliary heads are
removed, eligible CNN modules are fused, the model is scripted, and the native runner converts parameters and inputs
to BF16.

## Native inference results

All figures are complete native inference rates including int8-to-BF16 device input copy and float32 policy/WDL
output copy. `processed_replicas` additionally performs legal-policy processing with four concurrent model runners,
matching the configured two self-play processes by two inference workers per GPU.

### Direct, one worker

Each result is the mean of three 320,000-position replicates.

| 4M model | SDPA backend | Positions/s | Relative to CNN |
| --- | --- | ---: | ---: |
| CNN | Automatic | 33,686 | 100.0% |
| Attention | Automatic | 57,107 | 169.5% |
| Attention | Flash | 55,197 | 163.9% |
| Attention | Memory-efficient | 57,116 | 169.6% |
| Attention | cuDNN | 54,354 | 161.4% |

### Four processed replicas

Each result is the mean of three 512,000-position replicates.

| 4M model | SDPA backend | Positions/s | Relative to CNN |
| --- | --- | ---: | ---: |
| CNN | Automatic | 71,056 | 100.0% |
| Attention | Automatic | 86,851 | 122.2% |
| Attention | Flash | 86,353 | 121.5% |
| Attention | Memory-efficient | 95,949 | 135.0% |
| Attention | cuDNN | 86,319 | 121.5% |

The shorter CNN replicates varied more than the attention replicates. Two longer interleaved 1.28-million-position
replicates measured 69,860 positions/s for CNN and 97,822 positions/s for memory-efficient attention: a 40.0%
attention advantage. Memory-efficient SDPA is the clear backend choice for concurrent batch-64 inference on this
GPU; cuDNN offers no gain over Flash here.

## Training and scripted-forward control

The single-GPU plan preserves the requested global/local training batch of 2,048 and uses BF16 mixed precision.

| 4M model | Training samples/s | Relative to CNN | Scripted batch-64 positions/s |
| --- | ---: | ---: | ---: |
| CNN | 19,876 | 100.0% | 35,528 |
| Attention | 11,880 | 59.8% | 59,975 |

At this unusually large per-GPU training batch, attention training is 40.2% slower. That does not offset its native
self-play advantage when inference supply is the limiting resource, but training and self-play must be weighted by
their actual wall-time shares before choosing the architecture.

## Conclusion

The RTX 4070 SUPER is substantially more favorable to this attention architecture than the RTX 3060. The earlier
fear of a large self-play slowdown does not apply: on this node, packed attention is faster than the CNN in every
native batch-64 inference mode tested, and forcing memory-efficient SDPA raises the production-like four-worker lead
to approximately 40% in the longer paired comparison.

No production configuration was changed. Selecting a nonautomatic SDPA backend in the real inference runner remains
an explicit implementation and review decision.

## Artifacts

- `native-results.json`: raw native replicate rates, hardware identity, model hashes, and protocol sizes.
- `chess-cnn-4m-4070s-gpu0-15s.json`: single-GPU training/scripted-forward control.
- `chess-attention-4m-4070s-gpu0-15s.json`: matching attention control.
