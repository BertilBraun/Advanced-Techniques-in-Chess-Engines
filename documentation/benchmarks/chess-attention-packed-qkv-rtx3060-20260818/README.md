# Packed-QKV attention optimization on RTX 3060

> **Inference correction:** the batch-64 inference figures below were measured with FP32 model parameters and
> inputs even though CUDA production inference converts both to BF16. They are retained as historical artifacts but
> must not be used to estimate self-play throughput. The corrected BF16 investigation is recorded in
> [`../chess-attention-sdpa-backends-rtx3060-20260818/README.md`](../chess-attention-sdpa-backends-rtx3060-20260818/README.md).

## Question

The first contended comparison found the 4M attention model trained 29% slower than its parameter-matched CNN
control. Profiling showed that `nn.MultiheadAttention` was using flash attention correctly, but layout copies,
LayerNorm, and fragmented elementwise/GEMM work dominated its runtime. This experiment tests whether a direct
packed-QKV projection feeding scaled dot-product attention materially closes that gap on the existing hardware.

## Change

Candidate revision `c470fb8bfb44ecb2f0ce355036ebfa8d97e5b41d` replaces each `nn.MultiheadAttention` with:

1. one linear projection from the embedding to packed query, key, and value tensors;
2. a view into `[batch, heads, squares, head_size]` without sequence-first conversion;
3. `torch.nn.functional.scaled_dot_product_attention`;
4. one output projection back to the token embedding.

Architecture configuration, exact parameter counts, canonical heads, replay contract, and CNN code are unchanged.
Internal attention state-dict key names change, so checkpoints from the earlier experimental attention implementation
are not directly loadable.

## Short uncontended protocol

- Eight RTX 3060 DDP ranks, global batch 2,048, local batch 256, bfloat16 autocast.
- Ten warmup optimizer steps followed by a 15-second equal-wall-time measurement.
- Same deterministic 8,192-sample frozen replay used by the earlier comparison; SHA-256
  `6ff14bf9f605132c788e325351ea423f9315d96881c0bab8551af4aeb3f897f2`.
- No evaluation or other project GPU workload was running.
- Original attention uses revision `e35a9a0b`; packed attention and the unchanged CNN control use `c470fb8b`.

## Results

| 4M implementation | Training samples/s | Relative to original attention | Relative to CNN | Peak allocated MiB | Batch-64 positions/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original attention | 31,606 | 100.0% | 75.18% | 741.58 | 93,480 |
| Packed-QKV attention | 36,216 | 114.59% | 86.15% | 742.25 | 93,009 |
| CNN control | 42,039 | 133.01% | 100.0% | 305.56 | 128,732 |

Packed QKV improves attention training throughput by 14.6% and cuts the deficit to CNN from 24.8% to 13.9%.
It does not reduce peak training memory, and it does not improve the short batch-64 inference diagnostic. The
remaining training gap is consistent with flash-attention backward, LayerNorm, feed-forward, residual, and optimizer
kernel fragmentation rather than an attention fallback or incorrect tensor shape.

The optimization is worth retaining as the cleaner attention primitive. It leaves a 14% training-throughput deficit
at this scale. The inference and self-play conclusion originally drawn from this table is invalid because of the FP32
benchmark mismatch described above.

## Artifact hashes

| Result | SHA-256 |
| --- | --- |
| `chess-attention-4m-original-uncontended-15s.json` | `e8aae7f6fbf5a2392d7d532ae6e4a7e69933d1971f7d91181dc891eb3d329408` |
| `chess-attention-4m-packed-qkv-uncontended-15s.json` | `ae96d64334072e370578a714f61682ccdde5f38d3ace3fab0abc708a15e7550e` |
| `chess-cnn-4m-uncontended-15s.json` | `1975c29591dc1e6b8e358d1d8863d22b8510c38af278c2665f01f547e27986e2` |
