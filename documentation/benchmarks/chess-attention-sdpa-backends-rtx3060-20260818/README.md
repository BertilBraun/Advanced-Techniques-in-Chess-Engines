# Chess attention SDPA backend diagnostic on RTX 3060

## Question

Test whether FlashAttention is counterproductive for the fixed 64-square, batch-64 chess attention workload, and
verify the inference precision used by the earlier architecture comparison against the native production runner.

## Finding

The earlier Python architecture harness declared BF16 training but left its inference model and input in FP32. The
native CUDA runner instead calls `model.to(torch::kBFloat16)` and allocates a BF16 device-input buffer. FlashAttention
cannot run on the FP32 diagnostic tensors, so the reported 28% attention inference deficit was already produced
without FlashAttention and was not representative of production inference.

The harness now follows the inference export path before measurement: it removes auxiliary heads, fuses eligible
CNN modules, scripts the model, and converts both model and inputs to BF16.

## Protocol

- Revision `c470fb8bfb44ecb2f0ce355036ebfa8d97e5b41d` validation snapshot.
- One clean RTX 3060 12 GiB (GPU 0), PyTorch 2.12.1+cu126, CUDA runtime 12.6, driver 580.126.20.
- Complete 4M model forward, batch 64, random initialized weights and zero inputs.
- TorchScript models converted directly to BF16, matching the native inference runner.
- 50 warmup forwards and 500 measured forwards per replicate; three replicates.
- SDPA backends forced individually with `torch.nn.attention.sdpa_kernel`, preventing fallback.
- No production run, evaluation, or other GPU process was active.

This diagnostic times GPU-resident full-model forwards. Host/device transfers, result copies, native policy
processing, multiple inference workers, and the complete production topology remain to be measured before making a
deployment decision.

## Results

| 4M model | SDPA backend | Mean batch latency | Positions/s | Relative to CNN |
| --- | --- | ---: | ---: | ---: |
| CNN | Not applicable | 2.0718 ms | 30,890 | 100.0% |
| Attention | Automatic (FlashAttention) | 2.3143 ms | 27,655 | 89.5% |
| Attention | Memory-efficient | 2.1243 ms | 30,128 | 97.5% |
| Attention | cuDNN attention | 2.1191 ms | 30,202 | 97.8% |

Replicate latencies were stable within 0.5%. Disabling Flash and forcing cuDNN attention improved the complete
attention model by 9.2% relative to automatic Flash selection. Memory-efficient attention improved it by 8.9%.
The useful non-Flash backends reduce the attention deficit from 10.5% to 2.2-2.5% in this GPU-resident TorchScript
diagnostic.

For comparison, forced math SDPA under BF16 eager execution reached only 10,210 positions/s, so the unfused math
backend is not a viable alternative. The useful non-Flash choices are cuDNN and memory-efficient SDPA.

## Decision boundary

The prior 28% inference penalty should be withdrawn. Current evidence says automatic Flash selection is materially
suboptimal at 64 squares, while a forced cuDNN or memory-efficient backend puts attention close to parity for the
neural-network forward itself. A native end-to-end inference and self-play-topology comparison is still required
before claiming the same 2-3% deficit for production self-play.

The corrected eight-rank automatic-backend harness measured 185,841 aggregate attention positions/s and 207,651
aggregate CNN positions/s, also a 10.5% deficit. Its slowest-rank statistic was affected by a persistent unowned CUDA
context on GPU 1 after earlier diagnostics, so the clean GPU-0 replicates above are the backend-comparison evidence.
The eight-rank JSON artifacts are retained for inspection.

## Artifact hashes

| Result | SHA-256 |
| --- | --- |
| `chess-attention-4m-scripted-bf16-15s.json` | `e3a37bf821974bbab7f4b34792ed4b436c341d9d7f7839e7a7e6d2c876f5d16e` |
| `chess-cnn-4m-scripted-bf16-15s.json` | `736b630d7843efe65d1f705c09dd8bb82725aa3e1f2cfcd529342948fa406b07` |
