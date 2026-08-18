# Chess progressive-model inference benchmark

The active Chess throughput harness benchmarks the three progressive attention models directly from
`py/configs/production/vast-chess-8gpu-optimal.yaml`. There is no separate architecture catalog, benchmark plan,
synthetic replay, or benchmark checkpoint. The network emits raw `chess_76_plane_direct_v2` policy logits; legal
move selection and softmax normalization remain native search responsibilities and are outside this model-only
measurement.

From `py/`, an explicitly authorized run on GPU 0 is:

```powershell
uv run python -m tools.benchmark_chess_inference `
  --gpu-id 0 `
  --batch-sizes 1,16,64,256 `
  --modes eager,compiled `
  --warmup-iterations 10 `
  --duration-seconds 5 `
  --output benchmark-results/chess-progressive-inference.json `
  --acknowledge-gpu-load
```

The command refuses to touch CUDA without `--acknowledge-gpu-load`. Its typed JSON report records the exact Git
revision and dirty state, production-config SHA-256, full model definitions, backbone and head parameter counts,
GPU and framework versions, peak CUDA memory, elapsed time, and positions per second for every model/mode/batch.
Batch 64 is included by default because it is the production search batch.

The retained configuration's exact parameter counts are pinned by the harness test:

| Model | Training total | Inference total | Backbone | Primary policy | Value | Auxiliary |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `chess-attention-500k` | 474,754 | 467,219 | 453,312 | 7,372 | 6,535 | 7,535 |
| `chess-attention-2m` | 2,104,642 | 2,092,179 | 2,073,280 | 12,236 | 6,663 | 12,463 |
| `chess-attention-5m` | 4,797,922 | 4,782,995 | 4,761,600 | 14,668 | 6,727 | 14,927 |

The inference model excludes the next-policy and remaining-game-length auxiliary heads. The primary Chess policy
head is a single 76-plane projection and emits 4,864 raw logits.
