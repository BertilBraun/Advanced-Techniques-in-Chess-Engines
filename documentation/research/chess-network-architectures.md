# Chess convolutional and attention architecture study

## Scope and evidence

This study defines CPU-validated model candidates only. It does not authorize a production configuration change,
GPU benchmark, training run, queue update, checkpoint import, or live-service operation.

The primary AlphaZero chess source uses a deep residual convolutional network over spatial input planes, with a
policy output trained against MCTS visits and a value output trained against the game result. The paper emphasizes
that chess contains asymmetric rules and long-range interactions, making it less naturally matched to local,
translation-invariant convolutions than Go. It used a 20-block, 256-channel residual tower for its reported chess
system and a batch size of 4,096. See [Silver et al., 2017](https://arxiv.org/abs/1712.01815).

Lc0's current training source supports either residual blocks or an attention body, not both. Its attention body
turns the 8x8 board into 64 square tokens, adds square positional features, embeds each square, and applies repeated
multi-head self-attention and feed-forward encoder blocks. The attention-body path retains separate policy, WDL,
and moves-left heads. Lc0 also supports an attention policy mapping and optional Smolgen attention-logit generation;
those are distinct from the encoder body. See the primary
[Lc0 training implementation](https://github.com/LeelaChessZero/lczero-training/blob/master/tf/tfprocess.py#L1303-L1533),
[network serialization](https://github.com/LeelaChessZero/lczero-training/blob/master/tf/net.py), and
[Lc0 attention-body release history](https://github.com/LeelaChessZero/lc0/blob/master/changelog.txt).

## Implemented comparison boundary

The convolutional variant is the existing project model, including its selectable disabled, squeeze-excitation, or
global-pooling residual context. Its module names and computation remain unchanged so existing CNN behavior is the
control.

The attention variant is a mutually exclusive square-token backbone:

1. project input planes are transposed to one token per board point;
2. one linear projection creates the token embedding;
3. learned row and column embeddings provide board position;
4. pre-normalized multi-head self-attention and GELU feed-forward blocks update all tokens;
5. tokens are reshaped back to spatial features;
6. the existing project heads produce the canonical flat policy, three-way WDL, next-policy, and remaining-length
   outputs.

This intentionally does not copy Lc0's chess-specific from-square/to-square attention policy map, DeepNorm,
Smolgen, or current large-network scale. Keeping the project's existing heads isolates the backbone comparison and
preserves replay action IDs, objectives, JIT `(policy, WDL)` inference output, and native search integration.

## Frozen model catalog

The authoritative definitions are in `py/configs/architectures/chess-cnn-attention-v1.yaml`. Counts include the
training backbone and all four canonical heads: policy (1,880), WDL (3), next-policy (1,880), and remaining game
length (1).

| Definition | Backbone | Body | Exact parameters |
| --- | --- | --- | ---: |
| `chess-cnn-1m` | convolutional global pooling | 2 blocks x 32 channels | 1,017,032 |
| `chess-attention-1m` | square-token attention | 2 layers x 64, 4 heads, FFN 128 | 1,043,856 |
| `chess-cnn-4m` | convolutional global pooling | 10 blocks x 128 channels | 3,809,520 |
| `chess-attention-4m` | square-token attention | 5 layers x 256, 8 heads, FFN 512 | 3,624,336 |
| `chess-cnn-9m` | convolutional global pooling | 16 blocks x 176 channels | 9,490,464 |
| `chess-attention-9m` | square-token attention | 7 layers x 384, 12 heads, FFN 768 | 9,283,856 |

The 1M band is dominated by the two 1,880-action policy heads. That limits how closely either backbone can approach
one million parameters without changing the fixed experimental head contract.

## Benchmark protocol awaiting authorization

The frozen plan is `py/configs/benchmarks/chess-architecture-v1.yaml`; the gated runner is
`py/tools/benchmark_chess_architectures.py`. `describe` is read-only. `run` refuses to start without
`--acknowledge-gpu-load` and must be launched with eight `torchrun` ranks.

The plan fixes:

- global training batch 2,048 as eight local batches of 256;
- bfloat16 training and the production eight-device trainer topology;
- two self-play processes per device, two inference workers per process, and two outstanding batches per worker as
  recorded topology metadata;
- inference batches 1, 8, 32, 64, 128, and 256, with 64 identified as production;
- per-rank peak allocated and reserved CUDA memory;
- an equal-sample comparison of 128 optimizer steps, or 262,144 global samples;
- a 1,800-second equal-wall-time comparison;
- identical samples from a caller-supplied immutable NPZ replay snapshot containing `states`, `policy_targets`,
  `wdl_targets`, `next_policy_targets`, and `remaining_length_targets`.

After explicit authorization and after copying a frozen replay snapshot away from production artifacts, one model
and protocol can be measured with:

```powershell
torchrun --standalone --nproc_per_node=8 -m tools.benchmark_chess_architectures `
  --plan configs/benchmarks/chess-architecture-v1.yaml run chess-attention-4m `
  --protocol equal_samples --frozen-replay C:\benchmark-data\chess-replay-v1.npz `
  --output C:\benchmark-results\chess-attention-4m-equal-samples.json `
  --acknowledge-gpu-load
```

No command above has been run as part of this implementation.
