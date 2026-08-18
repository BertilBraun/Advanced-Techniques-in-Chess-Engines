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
6. compact spatial policy heads produce 76 move planes and gather the 1,880 canonical action logits;
7. the value and remaining-length heads retain their canonical outputs.

The spatial policy layout contains 56 ray planes, eight knight planes, and 12 explicit promotion planes. The native
chess encoder remains the mapping authority and exposes one immutable gather index per canonical action. Both the
primary and next-policy heads use `backbone -> 32-channel 1x1 projection -> normalization/activation -> 76-channel
1x1 projection -> fixed gather`. Replay action IDs, sparse targets, losses, JIT `(policy, WDL)` inference output,
and native search integration therefore remain unchanged. Impossible spatial slots never enter the canonical
softmax. Legal-action masking remains search-owned; legal-move prediction is deliberately outside this rework.
The architecture manifest identifies these exact semantics with the versioned `chess_76_plane_v1` policy-head
variant. Native gather indices are reconstructed when loading raw weights rather than accepted from a checkpoint.

## Frozen model catalog

The authoritative definitions are in `py/configs/architectures/chess-cnn-attention-v1.yaml`. Counts include the
training backbone and all four canonical heads: policy (1,880), WDL (3), next-policy (1,880), and remaining game
length (1).

| Definition | Backbone | Body | Exact parameters |
| --- | --- | --- | ---: |
| `chess-cnn-1m` | convolutional global pooling | 8 blocks x 88 channels | 1,100,730 |
| `chess-attention-1m` | square-token attention | 8 layers x 128, 4 heads, FFN 256 | 1,086,114 |
| `chess-cnn-4m` | convolutional global pooling | 10 blocks x 144 channels | 3,603,454 |
| `chess-attention-4m` | square-token attention | 13 layers x 192, 6 heads, FFN 384 | 3,894,946 |
| `chess-cnn-9m` | convolutional global pooling | 16 blocks x 176 channels | 8,538,402 |
| `chess-attention-9m` | square-token attention | 17 layers x 256, 8 heads, FFN 512 | 9,001,762 |

The learned primary policy, WDL, next-policy, and remaining-length heads total 20,130, 24,418, and 28,706 parameters
for the three attention widths. The backbone therefore owns at least 98% of every attention candidate's capacity.
The fixed native gather indices are registered model buffers and are not trainable parameters.

The benchmark artifacts below predate the spatial policy rework and describe the former shallow models with dense
policy projections. They remain historical evidence for attention-kernel and SDPA decisions, not throughput claims
for this catalog. The revised models require a separately authorized benchmark before production use.

## Benchmark protocol and first contended comparison

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

The explicitly authorized first comparison used a separate 120-second contention plan while the confirmatory chess
evaluation shared all eight GPUs. All six primary runs completed without controlling that evaluation. Results and raw
JSON artifacts are recorded in
`documentation/benchmarks/chess-architecture-contended-rtx3060-20260817/README.md`. The CNN controls delivered
15-29% more training samples per second and used 1.9-2.4x less peak allocated training memory than the attention
models at the matched parameter bands. These throughput results do not compare playing strength; frozen-replay
equal-sample and equal-wall-time training-quality experiments remain separate work.

Profiling the 4M attention model identified avoidable batch-first/sequence-first materialization inside generic
`nn.MultiheadAttention`. Revision `c470fb8b` replaces it with a packed-QKV projection and direct scaled dot-product
attention. The short uncontended comparison in
`documentation/benchmarks/chess-attention-packed-qkv-rtx3060-20260818/README.md` found 14.6% higher attention
training throughput, reducing its deficit to the CNN from 24.8% to 13.9%. Memory and batch-64 inference did not
improve, so the RTX 3060 efficiency tradeoff remains material.
