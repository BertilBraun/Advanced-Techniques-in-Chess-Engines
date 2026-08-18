# Monte-Carlo graph search on RTX 4070 SUPER

## Result

Graph search is not useful at 150–600 searches under this workload. Verified links were absent or negligible, no
neural evaluations were avoided, and graph bookkeeping cost 2.3–8.8% throughput. At 30,000–60,000 searches with 64
parallel reservations, verified links rose to 2.37–3.46%, but threshold-triggered evaluation avoidance remained
0.0001–0.0348% and graph throughput was 5.8–7.1% lower. Tree search should remain the default.

Both 30,000- and 60,000-search graph paths completed correctly with 64 parallel reservations. No cycle cutoff,
capacity failure, inference failure, or leaked-reservation assertion occurred. This is throughput and correctness
evidence, not strength evidence.

## Method

- Source revision: `2949744f7d5f2b22e313fd6673fa22fd83ed3763`
- Hardware: assigned RTX 4070 SUPER device 1 on Vast instance 48008789
- Driver/runtime: NVIDIA 595.71.05, PyTorch 2.12.1+cu126, CUDA 12.6, cuDNN 9.10.2
- Device isolation: a separate architecture benchmark occupied device 0; every graph-search arm explicitly used
  device 1
- Model architecture: current 29-plane chess contract and the production `12 x 112` residual/global-pooling
  configuration from `vast-chess-8gpu-1d-r3.yaml`
- Model weights: deterministic random initialization, seed 20260818, 3,091,097 parameters
- Model SHA-256: `52d735c1e4f6d4c1a651e40240c5453afef8c46b4979fe495c98e53c00a13554`
- Workload: 16 games from the first 16 final FENs in the Stockfish 8moves-v3 suite, maximum inference batch 256
- Measurement: at least 120 seconds per arm, completed only at full-ply boundaries
- Low budgets: 150, 300, and 600 searches with four parallel searches and two warm-up plies
- Large budgets: 30,000 and 60,000 searches with 64 parallel searches and no full-budget warm-up
- Graph correction threshold: 0.01
- Pair order: tree/graph at 150, graph/tree at 300, tree/graph at 600, graph/tree at 30k, tree/graph at 60k

The historical checked-in model was rejected during preflight because it accepts 25 input planes rather than the
current 29. The benchmark model was generated with `tools/prepare_benchmark_model.py` rather than adapting an
incompatible checkpoint.

## Results

| Searches | Parallel | Tree searches/s | Graph searches/s | Graph slowdown | TT hit rate | Continued shared nodes | Evaluations avoided |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 150 | 4 | 21,650.8 | 19,741.7 | 8.82% | 0.000000% | 0.000000% | 0.000000% |
| 300 | 4 | 20,811.0 | 20,328.0 | 2.32% | 0.001356% | 0.000000% | 0.000000% |
| 600 | 4 | 20,234.0 | 19,272.4 | 4.75% | 0.011606% | 0.010551% | 0.000000% |
| 30,000 | 64 | 30,168.6 | 28,037.3 | 7.06% | 2.365374% | 0.466282% | 0.000114% |
| 60,000 | 64 | 27,684.4 | 26,088.8 | 5.76% | 3.462692% | 2.536727% | 0.034818% |

TT hit rate is equality-verified links divided by table probes. Continued shared nodes are transpositions revisited
after both the incoming edge and shared child had completed visits, with their value difference at or below 0.01.
Evaluation avoidance is correction backups divided by correction backups plus performed neural evaluations.

At 30k, graph used 2.31 GiB peak RSS versus tree's 2.27 GiB. At 60k, graph used 3.51 GiB versus tree's 3.46 GiB.
Both large-budget pairs averaged approximately 255.7 positions per 256-position inference batch and used 333 MiB
peak device memory.

## Why early move-order hits remain rare

Exact repetition semantics exclude most board-identical move-order transpositions. Two reversible move orders can
reach the same current board while retaining different intermediate positions. Those positions remain relevant
because a future reversible line can revisit them and change threefold adjudication. The implemented identity
therefore requires the same multiset of exact repetition positions and occurrence counts, not merely the same board
or current repetition count.

Removing history order permits sharing when the same retained positions occurred in another order, and an
irreversible move resets obsolete context. It does not merge ordinary move orders whose intermediate-position sets
differ. A much larger practical hit rate requires deliberately approximate history identity or a design that shares
less than full node statistics across distinct history contexts. Either would change the original exact-state
requirement and must be evaluated as a separate algorithm.

## Threshold interpretation and limitations

`transposition_value_threshold` does not affect matching. After an exact shared node is linked and revisited, it
compares the incoming edge mean with the shared child mean. Difference greater than 0.01 stops descent and backs up
a correction without inference; difference at or below 0.01 continues through the shared subtree.

Random model values cluster differently from a trained model, directly affecting threshold crossings, PUCT paths,
and hit rates. These measurements validate hardware capacity and establish that low-budget reuse can be absent, but
a trained 29-plane checkpoint is required before drawing a production conclusion about correction frequency or
strength. A threshold sweep should also use that trained checkpoint; lowering the threshold on random values would
measure a different search policy without answering the production question.

## Validation and artifacts

The isolated Release extension built for SM 8.9. A separate Release `NativeTests` build passed its unified CTest
entry, including graph transpositions, history identity, cycles, parallel visits, rerooting, and tree-equivalence
tests. The one-root 30k preflight also completed with 64 parallel reservations before the sustained matrix.

`01` through `10` are the sustained paired arms. The two `smoke` JSON files are preflights. `pre-run-gpu-processes`
and `post-run-state` preserve device ownership and environment/process evidence. The generated model is reproducible
from the recorded configuration and seed and is not duplicated in Git.
