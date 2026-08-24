# Node A vs node B self-play search throughput — 8xrtx4070super, 2026-08-24

| | |
| --- | --- |
| `experiment_configuration_sha256` | `f11f7aba4bc9c0be757204084ba4b4f66ed08f3516af7c0c2af4bd35e78c1a97` |
| Source revision | `b333d2ad7ded03ea7d337fa2ee3a955ca7027db8` (`clean`) |
| Node | A: container 48571853, 8x RTX 4070 SUPER, driver 595.71.05, 80 effective CPUs, 251 GiB RAM — B: container 48395477, 8x RTX 4070 SUPER, driver 595.71.05, 80 effective CPUs, 188 GiB RAM. Both dual Intel Xeon E5-2673 v4 @ 2.30 GHz |
| Date | 2026-08-24 |

## Method

Self-play search throughput only; no training, no ingestion, no evaluation. Driver script
`nodecmp.sh` (identical copy on both nodes), which refuses to start if any compute process is
already resident on the GPUs — both nodes were exclusively idle for every repeat.

32 worker processes (8 GPUs x 4 per GPU), 512 parallel games per process, generation 60, fixed
TorchScript model `chess-v2.jit.pt` (sha256
`9c9a24b3dbca892c2c4595d5c23ddc6e25b3af6a41fdeeeb23e43e19b314c158`, byte-identical on both nodes),
2 warm-up batches, barrier-synchronised start, 90 s sample window, 3 repeats per node. Both nodes
ran the same Release build of the same revision, compiled locally on each node.
`vast-chess-4day-production-v2.yaml` is used as a measurement fixture, not as a production recipe.

## Results

| Repeat | Node A searches/s | Node B searches/s |
| --- | --- | --- |
| 1 | 634,335 | 617,471 |
| 2 | 627,843 | 619,087 |
| 3 | 623,715 | 613,975 |
| **Mean** | **628,631** | **616,844** |
| Spread | 1.7% | 0.83% |

| | Node A | Node B |
| --- | --- | --- |
| Mean GPU utilization | 95.6% | 93.2% |
| CPU per worker process | 54% | 63% |
| Average inference batch | 222 | 222 |
| Completed games per repeat | 182-208 | 181-185 |

**Node A is 1.9% faster than node B.** Both nodes carry the same CPU and GPU models; the material
difference between them is RAM (251 vs 188 GiB) and price ($0.72 vs $0.755 per hour).

## Correction to an earlier claim

A comparison recorded earlier on 2026-08-24 reported node A at 662,137 searches/s against node B at
509,511 and concluded "node A ~30% faster, ~35% better per dollar". That node B figure was a single
repeat taken while the node was **not** exclusively idle, and it understated node B by roughly 20%.
The measurement above supersedes it: the real difference between these two nodes is ~2%, not ~30%.
Node A remains the better host on price and a small throughput margin, not on hardware class.

## Open question: revision-over-revision is confounded

These numbers are **not** a verdict on the self-play optimisation work merged at this revision.
Node A measured 662,137 searches/s at `b0719ac7` and 628,631 here, about 5% lower, while the average
inference batch rose from 138 to 222 and GPU utilization rose — but the optimisation work also
changed the fixture configuration itself (`inference_workers: 2 -> 1`, on the grounds that batch
submission is now a CUDA graph launch, so a second worker has nothing to overlap and two capturing
workers in one process fault the device). Old-code-with-two-workers versus new-code-with-one-worker
is not a controlled comparison.

The work was developed and tuned on node B, whose pre-merge baseline is the contaminated one, so
there is currently no trustworthy measurement of what it did to a node that is not CPU-starved.
Deciding this needs a clean `b0719ac7` measurement on both nodes under the same exclusivity
discipline; node B retains a worktree at that revision under `/workspace/wt/base`.

## Raw data

Per-repeat `summary.json`, `utilization.txt`, per-worker JSON, GPU and CPU samples remain under
`/workspace/nodecmp/<tag>/` on each node (tags `nodeA-m1..m3`, `nodeB-m1..m3`). Nothing on a rented
node is durable; copy before release.
