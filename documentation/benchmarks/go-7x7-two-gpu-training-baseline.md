# Proposed Go 7x7 two-GPU training baseline

Status: authored and validated for review; not approved or launched.

The complete configuration is
[`py/configs/baselines/vast-go-7x7-2gpu-2h.yaml`](../../py/configs/baselines/vast-go-7x7-2gpu-2h.yaml). It defines the
first two-hour Go 7x7 learning baseline. Subsequent self-play optimizations should retain every other setting unless
the comparison explicitly declares another variable.

## Resource slot

| Resource | Baseline request |
| --- | ---: |
| Visible GPUs | 2 x RTX 3060 12 GiB |
| CPU affinity | 42 logical CPUs |
| Aggregate RAM budget | 40 GiB |
| Minimum disk | 20 GiB |
| Wall time | 7,200 seconds |
| Recorded node price / maximum cost | $0.303/hour / $0.65 |

The shared host exposes 256 logical CPUs and 251 GiB RAM, but the rental owns only 85.33 effective CPUs and 86 GiB.
This experiment deliberately takes approximately half of that allocation. `CUDA_VISIBLE_DEVICES=0,1` and inherited
CPU affinity can enforce the GPU and CPU parts. The current Vast container exposes a read-only cgroup v1 hierarchy,
so it cannot enforce an aggregate 40-GiB process-tree limit. The YAML records the budget, while the launch must
monitor process-tree RSS and abort at 40 GiB unless the node is replaced with a delegated cgroup-v2 environment.
`maximum_host_ram_percent` remains a last-resort host-pressure guard; it is not the run's RAM limiter.

The $0.303 hourly price is conservatively the complete rented-instance price. Two hours cost at most $0.606 before
rounding, so the configuration stops at a $0.65 cap as well as the wall-time limit.

## Model and DDP training

The model is the established Go screening network: 6 residual blocks, 64 hidden channels, squeeze-excitation every
second block, a two-channel policy head, and a one-channel/32-unit value head.

Two persistent NCCL ranks share the same two GPUs as self-play:

- DDP devices are logical CUDA devices 0 and 1;
- local batch size is 512 rows per rank;
- global batch size is 1,024 rows per optimizer step;
- each rank may use eight intra-op CPU threads and one inter-op thread;
- AdamW uses the existing constant learning rate 0.002 and gradient norm cap 0.5.

Batch 512 is a conservative first training baseline inherited from the established per-GPU batch target. It has not
yet been optimized for the much smaller Go model. Training-batch throughput, GPU power, memory, and step latency are
baseline outputs to measure, not assumptions to tune before this run.

Policy and value losses both have weight 1.0. The value target uses the final game outcome only
(`root_value_blend: 0.0`), and no auxiliary target is enabled. Run seed 1207 is fixed for self-play, replay sampling,
dataset preparation, openings, and paired-match assignment so later comparisons can reuse the same stochastic input.

## Optimized self-play input

The configuration applies the isolated Go optimum measured on the same RTX 3060 hardware:

- four self-play processes per GPU, eight processes total;
- 1,024 independent games per process, 8,192 active games total;
- one sequential search per root, with no parallel-leaf approximation;
- one inference worker, two outstanding batches, and a batch limit of 512 per process;
- 128 visits for every move because `full_search_probability` is 1.0;
- no self-play process is paused during a training quantum.

The configured 32-visit fast-search budget is therefore inactive in this baseline; it remains explicit so a later
experiment that changes full-search probability has a defined fast path.

The four-GPU isolated measurement sustained 446,725 searches/s, so linear scaling suggests roughly 223,000
searches/s on two GPUs before trainer and evaluation contention. The integrated run is authoritative: its sustained
self-play rate, not the isolated projection, becomes the optimization baseline.

Self-play retains the existing AlphaZero policy: 0.25 Dirichlet mixing with alpha 0.3, exploration constant 1.5,
temperature annealing from 1.0 to 0.1, greedy play after ply 40, no random opening plies, and one primary sample per
eligible position. Go uses 7.5 komi, area scoring through the existing game contract, and a 196-ply safety limit.

## Replay and optimizer cadence

Replay capacity grows linearly from 100,000 rows at generation 0 to 2.5 million at generation 200. Fifty sparse
policy entries retain the complete 7x7 action space, including pass. The static maximum remains 2.5 million.

Each training generation contains 100 optimizer steps. At global batch 1,024 that consumes 102,400 presentations.
The replay ratio of 8 requires 12,800 new unique samples before the next quantum. The 100,000-step ceiling is only a
safety bound; the two-hour wall-time limit is expected to stop the run first.

Full resumable checkpoints are retained each generation. Inference artifacts retain the latest 11 generations and
every tenth generation, while evaluation additionally protects referenced historical checkpoints.

## Evaluation baseline

Evaluation remains on the accepted 20-minute elapsed cadence and logs at the scheduled boundary. Jobs time out after
20 minutes and cycle independently across devices 0 and 1. Dataset and opening artifacts use new immutable baseline
paths and are generated only if absent:

- engine-labelled dataset: `py/reference/go-7x7-baseline-v1.bin`;
- four-ply, 50-line opening suite: `py/reference/go-7x7-baseline-openings-v1.json`.

KataGo labels the dataset at 256 visits and plays matches at 64 visits. Every paired match definition uses all 50
openings with the candidate playing each side, hence 100 games per definition. The fixed ladder contains:

- fixed-dataset policy accuracy and cross-entropy;
- search versus random and policy-only versus random;
- checkpoints selected at the preceding 20-, 40-, and 60-minute boundaries;
- available retained generations 10, 20, ..., 100;
- KataGo.

All due definitions run concurrently, as required by the evaluation architecture. This can create substantial GPU
contention once all fixed generations exist. That contention is intentionally part of the end-to-end baseline and
must remain identical in later comparisons. Evaluation duration, timeout/failure status, and the self-play-rate dip
at each boundary must be included in the baseline report.

## Measurements required from the run

The baseline is complete only when its artifacts record:

- learning curves against elapsed time for policy, value, and every evaluation definition;
- sustained and boundary-local self-play searches/s, completed games/s, and inference batch distribution;
- optimizer steps/s, batch preparation time, DDP synchronization time, and model-publication latency;
- per-device utilization, power, memory, and contention during self-play, training, and evaluation;
- process-tree CPU and RSS against the 42-CPU/40-GiB slot;
- replay size, credit balance, replay age, ingestion rate, and generation cadence;
- every evaluation completion, failure, and timeout.

Do not launch this run until the configuration and the 40-GiB monitoring approach receive explicit approval.
