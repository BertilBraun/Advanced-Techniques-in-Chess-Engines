# Go 7x7 two-GPU training baseline

Status: four-hour baseline and screening runs are active on the rented four-RTX-3060 node.

The complete revised configuration is
[`py/configs/baselines/vast-go-7x7-2gpu-4h.yaml`](../../py/configs/baselines/vast-go-7x7-2gpu-4h.yaml). It defines the
four-hour Go 7x7 learning baseline plus five minutes for the final scheduled evaluation to launch. Subsequent
self-play optimizations should retain every other setting unless
the comparison explicitly declares another variable.

## Resource slot

| Resource | Baseline request |
| --- | ---: |
| Visible GPUs | 2 x RTX 3060 12 GiB |
| CPU affinity | 42 logical CPUs |
| Aggregate RAM budget | 40 GiB |
| Minimum disk | 20 GiB |
| Wall time | 14,700 seconds |
| Recorded node price | $0.303/hour |

The shared host exposes 256 logical CPUs and 251 GiB RAM, but the rental owns only 85.33 effective CPUs and 86 GiB.
This experiment deliberately takes approximately half of that allocation. `CUDA_VISIBLE_DEVICES=0,1` and inherited
CPU affinity can enforce the GPU and CPU parts. The current Vast container exposes a read-only cgroup v1 hierarchy,
so it cannot enforce an aggregate 40-GiB process-tree limit. The YAML records the budget, while the launch must
monitor process-tree RSS and abort at 40 GiB unless the node is replaced with a delegated cgroup-v2 environment.
`maximum_host_ram_percent` remains a last-resort host-pressure guard; it is not the run's RAM limiter.

The $0.303 hourly price is conservatively the complete rented-instance price. The four-hour-five-minute wall-time limit is the
sole configured run-duration stop; no separate cost limit is set.

## Model and DDP training

The model is the established Go screening network: 6 residual blocks, 64 hidden channels, squeeze-excitation every
second block, a two-channel policy head, and a one-channel/32-unit value head.

Two persistent NCCL ranks share the same two GPUs as self-play:

- DDP devices are logical CUDA devices 0 and 1;
- local batch size is 512 rows per rank;
- global batch size is 1,024 rows per optimizer step;
- each rank may use eight intra-op CPU threads and one inter-op thread;
- AdamW uses the existing constant learning rate 0.002 and gradient norm cap 1.0. The earlier calibration ran close
  to the 0.5 cap, so the screening baseline raises it to avoid clipping ordinary updates too aggressively.

Batch 512 is a conservative first training baseline inherited from the established per-GPU batch target. It has not
yet been optimized for the much smaller Go model. Training-batch throughput, GPU power, memory, and step latency are
baseline outputs to measure, not assumptions to tune before this run.

The live baseline exposed a phase-specific rank imbalance. Across five-second samples, self-play/waiting averaged
91.3% utilization on GPU 0 and 93.6% on GPU 1, while inferred training windows averaged 88.9% and 70.6%. Before the
first evaluation boundary, training alone averaged 87.9% and 62.7%, so external evaluation does not explain the
difference. Worker pausing is symmetric: workers 1–3 and 5–7 pause, leaving workers 0 and 4 on the two devices.

The trainer performs no per-batch host scalar extraction, explicit CUDA synchronization, statistics aggregation, or
checkpoint write. Necessary DDP gradient reduction occurs during every backward pass. The current DDP wrapper also
uses the default per-forward BatchNorm-buffer broadcast even though only rank zero's checkpoint is retained. More
materially, each rank constructs, augments, densifies, and pins its next replay batch synchronously in the optimizer
loop; the documented CPU/GPU prefetch overlap is not implemented. A rank that prepares its batch first can therefore
wait at DDP synchronization while the other device catches up. Disable unnecessary buffer broadcasts and implement
bounded next-batch prefetch before interpreting the imbalance as insufficient GPU work. Testing two surviving
self-play workers per GPU is a separate topology experiment because it increases CPU contention as well as filling
GPU gaps.

Policy and value losses both have weight 1.0. The value target uses the final game outcome only
(`root_value_blend: 0.0`), and no auxiliary target is enabled. Run seed 1207 is fixed for self-play, replay sampling,
dataset preparation, openings, and paired-match assignment so later comparisons can reuse the same stochastic input.

## Optimized self-play input

The configuration applies the isolated Go optimum measured on the same RTX 3060 hardware:

- four self-play processes per GPU, eight processes total;
- 1,024 independent games per process, 8,192 active games total;
- one sequential search per root, with no parallel-leaf approximation;
- one inference worker, two outstanding batches, and a batch limit of 128 per process;
- 256 visits for every move because `full_search_probability` is 1.0;
- workers 1, 2, 3, 5, 6, and 7 pause during a training quantum, leaving one self-play process running on each GPU.

The configured 32-visit fast-search budget is therefore inactive in this baseline; it remains explicit so a later
experiment that changes full-search probability has a defined fast path.

The first live calibration used 128 visits and a 512-position inference limit. It exposed bursty inference, paused
all eight self-play workers during every training quantum despite the authored empty pause list, and therefore does
not define the accepted baseline. The revised run measures whether smaller, more frequent inference calls and two
continuing workers improve integrated GPU duty cycle while 256 visits produce stronger search targets. Its measured
sustained self-play rate becomes the optimization baseline; the earlier isolated projection is not comparable to
this revised search budget.

Self-play retains the existing AlphaZero policy: 0.25 Dirichlet mixing with alpha 0.3, exploration constant 1.5,
temperature annealing from 1.0 to 0.1, greedy play after ply 40, no random opening plies, and one primary sample per
eligible position. Go uses 7.5 komi, area scoring through the existing game contract, and a 196-ply safety limit.

## Credit calibration evidence

The 128-visit calibration completed generation 49 in 46 minutes 42 seconds after checkpoint zero. Excluding the
long first-generation startup, generations 1 through 49 averaged 55.5 seconds. Its final ledger recorded 1,276,728
materialized positions, 5,017,600 consumed presentations, and 89,312 available presentations. The cumulative
observed replay ratio was therefore 3.930 against the configured 4.0, and only 0.872 of the next 102,400-presentation
quantum was available. This run was credit-bound rather than accumulating a useful producer surplus.

Training 100 steps took approximately 32 seconds at the measured 3,100 to 3,400 samples/s. The revised 200-step
quantum consumes 204,800 presentations and requires 51,200 newly materialized positions at replay ratio 4. This
keeps publication frequent enough for stronger self-play models to enter the data stream early. The production
baseline uses 256 visits and permits one worker per GPU to continue during training. The intended range is roughly
120 generations in four hours, but the first complete baseline must measure whether credit waits or training time make
that estimate optimistic.

The 256-visit calibration ended during its initial game-completion wave and did not reach generation one, so it
does not establish a steady production rate. The baseline deliberately pauses six of eight workers during training
and leaves one producer per GPU. The new `credit/available_quantum_fraction`, credit balance, observed replay ratio,
and credit-wait metrics determine whether those two producers keep up or whether the topology needs adjustment.

## Replay and optimizer cadence

Replay capacity grows linearly from 100,000 rows at generation 0 to 500,000 at generation 15. At the baseline's
51,200 new positions per generation, the final buffer represents approximately 9.8 generations of production, so
early-model positions are continuously evicted instead of surviving for the complete run. Fifty sparse policy
entries retain the complete 7x7 action space, including pass. The static maximum is 500,000 rows.

Each training generation contains 200 optimizer steps. At global batch 1,024 that consumes 204,800 presentations.
The replay ratio of 4 requires 51,200 new unique samples before the next quantum. The 1,000,000-step ceiling is only
a safety bound; the two-hour wall-time limit is expected to stop the run first.

Full resumable checkpoints are retained each generation. Inference artifacts retain the latest 11 generations and
every tenth generation, while evaluation additionally protects referenced historical checkpoints.

## Evaluation baseline

Evaluation remains on the accepted 20-minute elapsed cadence and logs at the scheduled boundary. Jobs time out after
20 minutes and cycle independently across devices 0 and 1. The production dataset and opening suite are immutable,
checked-in baseline inputs. Run preparation validates and reuses them rather than invoking KataGo generation:

- engine-labelled dataset: `py/reference/go-7x7-baseline-v1.bin`;
- four-ply, 50-line opening suite: `py/reference/go-7x7-baseline-openings-v1.json`.

The dataset contains 486 positions retained from 30 complete games. Its SHA-256 is
`c63c3e8894ea01d016e5c3bc5a0bc90a1214b98eb2eb3cac7301a7270d8ba5b0`; the manifest SHA-256 is
`c0aa1b1da69186e6062f64c0f7a1e4ef37b4b573f270084056228e4f627dbd99`; and the opening-suite SHA-256 is
`57283f2e27d27af6c8e3403c1979dd5431948cd8132ccb01177d727e6c00f80e`.

KataGo labels the dataset at 256 visits. Separate match definitions use 16, 64, and 256 visits. Every paired match
definition uses all 50 openings with the candidate playing each side, hence 100 games per definition. The fixed
ladder contains:

- fixed-dataset policy accuracy and cross-entropy;
- checkpoints selected at the preceding 20-, 40-, and 60-minute boundaries plus alternating older offsets;
- the same-time checkpoint from the completed baseline run;
- KataGo at 16, 64, and 128 visits.

## Overnight screening matrix

The baseline and nine one-variable challengers live under
[`py/configs/screening/go-7x7-overnight`](../../py/configs/screening/go-7x7-overnight). `00-baseline.yaml` inherits
the complete baseline and changes only run identity and output path. Every later YAML inherits `00-baseline.yaml`
and declares only its experimental difference: learning-rate decay through generation 120, constant learning rate
0.004, 75% fast search, a 64-to-512 search-budget progression through generation 120, root-value blending, replay
ratio 8, tree retention, uniformly sampled random openings of zero through twelve plies, and finally the next-policy
auxiliary target.

The replay-ratio-8 experiment uses a 250,000-row maximum rather than 500,000 because it produces only 25,600 new
positions per generation; this preserves approximately the baseline's ten-generation replay-age window.
Changing the baseline automatically changes every inherited run and invalidates their existing approvals and queue
summary fingerprint.

Up to ten due evaluation jobs run concurrently, with excess work queued by the evaluation manager. This can create
substantial GPU contention at later boundaries. That contention is intentionally part of the end-to-end baseline and
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

The coordinator records configured and observed replay ratio, cumulative materialized positions and consumed
presentations, available and required presentation credits, available-quantum fraction, replay rows/capacity/fill,
credit-wait and training-quantum durations, optimizer steps, learning rate, search budgets, full-search probability,
root-value blend, and per-evaluation duration in TensorBoard. The generation-completion console line includes the
credit backlog, observed ratio, wait time, and replay fill state.

The live run uses the sampled 40-GiB process-tree monitor described above.
