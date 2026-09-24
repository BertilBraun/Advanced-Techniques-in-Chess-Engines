# 5. Systems optimization

The purpose of systems optimization in this project is not to maximize an isolated benchmark. It is to turn a fixed
hardware budget into more useful self-play positions and, ultimately, more strength per wall-clock hour. That
distinction matters because model throughput, search throughput, completed games, admitted replay positions, and
optimizer progress can move by very different amounts.

## Native ownership of the search loop

Python owns experiment configuration, process supervision, replay, training, evaluation, and artifact publication.
C++ owns the latency-sensitive path: game rules, encoded positions, search trees, selection and backup, batched
inference submission, and inference-result processing. Moving this loop into one native runtime removed Python
message serialization and per-position dispatch from the dominant self-play path. The old Python MCTS implementation
is a historical performance floor, not an alternative production search engine.

The native boundary also enables batching across many simultaneous games. Each self-play process advances 512 game
roots, and requests from those roots share a preallocated asynchronous inference pipeline. Trees survive played moves
so previous work can be retained rather than reconstructed. The final topology uses four actor processes per GPU,
one inference worker per process, batches of at most 320 positions, and two outstanding batches. Those settings are a
joint CPU, memory, batching, and latency choice; they should not be interpreted independently.

## Batching and host-side submission

Large neural batches are useful only if the host can continuously assemble, submit, and consume them. The project
therefore optimized the entire submission path rather than the network kernel alone: board encoding, persistent
staging buffers, host-to-device copies, result processing, CUDA event handling, CUDA graph replay, process count,
and the number of inference threads.

In a controlled 32-process self-play workload, the optimized path increased search throughput from 512,679 to
617,782 searches/s, a 20.5% gain, while reducing aggregate actor CPU consumption from 52.6 to 19.8 cores. Average
inference batch size rose from 141 to 222 and the number of model calls fell by 24%. On the same node restricted to
24 CPU cores, throughput increased from 215,265 to 496,036 searches/s because reducing submission overhead allowed
the previously starved GPUs to remain busy. These figures are specific to the tested network and host, but they show
why CPU efficiency can be decisive even in a nominally GPU-bound workload.

More processes are not automatically wasteful. With the TensorRT actor, reducing the topology from four to two
processes per GPU lowered simulations/s by 39%, despite both configurations filling individual inference batches.
Process concurrency was needed to overlap the work around inference and saturate the device. Conversely, excessive
inference threads duplicated CUDA contexts and divided available requests into smaller batches. One inference thread
per actor was the measured operating point after submission became cheap.

Search parallelism is a separate batching lever. Multiple leaves from one root can fill a batch after other games
finish, but virtual reservations make the search less serial and can reduce search quality. Chapter 4 therefore
treats parallel search as an algorithmic quality-throughput trade, not a free systems optimization.

## Inference runtimes and precision

TorchScript provided the first production-quality trimmed policy/WDL artifact and remains the bootstrap and fallback
runtime. It is important to compare against that serving artifact rather than against eager PyTorch. In an eager
diagnostic, `torch.compile` improved batch-64 model throughput by approximately 27–33%, but the corrected comparison
showed that fused TorchScript was faster than the compiled eager path. Training compilation was also unfavorable in
the relevant eight-GPU control: at the selected batch and contention level it reduced throughput by about 18%
relative to eager execution, whereas bfloat16 autocast improved it by 9.2%. `torch.compile` was
therefore not used as the production inference compiler.

TensorRT quantization and engine refitting [8]
provided the stronger serving path. A native FP16 engine reached 75,889 positions/s against 40,716 for the
TorchScript BF16 control in one matched backend benchmark, a 1.86x ratio. Quantization-aware INT8 inference added a
further 1.31x over the TensorRT FP16 engine for the tested quantization-oriented network. Matched production-
topology comparisons measured smaller but still material INT8 gains that depended strongly on model shape: 14.4%
for the smaller network and 39.1% for the medium network.

These gains required the model and runtime to be designed together. Post-training INT8 was fast but changed policy
and value outputs too severely. The retained design uses scaled post-activation residual blocks trained with fake
quantization, quantizes the backbone convolutions, and leaves the start block, policy and value heads, and linear
layers at higher precision. Each published checkpoint is recalibrated, exported through an explicit Q/DQ ONNX
graph, and refitted into shape-specific TensorRT templates. Fidelity is checked on encoded chess positions with
legal-action masking, policy divergence, and WDL error; engine construction success is not treated as semantic
validation. The corresponding architectural failure and retained block are described in Section 4.3.

Model shape remained a systems variable even at similar parameter counts. Width and depth changed kernel efficiency,
TensorRT tactics, and memory behavior discontinuously. Channels-last layout and cuDNN autotuning helped relevant CNN
shapes, but no analytic parameter-count rule predicted the fastest network. Progressive model sizes were therefore
benchmarked at their actual serving batch and precision rather than selected from FLOPs alone.

## Replay materialization and training supply

Search work becomes useful training compute only after games finish and their observations pass replay admission.
Completed games are atomically published, converted into typed columnar rows, and appended to a fixed-capacity
memory-mapped replay. Materialization is partitioned before expensive conversion, bounds inbox scans and staging,
quarantines malformed individual games, and fails loudly if systemic rejection prevents the learner from receiving
data. This makes replay ingestion both a throughput stage and a scientific boundary.

The trainer reads vectorized batches from the memory-mapped store with pinned-memory prefetch and one persistent DDP
rank per GPU. A loader benchmark on 2.5 million rows improved from 3,943 to 32,579 samples/s after compacting 5,000
small producer shards into 25 containers—8.26x faster and 44% above the measured trainer demand. In a later live interval,
materialization appended 8,790 positions/s while accepted positions arrived at 1,365/s, leaving more than sixfold
headroom. Materialization was not the bottleneck in that workload.

A credit ledger links optimizer work to newly admitted positions. Each training quantum becomes eligible only after
enough materialized data has arrived for the configured replay-reuse ratio. This prevents rejected games, ingestion
stalls, or an accidental change in data reuse from appearing as normal optimizer progress. Persistent DDP ranks then
avoid repeated startup and model construction costs. On the measured eight-GPU topology, a global batch of 2,048,
bfloat16 autocast, and concurrent self-play produced 6,252 training samples/s; larger batches improved hardware
throughput further but would also change update count and optimization semantics, so they were not adopted merely
because the GPU benchmark was faster.

## Overlapping self-play and training

The runtime pipelines self-play, materialization, training, checkpoint publication, and evaluation, but it is not a
fully asynchronous learner. Training still occurs in explicit optimizer quanta, and workers switch models only at a
defined publication boundary. During a quantum, a selected subset of actors continues generating games while the
others pause. This preserves a clear data and model identity while using otherwise idle capacity.

The overlap fraction is a measured compromise. With no actors running, the trainer processed 25,275 samples/s.
Keeping 8, 16, or all 32 actors active reduced trainer throughput to 21,492, 17,139, and 9,210 samples/s, while
simultaneously producing approximately 506,000, 606,000, and 742,000 searches/s. When both training time and the
remaining self-play wait were included, the estimated complete-quantum times were 117.1, 112.9, and 111.2 seconds.
The broad result was that moderate overlap improved total cadence, while running every actor nearly doubled the
trainer quantum for only a small additional end-to-end gain. The retained half-active policy sits near that measured
tradeoff rather than maximizing either trainer or actor throughput alone.

The best overlap also changes with search cost. When searches are cheap, training dominates and pausing policy has
little effect. As visit budgets and model size grow, self-play becomes more expensive and additional concurrent
actors become more valuable. A fixed topology should therefore be justified in the regime where it will be used,
not extrapolated from the beginning of training.

## From inference speed to learning speed

The central systems result is that faster simulations do not translate mechanically into training throughput. In a
production-shaped 400-visit benchmark, a generation-34 TensorRT INT8 actor executed 1.86 times as many simulations
as a generation-18 floating TorchScript control with the same architecture and configuration. The weights were not
identical, so this is a backend-and-checkpoint comparison rather than an isolated precision effect. Separately, a
live 400-visit training stage admitted 23.3% more replay positions per second than an earlier, later-stage 600-visit
topology. Those stages differ in visit budget, checkpoint, and actor scheduling; the 23.3% is not the downstream
effect of the 1.86x benchmark. Faster actors can finish shorter games, actor population changes during training,
checkpoint publication consumes time, and replay credit appears only after complete-game materialization. Within
the live stage, the measured accepted-position rate predicted the observed optimizer cadence exactly.

The relevant sequence is visualized in Figure 6.

![Inference and search throughput must pass through games, replay, and optimization before improving playing strength](figures/throughput-to-learning.svg)

Figure 6: The denominator changes at each boundary. Model evaluations and search simulations describe local
service capacity; completed games and admitted rows determine training supply; only the final stage measures the
playing-strength gain per unit wall-clock time.

An optimization is valuable to this project only if its effect survives far enough down that chain. Core inference
and search benchmarks diagnose mechanisms, while the credit ledger and learning curve establish whether the saved
compute became useful training.

Finally, absolute throughput is not portable across machines. CPU quota, GPU power limits, PCIe and NUMA layout,
driver/runtime versions, model shape, batch fill, and concurrent training all affect the operating point. Node
comparisons informed hardware selection, but quantitative claims in this chapter remain attached to the documented
hardware and workload.
