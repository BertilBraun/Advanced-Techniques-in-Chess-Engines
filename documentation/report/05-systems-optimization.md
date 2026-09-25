# 5. From inference speed to learning speed

The model could learn only as fast as self-play supplied new games. Making one neural-network call faster helped,
but a game still had to finish, its positions had to enter replay, and the trainer had to consume them. Figure 6
follows that path from local speed to useful training data.

![Inference and search throughput must pass through games, replay, and optimization before improving playing strength](figures/throughput-to-learning.svg)

Figure 6: Search speed passes through game completion, replay admission, and optimizer work before it can affect
playing strength. Each boundary has its own throughput measure.

## Native ownership of the search loop

Each search leaf needs a board update, legal moves, encoding, and a neural evaluation. Dispatching that sequence
through Python for every position made host work a bottleneck. The production loop therefore keeps game rules,
search trees, selection and backup, and asynchronous inference requests in C++. Python coordinates configuration,
replay, training, publication, and evaluation outside that hot path.

That native loop can advance many games together: each process holds 512 roots whose leaf evaluations share one
preallocated inference pipeline. Trees survive played moves, retaining useful search work. Four actor processes per
GPU, one inference worker per process, batches of at most 320 positions, and two outstanding batches kept the GPUs
busy in the retained topology.

## Batching and host-side submission

Large neural batches help only while the host assembles, submits, and consumes them fast enough. Board encoding,
persistent staging buffers, host-to-device copies, result processing, CUDA events and graph replay, process count,
and inference-thread count were optimized as one submission path.

In a controlled 32-process self-play workload, the optimized path increased search throughput from 512,679 to
617,782 searches/s, a 20.5% gain, while reducing aggregate actor CPU consumption from 52.6 to 19.8 cores. Average
inference batch size rose from 141 to 222 and the number of model calls fell by 24%. On the same node restricted to
24 CPU cores, throughput increased from 215,265 to 496,036 searches/s because reducing submission overhead allowed
the previously starved GPUs to remain busy. The host-side gain was largest when CPU capacity was limited, showing
that GPU search can be starved by CPU work.

Batch fill alone did not guarantee device saturation. With the TensorRT actor, reducing four processes per GPU to
two lowered simulations/s by 39% even though both filled individual batches: concurrency overlapped CPU and GPU
work around inference. Additional inference threads had the opposite effect, duplicating CUDA contexts and splitting
requests into smaller batches. One inference thread per actor was the measured operating point.

Search parallelism is a separate batching lever. Multiple leaves from one root can fill a batch after other games
finish, but virtual reservations make the search less serial and can reduce search quality. Chapter 4 therefore
treats parallel search as an algorithmic quality-throughput trade, not a free systems optimization.

## Inference runtimes and precision

Once the host could submit full batches, the inference engine became the next lever. A native TensorRT FP16 engine
reached 75,889 positions/s against 40,716 for a TorchScript BF16 control in a matched benchmark, a 1.86x gain.
Quantization-aware INT8 added a further 1.31x over TensorRT FP16 on the tested quantization-oriented network.
Production-topology tests found INT8 gains of 14.4% for the smaller network and 39.1% for the medium network.

Speed was useful only if the deployed network still played the same chess. Post-training INT8 changed policy and
value outputs too much, so the retained scaled post-activation blocks were trained with fake quantization. Backbone
convolutions run in INT8; the start block, heads, and linear layers remain at higher precision. Publication
recalibrates the checkpoint, exports an explicit Q/DQ ONNX graph, and refits a TensorRT template [8]. Policy and
value outputs are then checked on encoded positions. Section 4.3 explains the block design; Chapter 6 shows why
successful engine construction alone could not establish fidelity.

`torch.compile` was another attempted speedup. It accelerated eager batch-64 inference by roughly 27--33%, but
fused TorchScript remained faster. In the tested eight-GPU training workload, compilation reduced throughput by
about 18% relative to eager execution, while bfloat16 autocast improved it by 9.2%. Compilation was not retained
for production inference or training on this workload.

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

Self-play, materialization, training, publication, and evaluation overlap without making the learner fully
asynchronous. Optimizer work remains divided into explicit quanta, and actors switch models only at publication
boundaries. A selected subset keeps producing games during training while the rest pause. This uses idle capacity
without losing the identity of the model and data behind each quantum.

The overlap fraction is a measured compromise. With no actors running, the trainer processed 25,275 samples/s.
Keeping 8, 16, or all 32 actors active reduced trainer throughput to 21,492, 17,139, and 9,210 samples/s, while
simultaneously producing approximately 506,000, 606,000, and 742,000 searches/s. When both training time and the
remaining self-play wait were included, the estimated complete-quantum times were 117.1, 112.9, and 111.2 seconds.
The broad result was that moderate overlap improved total cadence, while running every actor nearly doubled the
trainer quantum for only a small additional end-to-end gain. The retained half-active policy sits near that measured
tradeoff rather than maximizing either trainer or actor throughput alone.

The useful overlap fraction depends on search cost. As visit budgets and model size rise, self-play becomes more
expensive relative to training; actor settings measured at the beginning cannot simply be extrapolated to later
stages.

## Where throughput becomes useful

Faster simulations did not translate mechanically into training throughput. A production-shaped TensorRT INT8
benchmark ran search 1.86 times faster than its floating TorchScript control. In a separate live stage, admitted
replay positions arrived 23.3% faster at 400 visits than at 600. The two comparisons changed different things, so
their speedups cannot be multiplied. Game length, publication pauses, and complete-game replay admission all stand
between simulation rate and optimizer cadence. In the live stage, admitted-position rate predicted that cadence.

The relevant systems measure is therefore how quickly the loop produces completed games, admitted replay, and
optimizer progress—not simulation rate alone. Appendix C records the local comparison boundaries.
