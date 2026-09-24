# 5. From inference speed to learning speed

Search-based learning needs enough completed games to supply the trainer. Under a fixed hardware budget, the useful
systems objective is therefore admitted self-play positions and strength per wall-clock hour, not model evaluations
per second in isolation. The gap between those quantities determined where the engineering effort mattered.

![Inference and search throughput must pass through games, replay, and optimization before improving playing strength](figures/throughput-to-learning.svg)

Figure 6: The denominator changes at each boundary. Model evaluations and search simulations describe local
service capacity; completed games and admitted rows determine training supply; only the final stage measures the
playing-strength gain per unit wall-clock time.

## Native ownership of the search loop

Python coordinates configuration, processes, replay, training, evaluation, and publication. C++ owns game rules,
encoding, search trees, selection and backup, and asynchronous inference requests and results. This division keeps
Python message serialization and per-position dispatch off the dominant self-play path. The earlier Python search
loop is a performance baseline, not an alternative production engine.

The native boundary also enables batching across many simultaneous games. Each self-play process advances 512 game
roots, and requests from those roots share a preallocated asynchronous inference pipeline. Trees survive played moves
so previous work can be retained rather than reconstructed. The final topology uses four actor processes per GPU,
one inference worker per process, batches of at most 320 positions, and two outstanding batches. Those settings are a
joint CPU, memory, batching, and latency choice; they should not be interpreted independently.

## Batching and host-side submission

Large neural batches help only while the host assembles, submits, and consumes them fast enough. Board encoding,
persistent staging buffers, host-to-device copies, result processing, CUDA events and graph replay, process count,
and inference-thread count were optimized as one submission path.

In a controlled 32-process self-play workload, the optimized path increased search throughput from 512,679 to
617,782 searches/s, a 20.5% gain, while reducing aggregate actor CPU consumption from 52.6 to 19.8 cores. Average
inference batch size rose from 141 to 222 and the number of model calls fell by 24%. On the same node restricted to
24 CPU cores, throughput increased from 215,265 to 496,036 searches/s because reducing submission overhead allowed
the previously starved GPUs to remain busy. These figures are specific to the tested network and host, but they show
why CPU efficiency can be decisive even in a nominally GPU-bound workload.

Batch fill alone did not guarantee device saturation. With the TensorRT actor, reducing four processes per GPU to
two lowered simulations/s by 39% even though both filled individual batches: concurrency overlapped CPU and GPU
work around inference. Additional inference threads had the opposite effect, duplicating CUDA contexts and splitting
requests into smaller batches. One inference thread per actor was the measured operating point.

Search parallelism is a separate batching lever. Multiple leaves from one root can fill a batch after other games
finish, but virtual reservations make the search less serial and can reduce search quality. Chapter 4 therefore
treats parallel search as an algorithmic quality-throughput trade, not a free systems optimization.

## Inference runtimes and precision

TorchScript supplied the trimmed policy/WDL bootstrap and fallback artifact, making it the relevant comparator to a
new serving runtime. `torch.compile` sped up eager batch-64 inference by roughly 27--33%, but fused TorchScript was
faster than the compiled eager path. In the tested eight-GPU training workload, compilation reduced throughput by
about 18% relative to eager execution; bfloat16 autocast improved it by 9.2%. Compilation was not retained for
production inference or training on this workload.

TensorRT quantization and engine refitting [8] provided the stronger serving path. A native FP16 engine reached
75,889 positions/s against 40,716 for the
TorchScript BF16 control in one matched backend benchmark, a 1.86x ratio. Quantization-aware INT8 inference added a
further 1.31x over the TensorRT FP16 engine for the tested quantization-oriented network. Matched production-
topology comparisons measured smaller but still material INT8 gains that depended strongly on model shape: 14.4%
for the smaller network and 39.1% for the medium network.

Those gains depended on model design: post-training INT8 was fast but changed policy and value outputs too much.
The retained scaled post-activation blocks are trained with fake quantization. Backbone convolutions run in INT8;
the start block, heads, and linear layers remain at higher precision. Published checkpoints are recalibrated,
exported as explicit Q/DQ ONNX graphs, and refitted into shape-specific TensorRT templates. Legal-move policy and
WDL fidelity are checked on encoded positions, because a built engine is not necessarily a faithful one. Section
4.3 describes the block design; Chapter 6 describes a refit failure that made semantic checks necessary.

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

Faster simulations did not translate mechanically into training throughput. In a production-shaped 400-visit
benchmark, a TensorRT INT8 actor executed 1.86 times as many simulations as a floating TorchScript control with
the same architecture and configuration. Their weights differed, so this is a backend-and-checkpoint comparison,
not an isolated precision effect. A separate live 400-visit stage admitted 23.3% more replay positions per second
than a 600-visit stage. Visit budget, checkpoint, and actor scheduling differed between those stages; the 23.3%
is not a measured downstream effect of the 1.86x benchmark. Faster actors may finish shorter games, checkpoint
publication interrupts work, and replay credit appears only after complete-game materialization. In the live
stage, admitted-position rate predicted optimizer cadence exactly.

Inference and search benchmarks diagnose a mechanism. Completed games, accepted replay, optimizer cadence, and
learning curves show whether that mechanism saved useful training time.

Finally, absolute throughput is not portable across machines. CPU quota, GPU power limits, PCIe and NUMA layout,
driver/runtime versions, model shape, batch fill, and concurrent training all affect the operating point. Node
comparisons informed hardware selection, but quantitative claims in this chapter remain attached to the documented
hardware and workload.
