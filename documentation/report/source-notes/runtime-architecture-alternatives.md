# Runtime architecture alternatives source dossier

## Purpose and evidence discipline

This dossier preserves substantial systems investigations that explain the current architecture without mixing
superseded components into the current-system figure. It is organized by design question, not by development
chronology. The report should describe each alternative, the pressure it addressed, what the measurement actually
established, and why the retained boundary is different.

The publication-quality system SVG remains a diagram of the implementation that exists now. Its checked node and
edge inventory is in [`system-architecture-and-figure-dossier.md`](system-architecture-and-figure-dossier.md).
Historical alternatives should appear in report prose, comparison tables, or a separate small design-space figure;
they must not be ghosted into the current runtime as though they still execute.

Evidence strength varies:

- current source and the final chess configuration are authoritative for retained ownership and topology;
- benchmark dossiers with commands, manifests, hardware, and raw artifacts support quantitative claims within their
  stated workload;
- older optimization notes are useful records of considered designs, but their headline timings lack enough
  provenance for a publication-quality cross-architecture speedup claim;
- a synthetic microbenchmark can identify a mechanism, but cannot establish end-to-end production throughput;
- measurements from different hardware, models, rules implementations, search budgets, or concurrency are not a
  controlled comparison even when their scale is informative.

## The design questions

The runtime investigations answer four different questions that should not be collapsed into one optimization
story:

1. **Where should asynchronous boundaries live?** Pipes, shared queues, direct clients, fine-grained coroutines, and
   same-process bounded slots impose different scheduling and serialization costs.
2. **Which component owns mutable search state?** Separating tree ownership from inference execution avoids shared
   tree locks and makes completion ordering explicit.
3. **How much concurrency should be exposed at each layer?** Processes per GPU, active games per process, search
   threads, inference workers, outstanding submissions, and batch caps jointly determine utilization and latency.
4. **Which metric matches the service regime?** Aggregate positions per second under thousands of active games is
   not single-move latency for one interactive tree.

## Python process-boundary alternatives

The early architecture study in
[`documentation/history/optimizations/architecture.md`](../../history/optimizations/architecture.md) records three
Python multiprocessing arrangements. Its diagrams remain useful for naming the boundaries:
[`pipe.dot`](../../history/optimizations/architecture/pipe.dot),
[`queue.dot`](../../history/optimizations/architecture/queue.dot), and
[`client-server.dot`](../../history/optimizations/architecture/client-server.dot).

| Alternative | Request and result path | Intended benefit | Limiting mechanism | Disposition |
| --- | --- | --- | --- | --- |
| Central pipe load balancer | Self-play processes send positions through pipes to one load balancer, which dispatches to dedicated inference servers and routes results back | Centralized batching and device control | One Python process owns routing, serialization, polling, and response dispatch; it becomes a contention and IPC concentration point | Rejected as the scalable hot-path boundary |
| Shared queue and cache manager | Producers submit to a shared inference queue; inference servers return through a response queue; a cache manager mediates lookup and completion | Decoupled producers and consumers, shared batching, possible reuse | Queue synchronization, object transfer, response matching, and a central cache remain in the per-leaf path; cache value did not justify the boundary under production-like search | Rejected as the production search path |
| Per-process inference client | Each self-play process communicates more directly with inference/model resources and owns more local state | Removes the central load balancer and reduces cross-process coordination | Replicated clients/models consume more memory; Python and operating-system scheduling still sit in the hot path | Useful evidence for decentralization, then superseded by native in-process search and inference coordination |
| Native bounded-slot pipeline | One native search instance owns its trees and submits encoded leaves through reusable slots to dedicated native inference workers; completions return to the same owner | No Python IPC per leaf, bounded memory, reusable buffers, explicit backpressure, single tree writer | Requires a specialized search/inference contract rather than a general request service | Retained current boundary |

The historical note reports approximately 431 seconds, 224 seconds, and 40 seconds per 1,000 samples for its pipe,
queue, and client arrangements. These values are an **archival screening result**, not a publishable speedup ratio:
the note does not lock the current model, hardware, search workload, revision, warmup, or raw timing artifacts. The
defensible conclusion is directional: central Python routing and queueing were costly enough to motivate removing
interprocess communication from the leaf-inference loop. The current architecture should not be claimed to be a
specific multiple faster on the strength of that table.

### Why the cache-manager boundary did not rescue the queue design

Caching and transport were coupled in the queue design, but they are separate decisions. A shared cache can only
repay central lookup, synchronization, storage, and response-routing costs when transpositions produce enough useful
hits under the real search distribution. The project later audited inference caching under production-like search
and rejected it because useful hit rates were too low to amortize the machinery. Consequently, the current figure
must not contain a cache manager, and the report should not imply that queue transport failed only because of an
unoptimized implementation. The broader lesson is that reuse must be measured at the search boundary where it will
run, rather than inferred from the existence of transpositions.

## Fine-grained `asyncio` inside search

[`documentation/history/pre-cpp-port/asyncio/README.md`](../../history/pre-cpp-port/asyncio/README.md) records an
attempt to express parallel games and recursive tree search as coroutines. The appeal was legitimate: coroutines
made suspension at inference leaves explicit and produced readable control flow without a process per game.

The granularity was wrong for Python's coroutine scheduler. In the archived synthetic workload, synchronous
recursive computation took about 0.59 seconds, whereas the asynchronous form spent about 0.36 seconds doing the
traced work but about 10.63 seconds wall-clock overall. Scheduling and coroutine overhead therefore dominated the
fine-grained recursive operations. This is mechanism evidence, not a production throughput benchmark: the workload
was synthetic and should not be compared numerically with later native self-play measurements.

The resulting decision is narrower than “asynchrony is bad.” Fine-grained Python coroutine suspension was removed
from tree traversal and backup. The current system still uses asynchronous execution at coarser, bounded ownership
boundaries: native inference slots, independent self-play processes, replay materializers, evaluation jobs, and the
trainer process group. The report should say **move asynchrony outwards**, not claim that the runtime became wholly
synchronous.

## Migration of search ownership into the native engine

### What moved

The architectural migration placed the latency-critical and highly repeated operations in one native ownership
domain:

- legal move generation and state transitions use the Stockfish-backed chess rules implementation;
- each native search instance owns active game states, trees, selection, expansion, virtual loss, backup, and output
  construction;
- native inference workers receive encoded leaves through persistent buffers and never mutate a tree;
- the owning search scheduler applies completions, so tree mutation has one writer;
- the same native search implementation serves self-play, evaluation, and interactive analysis, with typed mode
  configuration rather than duplicate algorithms.

Python retains the boundaries for which its orchestration and data tooling are valuable: typed configuration,
process and GPU placement, lifecycle control, trajectory handoff, replay materialization, training, checkpoint
publication, evaluation scheduling, and reporting. The authoritative present boundary is summarized by
[`cpp/README.md`](../../../cpp/README.md) and implemented through the entry points inventoried in the
[current-system dossier](system-architecture-and-figure-dossier.md).

### Why ownership, not only language, mattered

The migration should not be described merely as “rewrite Python in C++.” It removed specific runtime crossings and
made ownership explicit:

1. A selected leaf stays within the process that owns its tree.
2. Encoding writes into a reusable slot rather than constructing a general per-request Python object.
3. A bounded number of outstanding submissions supplies backpressure.
4. Inference workers operate on model buffers and return results; they do not lock or update the search tree.
5. The tree owner applies results and advances games in a deterministic ownership domain.

That structure removes per-leaf Python IPC, promise/future allocation, general request dispatch, and shared-tree
locking simultaneously. It also means the performance gain cannot honestly be attributed to “C++” alone without a
controlled factorial study.

### Superseded ownership designs and caveats

The pre-port account in
[`documentation/history/pre-cpp-port/chess-port.md`](../../history/pre-cpp-port/chess-port.md) reports very large
speedups for an early custom C++ chess implementation. That implementation later failed a perft correctness audit
and was replaced by Stockfish-backed rules. Its magnitude is historical motivation, not evidence for the correctness
or precise speed of the current engine.

The platform once contained separate facades and owners for self-play, evaluation, and interactive search, plus both
Python and C++ inference-client abstractions. The closed design ledger
[`documentation/architecture/platform-rework.md`](../../architecture/platform-rework.md) records their removal and
the move to one authoritative native search surface. It is appropriate evidence for the design decision, but current
source remains the authority for live edges.

For scale context only, the deliberately naive batch-one Python reference in
[`naive-python-mcts-rtx3060-20260816`](../../benchmarks/naive-python-mcts-rtx3060-20260816/README.md) measured about
80.8 simulations per second during a contended live-training observation. Native integrated measurements operate on
batched populations of games and reach a different scale, but the workload differences prevent clean attribution.
The report may say that batching plus native ownership changed the feasible operating scale by orders of magnitude;
it should not present the ratio as a controlled language benchmark.

## General inference service versus direct search slots

The general inference client supported flexible request objects, promises or futures, shared queues, batching
timeouts, tensor stacking, policy filtering, and result dispatch. This is a reasonable service abstraction when
callers and result shapes are heterogeneous. It was expensive when every search leaf followed exactly the same path
and search threads blocked on each request.

The direct-evaluation harness
[`documentation/benchmarks/harnesses/direct-evaluation-inference.md`](../../benchmarks/harnesses/direct-evaluation-inference.md)
isolated much of this cost. At batch 50 it measured roughly 14,584 positions/s for direct persistent buffers and
14,716 positions/s for an SPSC pipeline, compared with 1,889 for the cached general client and 3,635 for the
noncached client. Within that harness, three replicas at batch 50 achieved about 31,404 positions/s versus 23,313
for one combined batch of 150. Those results support two decisions:

- reusable typed slots can remove substantial service-framework overhead when the producer and consumer contract is
  fixed;
- one larger batch is not automatically better than several independent execution resources, because host launch,
  device concurrency, and queuing all contribute.

The dedicated self-play study
[`self-play-direct-inference-4x4070-super-20260722`](../../benchmarks/self-play-direct-inference-4x4070-super-20260722/README.md)
then exercised the direct design end to end: one tree owner, reusable pinned integer feature slots, persistent model
replicas and streams, and asynchronous completions that never mutate trees. Its selected historical topology improved
search throughput by about 63% over the old general-client baseline while reducing CPU and resident memory. The
exact worker count chosen there was later superseded; the durable result is the ownership and buffer contract.

## Process, actor, and inference-worker topology alternatives

### The coupled tuning space

“Number of workers” is not one parameter. The investigated axes were:

- self-play processes per GPU;
- active games owned by each process;
- search or tree-owner parallelism within a process;
- native inference workers and therefore model replicas or execution contexts per process;
- outstanding submissions per inference worker;
- inference batch cap and collection timeout;
- device streams and CUDA graph replay ordering;
- CPU affinity and NUMA placement;
- whether cache or model state is shared across processes;
- training overlap, which changes available GPU and CPU capacity.

Changing one axis changes the batch distribution and the resource pressure seen by the others. The report should not
present any single topology as a universal optimum.

### More processes can improve collection but add replicas and CPU pressure

The native self-play tuning study
[`self-play-cpp-final-tuning-20260720`](../../benchmarks/self-play-cpp-final-tuning-20260720/README.md) compared process,
search-thread, active-game, timeout, and NUMA arrangements. Moving from a high-process reference to four processes
per GPU with four search threads and more games per process improved its selected workload by about 39%; reducing to
two processes with more threads was slower despite larger potential batches. This established that aggregation,
process overhead, CPU placement, and model/cache replication had to be tuned together. Its exact internal thread
architecture predates the retained direct scheduler and is not the current node graph.

### More games improve saturation but lengthen a trajectory's wall time

[`chess-self-play-latency-rtx3060-20260812`](../../benchmarks/chess-self-play-latency-rtx3060-20260812/README.md) makes
the tradeoff concrete. A configuration with three processes and 1,024 games per GPU reached about 213,401 searches/s.
The selected latency-oriented point with two processes and 512 games reached about 201,404 searches/s, or 94.4% of
that aggregate throughput, while searches per game per second increased from about 8.68 to 24.59. Estimated searched
ply latency fell from about 30.2 seconds to 10.7 seconds.

Thus, adding active games can make the GPU look busier while delaying game completion, replay arrival, and the
policy/value feedback loop. Aggregate completed games may remain similar. A throughput-only choice can therefore
produce older training data without buying proportional learning volume.

### More inference workers can split rather than fill batches

Before CUDA graph submission, the same latency study found that one inference worker formed full batches frequently
but lost about 17% search throughput relative to two workers. “Percent full batches” was not a sufficient objective;
submission and result-processing latency still needed overlap.

The submission study
[`self-play-submission-8xrtx4070super-20260824`](../../benchmarks/self-play-submission-8xrtx4070super-20260824/README.md)
then reduced host submission cost from roughly 3,537 microseconds to 31 microseconds with CUDA graph replay. Full
topology throughput increased about 20.5% and CPU consumption fell about 62%. Once launch overhead was cheap, moving
from two inference workers to one increased the average batch from roughly 141 to 222 and removed duplicate CUDA
contexts and threads. The reason for the preferred worker count changed when the cost structure changed.

The graph multiworker follow-up
[`self-play-graph-multiworker-8xrtx4070super-20260824`](../../benchmarks/self-play-graph-multiworker-8xrtx4070super-20260824/README.md)
also exposed a correctness boundary: replaying graphs concurrently on separate streams in one process produced an
illegal device address. A shared per-device stream restored safe ordering. Two graph workers were then correct but
slower because they split the available leaves into smaller batches. Raising active games and the batch cap increased
peak throughput further, but stretched game duration and memory use.

The later CNN topology sweep
[`cnn-inference-throughput-rtx4070s-20260827`](../../benchmarks/cnn-inference-throughput-rtx4070s-20260827/README.md)
confirmed the retained shape for the float pipeline. Four processes, one worker per process, and two outstanding
submissions reached about 95,458 positions/s on one GPU in that benchmark; arrangements with two inference workers
per process were slower. The current final chess configuration retains four self-play processes per GPU, 512 games
per process, one inference worker, two outstanding slots, and a batch cap of 320. TensorRT becomes the inference
backend after bootstrap, but it does not change the ownership graph.

### Retained current topology, stated without implying universality

The current deployment uses multiplication rather than shared global services:

- four self-play processes per GPU;
- one native tree-owning scheduler per process, interleaving 512 games;
- one dedicated native inference worker per process;
- two bounded outstanding slots;
- a maximum inference batch of 320;
- a process-local model/backend instance, with GPU assignment set by Python orchestration.

This point is the retained optimum for the final chess workload, model, backend, and hardware envelope. The report
should state the topology and its causal evidence, while making clear that a different network shape or launch cost
can move the optimum.

## Interactive latency versus saturated service throughput

Self-play, evaluation, and interactive search share the native search implementation, but they exercise it under
different demand regimes.

| Property | Saturated self-play or batched evaluation | Interactive analysis or move choice |
| --- | --- | --- |
| Active trees | Hundreds or thousands across processes/jobs | Usually one tree, occasionally a small number |
| Primary objective | Aggregate useful positions, searches, or completed games per unit wall time | Useful search completed before one move deadline |
| Batch formation | Many independent leaves are naturally available | Leaves are causally exposed by one evolving tree |
| Acceptable queueing | Some queueing can improve accelerator occupancy | Queueing consumes the user's time budget |
| Concurrency risk | Slow trajectories make replay less fresh | Virtual loss and parallel completion order can change visit allocation and the selected move |
| Required reporting | Throughput, batch distribution, games and samples completed, resource use, trajectory latency | Deadline overshoot, achieved visits/nodes, move latency distribution, result-processing time, deterministic exact-budget checks |

The integrated interactive study
[`integrated-interactive-rtx3060-20260722`](../../benchmarks/integrated-interactive-rtx3060-20260722/README.md)
used a single serial tree owner and direct inference workers. Two replicas at batch 64 reached about 9,762 median
searches/s and three reached about 9,987; a fourth regressed. The best point was about 2.8 times its general cached
client baseline, but result processing still occupied roughly 41–54% of wall time.

[`interactive-result-processing-rtx3060-20260722`](../../benchmarks/interactive-result-processing-rtx3060-20260722/README.md)
then optimized that boundary from roughly 51 microseconds to 1.33 microseconds per returned position. With two
replicas, batch 64, and two outstanding submissions it reached about 18,115 searches/s, leaving inference wait as the
dominant cost. The study also checked deadline behavior, correctness, and deterministic execution rather than using
throughput alone.

The CPU-only smoke in
[`interactive-engine-local-20260721`](../../benchmarks/interactive-engine-local-20260721/README.md) illustrates why
search semantics matter: parallel search increased searches but, under a uniform policy, virtual loss changed visit
allocation enough to select a different move. Exact-node and timed parallel modes therefore require their own
correctness and repeatability checks. More searches per second is not automatically an equivalent decision process.

### Reporting rule

Never use saturated ladder or self-play throughput as a claim about the latency of one public interactive move. A
ladder can batch many independent games and is a service-capacity measurement. Public play exposes one or few trees
and is a response-time measurement. Both are valuable, but they answer different questions. The report's current
systems chapter already states this distinction in
[`05-systems-optimization.md`](../05-systems-optimization.md); the quantitative examples above provide its source
record.

## Decision ledger by mechanism

| Mechanism | What was learned | Current consequence |
| --- | --- | --- |
| Central Python pipe routing | Central serialization and response dispatch bottleneck the leaf path | No Python load balancer between tree selection and inference |
| Shared Python request/response queues | Decoupling alone does not repay queue, matching, and central-cache overhead | No general multiprocessing inference service in the native search loop |
| Per-process Python clients | Decentralized ownership beats a central router, but Python scheduling and replica costs remain | Process-local ownership retained; hot loop moved native |
| Fine-grained Python coroutines | Coroutine scheduling overwhelms tiny recursive search operations | Asynchrony occurs at coarse process/job and bounded native-slot boundaries |
| Custom native chess rules | Speed without perft correctness is unusable | Current chess transitions use the audited Stockfish-backed implementation |
| Duplicate search facades | Separate owners drift and obscure behavioral equivalence | One native search implementation serves self-play, evaluation, and interactive modes |
| General inference client | Flexible promises, queues, stacking, and dispatch are costly for a fixed leaf contract | Persistent typed buffers and direct bounded slots |
| Shared-tree inference workers | Concurrent mutation requires locks and complicates ordering | Inference workers never mutate trees; the search owner applies completions |
| More active games | Raises occupancy but can lengthen games and stale the learning loop | Topology balances aggregate throughput with per-game progress |
| More inference workers | Can overlap expensive submission, but can split batches and duplicate contexts after graph replay | One worker per current self-play process, two outstanding slots |
| Independent graph replay streams | Concurrent graph replay violated device-memory safety | Shared per-device ordering where multiple graph workers exist; current point avoids needless workers |
| Throughput as a universal metric | Saturated capacity hides trajectory and move latency | Separate self-play, evaluation-service, and interactive measurements |

## Implications for report structure and figures

The systems chapter should use these investigations as causal subsections rather than a timeline:

1. define tree and inference ownership;
2. compare process-boundary choices and explain why per-leaf Python IPC disappeared;
3. explain persistent buffers, bounded outstanding slots, and single-writer tree updates;
4. show the coupled topology tradeoff among batch quality, contexts, CPU load, and trajectory latency;
5. separate saturated capacity from interactive response time;
6. close with correctness constraints: perft, deterministic exact budgets, deadline tests, and CUDA stream ordering.

The main multi-panel SVG specified in
[`system-architecture-and-figure-dossier.md`](system-architecture-and-figure-dossier.md) must show only the retained
system. If a historical visual is valuable, use a separate compact comparison with four abstract request paths:

```text
central pipes       producers -> load balancer -> inference servers -> load balancer -> producers
shared queues       producers -> cache/queue -> inference servers -> response queue -> producers
direct clients      producer-local client -> model/service -> producer
retained native     tree owner -> bounded reusable slots -> inference worker -> same tree owner
```

Use gray or amber for rejected/superseded alternatives and the report's normal component colors only for the retained
path. Do not place benchmark numbers inside the current architecture diagram. A small adjacent plot or table is more
honest because each measurement has a different workload envelope.

## Source map

### Current authority

- [`chess-final-config.yaml`](../../../py/configs/production/chess-final-config.yaml): deployed process, game, worker,
  outstanding-slot, and batch-cap multiplicities.
- [`cpp/README.md`](../../../cpp/README.md): current Python/native ownership boundary.
- [`system-architecture-and-figure-dossier.md`](system-architecture-and-figure-dossier.md): checked current node and
  edge inventory plus SVG specification.

### Process and ownership alternatives

- [`history/optimizations/architecture.md`](../../history/optimizations/architecture.md): pipe, queue/cache-manager,
  and client arrangements; archival timings require qualification.
- [`history/pre-cpp-port/asyncio/README.md`](../../history/pre-cpp-port/asyncio/README.md): fine-grained coroutine
  experiment and synthetic scheduler-overhead measurement.
- [`history/pre-cpp-port/chess-port.md`](../../history/pre-cpp-port/chess-port.md): early native-port motivation and
  explicit warning about the superseded custom chess implementation.
- [`architecture/platform-rework.md`](../../architecture/platform-rework.md): closed ledger for removal of duplicate
  owners and consolidation on the native search engine.
- [`benchmarks/harnesses/direct-evaluation-inference.md`](../../benchmarks/harnesses/direct-evaluation-inference.md):
  general-client versus direct-slot isolation.
- [`self-play-direct-inference-4x4070-super-20260722`](../../benchmarks/self-play-direct-inference-4x4070-super-20260722/README.md):
  end-to-end direct self-play scheduler evidence.

### Topology and service regimes

- [`self-play-cpp-final-tuning-20260720`](../../benchmarks/self-play-cpp-final-tuning-20260720/README.md): historical
  process/thread/game/NUMA topology study.
- [`chess-self-play-latency-rtx3060-20260812`](../../benchmarks/chess-self-play-latency-rtx3060-20260812/README.md):
  saturation versus trajectory-latency tradeoff.
- [`self-play-submission-8xrtx4070super-20260824`](../../benchmarks/self-play-submission-8xrtx4070super-20260824/README.md):
  CUDA graph submission and changed worker-count optimum.
- [`self-play-graph-multiworker-8xrtx4070super-20260824`](../../benchmarks/self-play-graph-multiworker-8xrtx4070super-20260824/README.md):
  graph stream correctness, split batches, and high-concurrency tradeoff.
- [`cnn-inference-throughput-rtx4070s-20260827`](../../benchmarks/cnn-inference-throughput-rtx4070s-20260827/README.md):
  float-pipeline topology sweep supporting the retained process/worker/outstanding shape.
- [`integrated-interactive-rtx3060-20260722`](../../benchmarks/integrated-interactive-rtx3060-20260722/README.md):
  single-tree interactive inference topology.
- [`interactive-result-processing-rtx3060-20260722`](../../benchmarks/interactive-result-processing-rtx3060-20260722/README.md):
  result-processing optimization and interactive deadline validation.
- [`interactive-engine-local-20260721`](../../benchmarks/interactive-engine-local-20260721/README.md): parallel-search
  semantic smoke under a uniform policy.
- [`naive-python-mcts-rtx3060-20260816`](../../benchmarks/naive-python-mcts-rtx3060-20260816/README.md): deliberately
  naive Python scale reference with explicit comparability caveats.

## Uncertainties and claims requiring restraint

1. The early pipe/queue/client numbers do not have a locked manifest or raw artifacts. Use them only as archival,
   directional evidence.
2. The `asyncio` measurement isolates fine-grained scheduler overhead in a synthetic workload. It does not reject
   coarse asynchronous I/O or independently scheduled jobs.
3. No single controlled experiment decomposes the native transition into language, rules engine, batching, IPC,
   buffer reuse, and ownership effects. Do not assign a precise fraction of the gain to one factor.
4. The early custom chess port failed correctness validation. Its speed claims must not be attached to the current
   Stockfish-backed engine.
5. Direct-client, CUDA-graph, CNN, and interactive benchmarks use different hardware and workload envelopes. Compare
   variants within each dossier; do not form a synthetic speedup chain across them.
6. The retained topology is configuration authority for the final chess workload, not proof of an optimum for every
   network, GPU, backend, or search budget.
7. Evaluation throughput may look like self-play saturation because many matches can be active, but evaluation does
   not own replay publication or trainer feedback edges. Preserve that architectural separation.
8. Interactive timed search needs final public-engine deadline and latency measurements if the report makes a user-
   facing response-time claim. Saturated ladder measurements cannot fill that placeholder.
