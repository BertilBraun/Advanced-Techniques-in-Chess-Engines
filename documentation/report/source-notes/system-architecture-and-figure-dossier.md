# System architecture and figure source dossier

## Purpose

This dossier is the source record for a publication-quality system figure and the later system-method prose. It
describes the implementation that exists now. It deliberately does not organize the system by experimental run,
development period, or discarded architecture. Historical designs belong in the investigation chapters, not in the
main architecture figure.

The eventual figure must let a reader answer four questions without reading the implementation:

1. Which work runs in Python and which work runs in the native C++ extension?
2. Which processes own the GPUs, and which operations can overlap?
3. How does a completed game become a replay sample, training credit, checkpoint, and deployed inference engine?
4. Which boundaries are durable, and what happens when a worker, evaluator, trainer, or complete run stops?

This file is not report prose and is not the final SVG. It is a checked inventory of nodes, edges, ownership,
synchronization, persistence, and failure behavior.

Substantial runtime alternatives that led to these boundaries are preserved separately in
[`runtime-architecture-alternatives.md`](runtime-architecture-alternatives.md). That dossier covers Python
pipe, queue, client, and `asyncio` designs; the migration of search ownership into the native engine; process and
inference-worker topology studies; and the distinction between interactive latency and saturated service
throughput. Those investigations belong in the systems discussion, but their superseded nodes and edges must not
be drawn as if they were part of the current runtime.

## Authority and scope

The authority order used for this dossier is:

1. current implementation;
2. the fully expanded final chess configuration;
3. current architecture documents where they agree with the implementation;
4. tests as executable clarification of boundary behavior.

The architecture documents include historical material and superseded proposals. They are useful as rationale but
are not sufficient authority for a runtime edge. In particular, the live replay implementation is the parallel,
file-staged, columnar pipeline in the current source. The earlier direct coordinator materialization design is not
shown.

Primary entry points:

- [`py/train.py`](../../../py/train.py) owns process entry, startup, top-level outcome recording, signals, and resource
  telemetry.
- [`Coordinator`](../../../py/src/training/coordinator.py) owns the live orchestration loop.
- [`chess-final-config.yaml`](../../../py/configs/production/chess-final-config.yaml) is the readable authority for the
  deployed topology and retained recipe.
- [`cpp/README.md`](../../../cpp/README.md) defines the Python/native ownership boundary.

Operational launch and archival are relevant to the artifact boundary but should sit outside the central learning
loop in the figure. [`deployment/run_control.sh`](../../../deployment/run_control.sh) validates an approved clean
checkout, launches the supervised process, requests checkpoint-safe stopping, and preserves evidence.

## One-sentence system model

A synchronous Python coordinator supervises persistent self-play and distributed-training process groups, a parallel
replay-materialization subsystem, and short-lived evaluation jobs; native C++ search instances batch tree leaves into
TorchScript or TensorRT inference, completed trajectories move through an atomic file-and-shard boundary into a
columnar memory-mapped replay store, and rank zero publishes the next durable training checkpoint from which the
coordinator derives the deployable inference engine.

## Current deployed topology

The fully expanded chess recipe configures the following topology:

| Resource | Current multiplicity | Ownership and placement |
| --- | ---: | --- |
| Coordinator | 1 process | Python main process |
| Self-play | 32 processes | Four processes assigned to each of eight GPUs |
| Active games | 512 per self-play process | Interleaved inside one Python worker and one native search instance |
| Native inference pipelines | 1 per self-play process | One dedicated native inference thread and two outstanding slots |
| Training | 8 persistent DDP ranks | One NCCL rank per GPU |
| Replay materialization | 8 processes | CPU processes supervised by one coordinator-process thread |
| Evaluation | Up to 16 concurrent processes | Jobs cycle across the eight GPUs |
| Replay store | 1 file | Coordinator-owned writable circular mmap; ranks open read-only views |

During a training quantum, the coordinator requests that two of the four self-play processes on each GPU pause. The
other half may continue generating games while all eight trainer ranks use the GPUs. Materialization processes also
continue producing sealed shards. The coordinator does not append those shards while training holds the replay
snapshot lock, so staging is bounded and naturally applies backpressure. Evaluation jobs that were already running
continue independently, but the coordinator does not collect results or schedule new jobs until the blocking training
call returns.

The implementation supports other multiplicities. The figure should visually encode the concrete deployed topology
while using multiplication badges rather than drawing dozens of repeated workers.

## Layer and node inventory

Stable node identifiers below are intended for the SVG source and review comments. They are not abbreviations for
evidence quality and should not appear as unexplained prose shorthand in the report.

### Layer 1: operational and configuration boundary

| ID | Node | Owner | Responsibility |
| --- | --- | --- | --- |
| `OPS` | Run control and supervisor | Shell/host supervisor | Validate clean source and approval identity; launch, stop, inspect, preserve, and fetch the run. |
| `CFG` | Resolved experiment configuration | Python startup | Typed, immutable configuration shared with every spawned child as serialized JSON. |
| `RUNMETA` | Run manifest and resolved environment | Python startup | Bind source, configuration, hardware, dependency, dataset, opening, engine, and initial-checkpoint identities. |
| `LIMITS` | Resource telemetry and run-limit monitor | Python main process plus telemetry worker | Observe cost, time, disk, memory, descriptors, and manual-stop state; record final outcome. |

These boxes should be small and peripheral. They establish reproducibility and safe control but are not in the
per-position hot path.

### Layer 2: Python control plane

| ID | Node | Process ownership | Responsibility |
| --- | --- | --- | --- |
| `COORD` | Coordinator | Main process | Serialize state transitions, decide whether to append, train, evaluate, pause, resume, retain, or stop. |
| `LEDGER` | Credit ledger | Coordinator object with atomic JSON persistence | Derive model generation from completed optimizer steps; reconcile sample-derived credits; atomically commit active checkpoints and consumed credits. |
| `SPGROUP` | Self-play group | Coordinator object | Own worker processes and duplex pipes; apply desired states; supervise death and restart. |
| `REPLAYMGR` | Replay manager | Coordinator object | Own writable replay store, dispatcher/materializer supervision, staged-shard append, rejection alarm, and replay snapshot lock. |
| `SESSION` | Training session | Coordinator object | Own the active and candidate trainer groups, progressive state, promotion decision, and public checkpoint selection. |
| `EVALMGR` | Evaluation manager | Coordinator object | Persist cadence state, checkpoint publications, pending jobs, device cycling, adaptive ladder state, results, and checkpoint-retention requirements. |
| `REPORT` | Training reporter | Coordinator object | Emit loss, throughput, replay, self-play, resignation, and lifecycle telemetry. |
| `RETENTION` | Checkpoint retention | Coordinator object | Keep resumable milestones, recent inference artifacts, evaluation references, and the active checkpoint; remove unneeded payloads. |

The coordinator owns the decisions, but it does not own a loaded training model, inspect replay rows, execute search,
or play evaluation games.

### Layer 3: parallel Python process groups

| ID | Node | Process ownership | Responsibility |
| --- | --- | --- | --- |
| `SPWORKERS` | Self-play workers | Persistent spawned processes | Hold active games, a native search object, local restart-state archive, resignation policy, and model identity; publish completed games. |
| `MATTHREAD` | Replay materialization supervisor | Background thread in coordinator process | Bounded inbox dispatch, report collection, worker liveness, and worker restart. |
| `MATWORKERS` | Materialization workers | Persistent spawned CPU processes | Parse completed games, reconstruct positions and targets, truncate sparse policies, encode aligned columns, seal shards, and quarantine rejected inputs. |
| `TRAINERS` | Trainer groups | Persistent spawned processes | Receive one typed quantum command per rank, sample a common replay snapshot, train under DDP, synchronize, and return typed results. |
| `RANK0` | Rank-zero publisher | One trainer rank | Write model, optimizer, QAT state, inference artifact, and checkpoint manifest. |
| `EVALJOBS` | Evaluation jobs | Short-lived spawned processes | Load the frozen job definition and checkpoint, run dataset or match evaluation, and atomically write a typed result or failure. |
| `TRTPUB` | TensorRT publisher | Short-lived subprocess invoked at deployment resolution | Refit a compatible template from checkpoint ONNX, verify numerical fidelity on probe states, and atomically publish/cache the engine. |
| `OPPONENT` | External reference engine | Child process owned by an evaluation job | Supply Stockfish or KataGo actions under the job's fixed protocol and limits. |

`TRTPUB` is not a persistent publisher service. It is invoked when a checkpoint's deployment path is resolved. The
checkpoint manifest continues to identify the training-produced inference artifact; the returned deployment
checkpoint is an in-memory reference whose inference path and digest identify the derived engine.

### Layer 4: native C++ hot path

| ID | Node | Ownership | Responsibility |
| --- | --- | --- | --- |
| `BINDING` | Pybind boundary | Loaded into self-play/evaluation process | Expose typed chess/Go state, root, request, result, and search construction surfaces. |
| `GAME` | Game rules and encoding | Native search instance | Legal actions, immutable transitions, terminal detection, action IDs, packed planes, and inference dimensions. |
| `GAMES` | Interleaved active games | Self-play worker | Supply roots across many games so leaf inference can be batched. |
| `TREE` | Retained search trees | Native `BatchedGameSearch` | Selection, reservations, expansion, backup, root noise, forced playouts, visit retention, and tree reset on model refresh. |
| `EXEC` | Batched search executor | Native search instance, tree-owner thread | Schedule heterogeneous root requests, collect leaves, fill inference batches, track outstanding work, and consume completions. |
| `PIPE` | Inference pipeline | One or more per native search; one in the deployed recipe | Own fixed slots and a dedicated inference thread; enforce ordered completion and propagate failures. |
| `RUNNER` | Inference runner | Inside each pipeline | Own persistent tensors, execution options, CUDA stream/graphs where applicable, and loaded TorchScript or TensorRT model. |
| `BACKEND` | TorchScript/TensorRT backend | Inside inference runner on assigned GPU | Map encoded state batches to policy logits and WDL probabilities. |
| `LEGAL` | Legal-output processing | Native inference completion path | Validate WDL, gather legal action logits, stable-sort action IDs, and normalize priors only over legal moves. |

The native runtime is shared by self-play and searched evaluation. The evaluation subsystem also supports a Python
policy selector for definitions that explicitly request direct-policy play, but the deployed chess ladder obtains its
one-visit policy signal through the native search path.

### Layer 5: durable data and artifact plane

| ID | Artifact | Writer | Readers / consumers |
| --- | --- | --- | --- |
| `INBOX` | Atomic completed-game JSON files | Self-play workers | Dispatcher thread |
| `SUSPENDED` | Per-worker suspended-game file | Self-play worker during graceful stop | Same worker identity after restart |
| `RESTARTDB` | Per-worker SQLite restart-state archive | Self-play worker | Start-position sampler in that worker |
| `WORKDIRS` | Per-materializer source directories | Dispatcher by same-filesystem rename | Assigned materialization worker |
| `STAGING` | Sealed columnar shard data and manifest | Materialization workers | Replay manager |
| `REJECTED` | Quarantined malformed/unmaterializable games | Materialization workers | Operator/postmortem inspection |
| `REPLAY` | Columnar circular mmap and header | Replay manager | Every trainer rank through read-only mappings |
| `CREDITFILE` | Atomic credit-ledger JSON | Coordinator | Coordinator on restart |
| `PROGRESSIVE` | Atomic progressive-training state | Training session | Training session on restart |
| `CANDIDATES` | Private candidate checkpoints | Rank zero of each candidate trainer group | Training session and candidate trainers |
| `CHECKPOINT` | Public model, optimizer, inference artifact, QAT state, manifest | Rank zero directly, or training session when publishing a selected candidate | Coordinator, trainer restart, self-play deployment, evaluation, retention, archival |
| `ENGINE` | Derived TensorRT engine and verification record | TensorRT publisher | Native self-play/evaluation runtime |
| `EVALSTATE` | Evaluation manager state and checkpoint-reference manifest | Evaluation manager | Evaluation manager on restart and checkpoint retention |
| `EVALRESULT` | Atomic typed evaluation result or typed failure plus optional traceback | Evaluation job or manager deadline handler | Evaluation manager, TensorBoard, progressive plateau controller, final export |
| `TELEMETRY` | TensorBoard event streams, resource samples, logs, outcome | Coordinator, workers, trainers, evaluator, telemetry process | Operator, final analysis, archival |
| `ARCHIVE` | Preserved evidence bundle and checksums | Run control | Workstation and report analysis |

## Directed interaction inventory

The SVG should distinguish four edge styles:

- **solid blue:** typed control message or direct synchronous call;
- **solid purple:** GPU/native computation path;
- **thick green:** durable data or artifact publication;
- **dashed orange:** telemetry, observation, supervision, or feedback that does not carry training samples.

### Startup and control edges

| From | To | Payload / action | Synchronization |
| --- | --- | --- | --- |
| `OPS` | `CFG` | Approved configuration path, expected source identity, approval file | Launch is rejected before training if identities or checkout cleanliness disagree. |
| `CFG` | all spawned Python children | Serialized resolved experiment configuration | Passed at process spawn; children reconstruct the typed configuration. |
| `CFG` | `RUNMETA` | Configuration, environment, engine and dataset identities | Written before the coordinator begins. |
| `COORD` | `SPWORKERS` | `running`, `paused`, or `stopped` desired state over a duplex pipe | Startup and explicit apply wait for responses; pause requests are asynchronous. |
| `SPWORKERS` | `COORD` | Applied state with worker ID, accepted checkpoint generation/digest, and optional search statistics | Initial `running` response follows a completed model load. Later refresh response precedes activation, which occurs before the next batch. |
| `COORD` | `TRAINERS` | Replay description, source/target progress, learning rate, and resolved objective | Blocking call; every rank receives the same typed quantum command. |
| `TRAINERS` | `COORD` | Per-rank losses, gradients, distributions, duration; rank zero also returns checkpoint reference | Coordinator waits for all ranks and validates exact progress agreement. |
| `COORD` | `EVALMGR` | Current public checkpoint and collection/scheduling calls | Nonblocking with respect to already running jobs; scheduling occurs at coordinator loop boundaries. |
| `EVALMGR` | `EVALJOBS` | Serialized experiment and typed job JSON | Process spawn; result transport is the durable result file rather than a pipe. |

### Self-play and native inference edges

| From | To | Payload / action | Synchronization |
| --- | --- | --- | --- |
| `CHECKPOINT` | `TRTPUB` | Checkpoint ONNX, compatible template, fixed batch shape, optional probe positions | Synchronous deployment-path resolution; cached compatible output may be reused. |
| `TRTPUB` | `ENGINE` | Refit engine plus fidelity metrics | Engine is returned only after refit and verification complete. |
| `ENGINE` | `RUNNER` | Deserialized backend model | Initial load is part of worker startup; later refresh prepares all pipeline models before committing them. |
| `GAMES` | `TREE` | Current roots and per-root search requests | One worker advances a batch of active games at a time. |
| `TREE` | `EXEC` | Selected, reserved leaves from heterogeneous roots | Reservations prevent duplicate in-flight work inside the tree search. |
| `EXEC` | `PIPE` | Encoded packed-plane positions placed into an acquired fixed slot | Submission is asynchronous; multiple slots can be outstanding. |
| `PIPE` | `BACKEND` | Fixed or bucketed tensor batch | Dedicated native inference thread; CUDA stream/graph execution where configured. |
| `BACKEND` | `LEGAL` | Policy logits and WDL probabilities | Completion is recorded by event; consumer waits only when results are needed. |
| `LEGAL` | `TREE` | Legal priors and value result | Tree expansion and backup release reservations; inference results are consumed in submission order. |
| `TREE` | `GAMES` | Root visit distribution, root/network values, corrections, stop reason, selected action | Python worker records the search observation and advances or completes each game. |

The executor batches leaves across many games inside one process. There is no cross-process inference cache or
shared inference server in the current architecture. Each self-play process owns its own native search and inference
pipeline.

### Completed game to replay and credits

| From | To | Payload / action | Synchronization |
| --- | --- | --- | --- |
| `SPWORKERS` | `RESTARTDB` | Eligible uncertain positions and untried alternative actions from completed trajectories | Local SQLite transaction; candidate reservation is also transactional. |
| `SPWORKERS` | `INBOX` | Complete typed trajectory with sparse visit targets, values, corrections, weights, identity, and termination metadata | Temporary file then atomic replace; the worker syncs after publishing its batch. |
| `MATTHREAD` | `WORKDIRS` | Bounded, least-loaded dispatch by same-filesystem rename | No file copy and no centralized queue journal. |
| `WORKDIRS` | `MATWORKERS` | Ordered source files | Each materializer consumes only its assigned directory. |
| `MATWORKERS` | `STAGING` | Encoded columnar shard data followed by sealed manifest | Shard identity derives from layout, worker, and counter span; sources are removed only after sealing. |
| `MATWORKERS` | `REJECTED` | Invalid individual source game | Quarantine; rolling rejection-rate breach becomes a fatal coordinator-visible error. |
| `MATWORKERS` | `MATTHREAD` | Materialized/rejected game and row counts through multiprocessing queue | Supervisor drains reports and restarts dead materializers. |
| `STAGING` | `REPLAYMGR` | Sealed shard readers and metadata | Coordinator appends every staged shard under the replay lock. |
| `REPLAYMGR` | `REPLAY` | Column arrays and updated FIFO/header state | Store flush completes before shard artifacts are deleted. |
| `REPLAY` | `LEDGER` | Monotonic total materialized rows | Credits are reconciled only after append and flush, making durable replay the credit ground truth. |
| `STAGING` metadata | resignation calibrator in `COORD` | Continuation outcomes, terminal reasons, thresholds, search observations | Calibrator updates when the coordinator appends games, then publishes a policy on the next worker state transition. |

Replay capacity may evict old live rows, but total appended rows remain monotonic for credit reconciliation. Training
credits are therefore earned by materialization, not by the current number of live replay rows.

### Replay to optimizer and publication

| From | To | Payload / action | Synchronization |
| --- | --- | --- | --- |
| `REPLAYMGR` | `TRAINERS` | Immutable replay description: path, head, size, logical capacity, maximum capacity, and layout | Coordinator holds the replay snapshot lock for the complete blocking quantum. |
| `REPLAY` | each DDP rank | Read-only mmap gathers, sparse-to-dense target reconstruction, symmetry augmentation, optional surprise-prioritized sampling | Sampling seed and source optimizer step produce one global sample order; each rank takes its nonoverlapping local slice. |
| host batch loader | trainer GPU | Prefetched pinned batches and nonblocking transfers | One CPU prefetch thread and shared CUDA transfer stream per rank. |
| DDP ranks | DDP ranks | Gradients and barriers | NCCL synchronization; model state advances together. |
| `RANK0` | `CANDIDATES` or `CHECKPOINT` | Model state, optimizer state, QAT state, trimmed inference artifact, then final manifest | Other ranks cross a distributed barrier after publication. Only a complete manifest makes a checkpoint adoptable. |
| `SESSION` | `CHECKPOINT` | Selected active candidate copied into public generation namespace | Private candidate checkpoints never become self-play/evaluation inputs directly. |
| `CHECKPOINT` | `LEDGER` | Completed optimizer steps and public checkpoint reference | Atomic ledger commit consumes exactly one quantum of available credit. |
| `LEDGER` | `TRTPUB` | Newly active public checkpoint selected for deployment | Engine derivation occurs before the new desired state is sent to self-play workers. |
| `COORD` | `RETENTION` | Active generation plus evaluation-required checkpoint generations | Retention runs after publication/reporting and never removes the active or referenced inference artifact. |

For progressive sizing, the training session may own more than one persistent trainer group. It trains the required
active and candidate models sequentially against the same replay snapshot, records candidate-versus-active match
observations from the evaluator, chooses the active model under the configured promotion rule, and publishes only
that choice to the public checkpoint namespace.

### Evaluation and feedback

| From | To | Payload / action | Synchronization |
| --- | --- | --- | --- |
| `EVALMGR` | `EVALJOBS` | Cadence boundary, checkpoint frozen at that boundary, device, definition, opening/dataset identity, deadline | Jobs run independently up to configured concurrency. |
| `CHECKPOINT` | evaluation candidate selector | Policy-only TorchScript/ONNX-derived artifact or native searched deployment engine | Each job owns its candidate model/search instance. |
| openings/dataset artifacts | `EVALJOBS` | Frozen paired openings or fixed probe positions | Read-only, hash-bound by run preparation. |
| `OPPONENT` | `EVALJOBS` | Reference actions | Job-local engine process and protocol. |
| native evaluation search | `EVALJOBS` | Candidate actions at the configured search budget | Uses the same native rules/search/inference stack as self-play, without root noise or forced playouts. |
| `EVALJOBS` | `EVALRESULT` | Game-level records, aggregate statistics, duration, or typed failure | Atomic result write; deadline failure may instead be written by the manager. |
| `EVALRESULT` | `EVALMGR` | Completed job | Manager updates adaptive ladder state, TensorBoard, pending state, and checkpoint references. |
| primary ladder observations | `SESSION` | Time-boundary Elo observations | Feedback controls when a larger candidate begins; it does not directly alter model weights. |
| candidate match observations | `SESSION` | Candidate score, game count, and evaluation boundary | Two consecutive scores at or above the configured threshold authorize promotion; failed jobs add no observation. |

Paired-opening match execution reverses candidate color. Concurrent match groups may share one candidate selector so
candidate positions from several opponent groups form one inference batch.

## Synchronization phases for the figure

The architecture is easier to understand as four repeating phases, not as a chronology of experimental runs.

### Phase A: game production and replay preparation

- Running self-play processes repeatedly advance one batch of active games.
- Completed trajectories are durably published to the inbox.
- The dispatcher and materialization workers independently move games into sealed columnar shards.
- The coordinator frequently appends staged shards, flushes replay, and reconciles credits.
- Due evaluation jobs are scheduled while the coordinator is waiting for enough credits or live samples.

### Phase B: training snapshot and overlap

- Once one quantum of credit and at least one global batch of live replay exist, selected self-play workers receive an
  asynchronous pause request.
- The coordinator captures a replay description while holding the replay lock.
- Persistent DDP ranks train one quantum; configured unpaused self-play workers and already-running evaluations may
  continue.
- Materializers may continue sealing shards, but the coordinator cannot append them until the snapshot lock is
  released.

### Phase C: durable publication and activation

- Rank zero writes checkpoint payloads and writes the manifest last.
- The training session selects/publishes the public checkpoint.
- The coordinator commits optimizer progress, consumed credits, and active checkpoint atomically in the ledger.
- The deployment engine is resolved/refit and hash-bound.
- Workers receive the new checkpoint and resignation policy. They return optional completed-generation statistics,
  accept the new identity, and activate the model before their next search batch.
- Reporting and retention run after the checkpoint transition.

### Phase D: evaluation feedback

- Cadence boundaries select the public checkpoint that existed at the boundary, not merely the newest checkpoint at
  launch time.
- Short-lived jobs evaluate that frozen identity and publish durable results.
- The manager aggregates ladder results and passes primary Elo observations to progressive-sizing control.
- Required checkpoint references feed retention so pending/resumable evaluation artifacts survive cleanup.

## Ownership and concurrency invariants

These statements should either appear visually or be available in the figure caption:

1. Only the coordinator writes the public credit ledger and writable replay store.
2. Only rank zero writes a checkpoint for a trainer group; only the training session publishes a private candidate
   into the public checkpoint namespace.
3. Self-play workers do not send trajectories through their control pipes. They publish them through atomic files.
4. Evaluation jobs do not return large results through process pipes. They publish typed result files.
5. Trainer ranks read the same immutable replay snapshot and communicate gradients through NCCL.
6. The tree owner and inference thread are separate inside each native pipeline; inference slots provide bounded
   asynchronous overlap.
7. There is no inference service shared between self-play processes and no production neural inference cache.
8. The checkpoint manifest is the commit marker for training artifacts; the sealed shard manifest is the commit
   marker for materialized replay; atomic JSON replacement is the commit marker for ledgers and evaluation state.
9. Deployment engines are derived artifacts. A public checkpoint remains the canonical training identity.
10. Model generation is derived from completed optimizer steps; it is not a separately writable progress counter.

## Failure, restart, and recovery paths

### Self-play worker failure

- The coordinator detects an exited process without blocking the main loop.
- The slot is retired, its pipe is closed, and restart is attempted with backoff.
- The replacement receives the current deployment checkpoint and resignation policy.
- Startup handshake waits for a completed model load and verifies the applied state.
- Completed-game files already published remain durable.
- In-memory active games from a crash may be lost. Graceful stop is different: the worker serializes resumable games
  before acknowledging `stopped`.
- The health monitor ends the run if too few workers remain live for too long.

### Replay materializer failure or bad input

- The supervisor thread restarts a dead materialization process for the same worker directory.
- A sealed shard is recoverable even if source unlinking did not finish: the worker adopts the matching manifest and
  removes already represented sources without resealing.
- Startup removes temporary files, removes unmanifested shard data, asserts same-filesystem rename boundaries, and
  returns sources from orphaned worker directories to the inbox.
- One bad game moves to `rejected/` and does not block its shard-mates.
- A configured rolling rejection-rate ceiling turns systematic schema/layout breakage into a fatal run error.
- Replay header tearing during a process crash is not transactionally journaled; the design intentionally accepts a
  small recovery discrepancy rather than a second replay or write-ahead log.

### Trainer-rank failure

- Any failed rank response or dead connection terminates the complete trainer group.
- The quantum is not committed to the credit ledger.
- Artifacts without a complete final manifest are not adoptable.
- The run exits for an explicit restart rather than attempting an in-process distributed retry.
- On restart, a complete uniquely-next checkpoint may be adopted when the ordinary fixed session owns publication;
  progressive training instead recovers through its persisted pending-quantum state and private checkpoints.

### Evaluation failure

- Execution exceptions produce a typed failed result and traceback.
- A missed deadline terminates, then kills if necessary, the job process group and records a typed deadline failure.
- A child that exits without an artifact becomes a typed missing-artifact failure.
- Evaluation failure does not stop training.
- Pending jobs and checkpoint references are persisted. On restart, jobs are resumed only if all referenced artifacts
  still exist; otherwise the manager records failure rather than silently substituting a newer checkpoint.

### Coordinator or host stop

- A normal stop closes evaluation under its grace policy, sends `stopped` to workers, closes trainer groups, stops
  materializers, flushes/closes replay, saves the ledger, and writes the run outcome.
- A termination signal is converted into an orderly top-level stopped outcome where possible.
- Manual stop, cost, wall time, descriptor pressure, host-memory pressure, disk pressure, and self-play health are
  explicit stop conditions.
- Run control preserves the resolved configuration, manifests, checkpoints, evaluations, logs, telemetry, and
  checksums before ephemeral compute is released.

## Proposed publication SVG

### Overall canvas

- View box: `0 0 1800 1000`; intended to remain legible at a two-column page width and crisp when enlarged.
- White or very light neutral background.
- Three labeled panels plus a compact evaluation-feedback inset. Do not encode the project as a historical timeline.
- Use native `<text>`, `<path>`, `<marker>`, `<pattern>`, and `<g>` elements so labels remain searchable and the
  source remains editable.
- Minimum final rendered type size: 8 pt for edge labels, 9 pt for node labels, 11 pt for panel titles.

### Panel layout

#### Panel A — Ownership and throughput path (`x=40..1760`, `y=50..260`)

Show only the ownership boundaries needed to understand the learning loop:

- Python orchestration owns configuration, process lifecycle, replay, training, evaluation, and publication;
- C++ owns chess state, legal actions, MCTS trees, and batched search execution;
- TensorRT on the GPUs evaluates leaf batches;
- completed games return to Python for replay and training;
- the published model returns to the native inference path.

Use one compact GPU group rather than drawing every process slot. Exact worker counts, pauses, CUDA streams, and
contention topology belong in the systems appendix and final-configuration table.

This panel answers **who owns each part of the throughput-critical loop**. Avoid recovery and artifact-lifecycle
detail here.

#### Panel B — Native self-play and inference loop (`x=40..860`, `y=300..740`)

Use a nested-box cutaway of one self-play process:

```text
Python self-play worker
├── 512 interleaved games
├── local restart-state archive
└── pybind native search
    ├── retained trees / selection / reservations
    ├── batched search executor
    └── inference pipeline
        ├── leaf batching
        └── TensorRT deployment model
```

Draw a clockwise loop:

`active positions → tree selection → leaf batch → encoded tensors → GPU inference → legal priors + WDL → expansion
+ backup → visit target + selected move → next positions`.

Show model refresh entering the inference runner from above and completed trajectories leaving the worker toward
Panel C. Do not add cache, fixed-slot, CUDA-graph, or restart annotations to the main figure.

#### Panel C — Replay, training, and publication (`x=900..1760`, `y=300..740`)

Use a predominantly top-to-bottom pipeline:

```text
atomic game inbox
    ↓ target materialization
columnar replay
    ↓ sampled batches
DDP trainer → checkpoint → TensorRT export + fidelity check → deployed model
```

Show the primary search policy and game outcome entering replay, and show the deployed model closing the loop back to
Panel B. Omit staging, locks, credit accounting, quarantine, candidate checkpoints, and recovery markers. Those are
important implementation contracts but not necessary to explain why throughput enables learning.

#### Evaluation feedback inset (`x=350..1450`, `y=790..950`)

Use one compact chain:

`published checkpoint → native policy/search ↔ Stockfish → paired-game result → Elo/selection feedback`.

Route the plateau observation upward to the training session using a dashed orange feedback edge. Pair openings should
be depicted as one opening feeding two color-reversed games.

Do not show recovery, failure paths, progressive-candidate state, evidence archival, or host-stop behavior in the
main SVG. They may remain documented in the architecture dossier but do not need a second publication figure.

### Color and shape legend

| Visual | Meaning |
| --- | --- |
| Rounded light-blue rectangle | Python owner/object |
| Dark-blue rectangle with process badge | Python process or process group |
| Purple nested rectangle | Native C++ component |
| Gold GPU card | CUDA device or device-resident backend |
| Green cylinder/document | Durable file, mmap, database, or artifact |
| Grey external box | Stockfish or another evaluation boundary |
| Solid blue arrow | Control call/message |
| Solid purple arrow | Native/GPU computation |
| Thick green arrow | Durable data/artifact publication |
| Dashed orange arrow | Evaluation or selection feedback |

Do not use single-letter legend codes. Every edge class must also differ by stroke pattern or width so the figure
works in grayscale and for color-vision deficiencies.

### Figure caption facts

The caption should explicitly state:

- self-play batching occurs inside each process across hundreds of games rather than through a shared inference
  service;
- C++ owns the latency-sensitive game/search loop while Python owns replay, training, evaluation, and publication;
- the loop's systems purpose is to turn many concurrent searches into enough fresh training positions per hour;
- TensorRT evaluates batched leaves and the published model closes the self-play/training loop;
- evaluation reuses the native policy/search stack against Stockfish and feeds strength measurements back into model
  selection.

### SVG implementation requirements

1. Group every panel and every repeated process group with stable semantic IDs.
2. Keep arrowheads outside node fills and route crossings through gutters.
3. Add `<title>` and `<desc>` to the root and each panel for accessibility.
4. Avoid raster screenshots, embedded fonts, filters, and CSS that depends on browser-specific features.
5. Use a font stack such as `Inter, Segoe UI, Arial, sans-serif` and monospace only for artifact names where needed.
6. Include a compact print-safe legend inside the SVG rather than relying solely on surrounding prose.
7. Test at full size, at the report's final column width, in grayscale, and with text extraction.
8. Keep detailed topology multiplicities out of the main figure; source them from the final configuration in the
   systems appendix or recipe table.

## Source map

### Orchestration, topology, and lifecycle

- [`py/src/training/coordinator.py`](../../../py/src/training/coordinator.py): authoritative loop, sequencing,
  backpressure, pauses, replay append, training, activation, evaluation collection, reporting, and shutdown.
- [`py/src/training/self_play_group.py`](../../../py/src/training/self_play_group.py): worker process/pipe ownership,
  asynchronous pause, restart handshake, backoff, and close behavior.
- [`py/src/self_play/protocol.py`](../../../py/src/self_play/protocol.py): exact desired/applied control messages.
- [`py/src/self_play/process_runtime.py`](../../../py/src/self_play/process_runtime.py): worker event loop and the
  distinction between initial load and staged refresh activation.
- [`py/src/training/session.py`](../../../py/src/training/session.py): fixed/progressive session boundary, private
  candidates, publication, promotion, and recovery.
- [`py/src/training/credit_ledger.py`](../../../py/src/training/credit_ledger.py): progress, credit, checkpoint commit,
  and fixed-session adoption rules.
- [`py/src/training/run_limits.py`](../../../py/src/training/run_limits.py): explicit resource stop conditions.

### Self-play and native search

- [`py/src/self_play/worker.py`](../../../py/src/self_play/worker.py): interleaved games, search observations,
  publication, starts, resignation, suspension, restart archive, and model refresh.
- [`py/src/self_play/completed_game.py`](../../../py/src/self_play/completed_game.py): durable game schema and atomic
  publication.
- [`py/src/self_play/restart_archive.py`](../../../py/src/self_play/restart_archive.py): local SQLite archive,
  reservation, eligibility, and recovery paths.
- [`py/src/games/implementation.py`](../../../py/src/games/implementation.py): common game/native configuration and
  deployment checkpoint resolution.
- [`py/src/games/chess/training.py`](../../../py/src/games/chess/training.py): chess search construction and evaluation
  deployment specialization.
- [`cpp/src/search/SelfPlay.hpp`](../../../cpp/src/search/SelfPlay.hpp): native self-play request/result facade.
- [`cpp/src/search/SearchEngine.hpp`](../../../cpp/src/search/SearchEngine.hpp): tree ownership and model refresh.
- [`cpp/src/search/SearchExecutor.hpp`](../../../cpp/src/search/SearchExecutor.hpp): leaf scheduling, reservations,
  batching, completion, and cancellation.
- [`cpp/src/search/InferencePipeline.hpp`](../../../cpp/src/search/InferencePipeline.hpp) and
  [`InferencePipeline.cpp`](../../../cpp/src/search/InferencePipeline.cpp): slots, inference thread, persistent
  buffers, CUDA events/graphs, backend refresh, and legal-output processing.
- [`cpp/src/search/TensorRtInferenceModel.cpp`](../../../cpp/src/search/TensorRtInferenceModel.cpp): native TensorRT
  execution boundary.

### Replay and training

- [`py/src/replay/dispatch.py`](../../../py/src/replay/dispatch.py): bounded least-loaded same-filesystem dispatch.
- [`py/src/replay/materialization_worker.py`](../../../py/src/replay/materialization_worker.py): target
  materialization, shard sealing/adoption, quarantine, and worker loop.
- [`py/src/replay/manager.py`](../../../py/src/replay/manager.py): supervision, append/flush, replay snapshot lock,
  rejection alarm, directory recovery, and resignation observation.
- [`py/src/replay/store.py`](../../../py/src/replay/store.py): columnar circular mmap and append transaction identity.
- [`py/src/replay/batch_loader.py`](../../../py/src/replay/batch_loader.py): deterministic global sampling, per-rank
  slicing, augmentation, dense target reconstruction, pinned prefetch, and device transfer.
- [`py/src/training/trainer/group.py`](../../../py/src/training/trainer/group.py): trainer process group, typed commands,
  response validation, and group failure.
- [`py/src/training/trainer/rank.py`](../../../py/src/training/trainer/rank.py): DDP rank runtime, quantum training, QAT,
  barriers, and rank-zero checkpoint save.
- [`py/src/training/checkpoint/persistence.py`](../../../py/src/training/checkpoint/persistence.py) and
  [`contracts.py`](../../../py/src/training/checkpoint/contracts.py): payload/manifest publication and checkpoint
  identity.
- [`py/src/self_play/native_configuration.py`](../../../py/src/self_play/native_configuration.py) and
  [`py/tools/publish_tensorrt_engine.py`](../../../py/tools/publish_tensorrt_engine.py): deployment engine refit,
  caching, and fidelity verification.

### Evaluation and operational evidence

- [`py/src/evaluation/manager.py`](../../../py/src/evaluation/manager.py): cadence, process scheduling, deadlines,
  persisted state, ladder aggregation, and retention references.
- [`py/src/evaluation/process.py`](../../../py/src/evaluation/process.py): job process, external-engine ownership,
  typed results, tracebacks, and failures.
- [`py/src/evaluation/match.py`](../../../py/src/evaluation/match.py): paired openings, shared candidate batching,
  native searched play, policy-only selector, and opponent routing.
- [`py/src/experiment/run.py`](../../../py/src/experiment/run.py): environment validation, initial checkpoint, fidelity
  probes, and run manifest.
- [`py/src/experiment/training_startup.py`](../../../py/src/experiment/training_startup.py): resolved configuration and
  startup artifacts.
- [`py/src/training/checkpoint/retention.py`](../../../py/src/training/checkpoint/retention.py): full/inference artifact
  retention.
- [`deployment/run_control.sh`](../../../deployment/run_control.sh): supported launch, stop, status, preservation, and
  fetch boundary.

## Uncertainties and items needing confirmation before drawing

These are narrow figure-audit questions, not missing architecture areas:

1. **Public terminology for model advancement.** The implementation consistently uses `generation` for one optimizer
   quantum and checkpoint step. The report should decide whether the figure says `training quantum`, `checkpoint`, or
   `generation`; it should not introduce run labels.
2. **TensorRT derivation detail.** The implementation establishes on-demand template refit and fidelity verification.
   The final archived artifacts still need to confirm the exact precision/template identity used by the largest model
   before the caption names it more specifically than `TensorRT engine`.
3. **Refresh acknowledgement wording.** Later self-play refreshes acknowledge the staged identity before performing
   the actual native load, although activation is guaranteed before the next batch. The figure must say `accepted`
   or `staged`, not `loaded`, for that response.
4. **Pause visualization.** `request_pause` is asynchronous and does not form a worker barrier before training starts.
   The diagram should depict a pause request, not imply all selected workers have already stopped when DDP begins.
5. **Evaluation artifact specialization.** Evaluation may specialize an ONNX artifact for its fixed batch size before
   TensorRT refit. This is a deployment detail inside an evaluation job, not a second checkpoint publication path.
6. **Operational archive contents.** The evidence boundary is clear, but the exact terminal bundle should be checked
   after the final preservation step before listing every included plot or raw match artifact in the published
   caption.

No further source discovery is required to begin the SVG. The remaining work is editorial selection: how much of the
recovery lane and progressive-candidate lane remains legible at publication scale.
