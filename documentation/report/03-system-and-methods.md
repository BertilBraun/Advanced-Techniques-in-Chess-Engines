# 3. System and training method

## Runtime boundary

The systems design matters here for one reason: AlphaZero training needs enough searched games and fresh positions to
learn within the available wall-clock budget. The report therefore describes the throughput-critical path—native
search, batching, inference, replay, and training ownership—while leaving topology sweeps and recovery machinery to
the linked engineering record.

![Python coordination, native self-play, batched TensorRT inference, replay, training, and evaluation feedback](figures/learning-loop.svg)

Figure 3.1. A published model returns to batched native self-play; searched positions become replay targets for the
trainer. Paired evaluation measures the published checkpoint and feeds selection decisions back to Python. The
diagram shows ownership and data flow, not the number or scheduling of every worker.

The system has one native implementation of the latency-sensitive path and one typed orchestration layer:

- [`cpp/`](../../cpp/README.md) owns chess and Go state, legal actions, encodings, batched inference, MCTS, self-play,
  and interactive analysis;
- [`py/`](../../py/README.md) owns validated experiment configuration, worker lifecycle, replay ingestion,
  distributed training, checkpoint publication, evaluation, telemetry, and tools.

This division avoids maintaining a second production search or rules implementation in Python. Cross-language
contracts—action mapping, tensor shape, feature planes, symmetry, and output order—are tested explicitly. The
[Python runtime architecture](../architecture/python-runtime-rework.md) is the detailed ownership record.

## Chess representation and outputs

The network receives a board-aligned tensor containing current state, history, and additional chess features. The
action space contains 1,880 reduced chess actions. Game-defined symmetry transforms the state and every action-space
target together. The production inference artifact exports only the primary policy and WDL/value outputs; auxiliary
training heads are stripped from serving artifacts.

The final architecture family is convolutional. Residual blocks use scaled post-activation with capped activations,
and global-pooling context is inserted every second block. A from-to attention policy head scores move origin and
destination structure far more compactly than the earlier dense spatial projection. The value path uses two value
channels and a small fully connected layer. Exact stage shapes are listed in
[the final recipe](06-final-chess-recipe.md).

## Search and self-play

Self-play uses PUCT-style Monte Carlo tree search with neural policy priors and WDL/value estimates. The final recipe
uses fixed per-generation visit budgets rather than per-position adaptive allocation. Root exploration uses
Dirichlet noise, reduced-parent-value first-play urgency, and forced playouts. A fraction of root visits is retained
when advancing the tree.

Thousands of games are interleaved so many small inference requests become large GPU batches. Native workers own
the trees; one inference worker per process submits batched work to TensorRT. Search, batching, and worker topology
must be interpreted together: raising parallel simulations can improve device fill while changing search semantics,
and excessive in-flight games can improve aggregate throughput while increasing individual-game latency.

Starting positions combine shallow random legal openings with archived restart states. Restart candidates are drawn
from uncertain, consequential positions rather than uniformly from all history. This targets useful diversity while
retaining a defined fraction of standard-game evolution. Calibrated resignation saves completed-game work only after
its false-nonloss evidence gate is satisfied; continuation games preserve auditing and terminal training examples.

## Replay and materialization

Native workers publish completed trajectories atomically. Parallel materializers reconstruct observations and write
fixed-layout columnar shards. A single circular memory-mapped store is the canonical training replay. Sparse policies
retain a bounded number of action entries, while dense batches are reconstructed only at the training boundary.

The layout stores the state, legal moves, primary search policy, WDL target, root value, sample weight, source
generation, surprise information, and configured auxiliary targets. Append and restart behavior are designed around
exactly-once claims and explicit rejection telemetry. See
[Columnar replay and shard ingestion](../architecture/replay-pipeline-rework.md) and the later
[materialization rework](../architecture/replay-materialization-rework.md).

Sampling mixes uniform probability with policy surprise, emphasizing positions where search disagreed with the
network prior while preserving broad coverage. Capacity grows in stages as the run matures. Credit accounting ties
optimizer work to materialized samples through an explicit replay ratio instead of allowing the learner to outrun
data generation silently.

## Training

The final trainer uses eight persistent NCCL ranks, one per GPU, with global batch 2,048 and bfloat16 training.
Training proceeds in blocking optimizer quanta while a configured subset of self-play workers remains active. A
*generation* is the approachable public term for one training-and-publication cycle; a *checkpoint* is the durable
model artifact written at such a boundary. This overlap turns many nominal self-play savings into slack rather than
wall-clock savings, a central result of the adaptive-stopping study.

The primary objective combines policy cross-entropy and WDL/value loss. Outcome value is discounted by ply, and a
small scheduled blend introduces search-root value later in training. Two training-only auxiliary targets are
retained: the next position's search policy and normalized remaining game length. Their purpose is representation
learning; they are not served during search.

## Progressive models and publication

Training begins with a smaller network to exploit its higher early self-play throughput. In the production
controller, the immediate larger candidate starts from an independent initialization and trains beside the active
model. Candidate starts are triggered by an Elo-improvement plateau, and promotion requires two qualifying
head-to-head matches. A separate final investigation tested function-preserving growth from the trained medium model;
it is not the controller's ordinary initialization path. A promotion publishes the new model without changing replay
identity or generation accounting. The full contract is in
[Progressive model sizing](../architecture/progressive-model-sizing.md).

Rank zero publishes a complete training checkpoint and a trimmed inference artifact. In the final path, QAT-aware
ONNX artifacts are refit into TensorRT templates for self-play and evaluation. Numerical fidelity is a publication
condition and is discussed in [Chapter 5](05-systems-optimization.md).

## Evaluation

Short-lived evaluation jobs run independently of training on a fixed cadence. They include fixed-dataset metrics and
paired-opening Stockfish ladders for policy-only and searched play. The final evaluation is a separate terminal
protocol with larger matches and deeper budgets. Engine binaries, datasets, opening books, and their hashes are
configuration-owned; [evaluation engine documentation](../operations/evaluation-engines.md) records installation and
identity rules.

Evaluation itself became an engineered subsystem. Adaptive ladders bracket the candidate against adjacent Stockfish
node limits rather than extrapolating from one score; paired openings reverse colors; concurrent jobs are
device-cycled; and reports retain raw W/D/L and exact search identity. Batch shape can alter serving behavior and
reshuffle outcomes, so it is part of the protocol rather than an invisible speed setting. The
[ladder-batching study](../benchmarks/ladder-batching-rtx4070s-20260906/README.md),
[strength-over-generation series](../benchmarks/ladder-elo-vs-generation-rtx4070s-20260906/README.md), and
[deep generation-936 match](../benchmarks/deep-match-generation936-50k-nodes-rtx4070s-20260906/README.md) show the
progression from frequent noisy signal to terminal-strength measurement.

Detailed restart and recovery semantics are implementation concerns rather than a report contribution. The public
reproducibility boundary is simply that completed trajectories, replay state, checkpoints, and evaluation evidence
are durably committed before they are credited or reported; the linked architecture documentation owns the mechanics.
