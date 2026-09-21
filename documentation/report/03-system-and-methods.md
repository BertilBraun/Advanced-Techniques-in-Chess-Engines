# 3. System and training method

## Runtime boundary

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
Training proceeds in blocking optimizer quanta while a configured subset of self-play workers remains active. This
overlap turns many nominal self-play savings into slack rather than wall-clock savings, a central result of the
adaptive-stopping study.

The primary objective combines policy cross-entropy and WDL/value loss. Outcome value is discounted by ply, and a
small scheduled blend introduces search-root value later in training. Two training-only auxiliary targets are
retained: the next position's search policy and normalized remaining game length. Their purpose is representation
learning; they are not served during search.

## Progressive models and publication

Training begins with a smaller network to exploit its higher early self-play throughput. Larger candidates are grown
from the current model and trained beside it. Candidate starts are triggered by an Elo-improvement plateau, and
promotion requires loss catch-up under a shared objective. A promotion publishes the new model without changing
replay identity or generation accounting. Persistence includes both candidate and active state so restart does not
silently repeat or skip transitions. The full contract is in
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
