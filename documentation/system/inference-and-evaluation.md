# Inference and evaluation

## Inference boundary

The training checkpoint and deployment artifact are deliberately different. Training retains optimizer state,
quantization state, and auxiliary heads. Deployment exposes only policy logits and normalized WDL probabilities to
the native runtime. `GameImplementation` resolves a checkpoint into the backend-specific deployment path and the
C++ inference pipeline validates legal actions and WDL probabilities before search uses them.

The shared native pipeline in
[`InferencePipeline.hpp`](../../cpp/src/search/InferencePipeline.hpp) owns preallocated input/output slots, a
dedicated inference thread, optional CUDA streams and CUDA graphs for TorchScript, legal-policy normalization, and
model refresh. Search workers can keep up to two batches outstanding; submission and consumption use explicit slot
states rather than allocating a new batch object for every inference call.

## TensorRT publication

For the final recipe, a checkpoint initially contains a fixed-batch ONNX deployment graph. The resolver selects a
TensorRT template by model ID, precision, and QAT phase, then invokes
[`tools/publish_tensorrt_engine.py`](../../py/tools/publish_tensorrt_engine.py) to refit and publish a checkpoint-
specific engine. Self-play and evaluation use different template batch sizes. Evaluation specializes the ONNX graph
down to its configured batch size before refitting.

Generation zero uses the configured TorchScript bootstrap because no trained INT8 artifact exists yet. From
generation one, self-play uses the INT8 TensorRT path. The native TensorRT owner is
[`TensorRtInferenceModel.cpp`](../../cpp/src/search/TensorRtInferenceModel.cpp).

Run preparation freezes real encoded probe positions. Publication compares the TensorRT engine against its source
artifact on those probes and records policy top-one agreement, policy KL, and WDL error. The final run allows a
fidelity deviation to warn rather than terminate, because this is the explicitly selected INT8 deployment recipe;
the warning is evidence and must not be described as a passed fidelity gate. TensorRT template engines are also
part of the environment contract: a structurally compatible but stale template can alter outputs, as documented in
[`int8-template-staleness-rtx4070super-20260921`](../benchmarks/int8-template-staleness-rtx4070super-20260921/README.md).

## Scheduled evaluation

Evaluation is owned by the coordinator but executed in independent short-lived child processes. The manager in
[`evaluation/manager.py`](../../py/src/evaluation/manager.py) uses persisted active-run time, not generation, as its
schedule. Every 1,200 seconds it selects the newest checkpoint that had been completely published at that boundary,
resolves due jobs, assigns devices through the configured eight-GPU cycle, and runs up to 16 jobs concurrently. A
job's 30-minute timeout starts when the process launches. Failure becomes a typed durable result and does not stop
training.

The final configuration schedules two Stockfish adaptive-node definitions from generation one:

| Definition | Project model | Stockfish ladder | Games at each rung |
| --- | --- | --- | --- |
| `stockfish-searched` | fixed 64-search MCTS | 30, 100, 300, 1,000, 2,000, 3,000, 5,000, 10,000 nodes | 50 paired openings / 100 games |
| `stockfish-policy-only` | one-search root expansion | same ladder | 50 paired openings / 100 games |

The policy-only definition uses one native search so the policy is evaluated through the same legal-action and
deployment boundary; it is effectively the network policy after root expansion, not a 64-search player. Evaluation
disables root noise and forced playouts and derives the AlphaZero exploration constant from the fixed search budget.

Adaptive evaluation remembers a selected Stockfish rung. It uses score thresholds of `0.3` and `0.7` to retreat or
advance and evaluates a three-rung bracket rather than fitting Elo from one isolated score. The primary progressive-
sizing signal is the searched ladder. Exact ladder fitting and persisted selection are in
[`evaluation/ladder.py`](../../py/src/evaluation/ladder.py).

## Match protocol and immutable inputs

The opening suite is a checked-in selection of 50 eight-ply chess-book lines, materialized as an immutable artifact
with its selection hash. Each opening is played twice with colors swapped. Results retain raw games, candidate side,
termination, duration, W/D/L, score, player-order scores, and paired-bootstrap uncertainty. The maximum evaluation
length is 300 plies.

The configured fixed dataset is an immutable Stockfish-labelled binary artifact and is also used as the QAT
calibration source. Although the final scheduled definitions are the two match ladders, dataset preparation and
validation remain part of run preparation and reproducibility. The shared contracts and execution paths are
[`evaluation/configuration.py`](../../py/src/evaluation/configuration.py),
[`evaluation/match.py`](../../py/src/evaluation/match.py), and
[`evaluation/preparation.py`](../../py/src/evaluation/preparation.py).

The configured external engine record labels policies with Stockfish at 10,000 nodes, uses 1,000 nodes for its
default non-adaptive match setting, one thread, 1,024 MiB hash, MultiPV 8, and policy temperature `0.15`. The two
adaptive definitions explicitly use `engines/stockfish-13`. These identities must be preserved with the resolved
configuration and installed-engine evidence; an engine name without the binary hash is not a reproducible opponent.

## Terminal evaluation and pending result

The elapsed ladder is training-time instrumentation, not the final strength claim. After training stops and all
GPUs are available, the terminal tools select Stockfish opponents and run larger fixed matches for policy-only and
searched budgets. The historical v34 commands are documented in
[`py/README.md`](../../py/README.md); the final protocol
must be frozen with the selected result rather than inferred from that example.

Final checkpoint selection, training volume, duration, cost, terminal W/D/L, confidence intervals, Elo calibration,
and high-search measurements are intentionally not stated here until their evidence is archived. Their single
authoritative landing page is [`final-chess-run.md`](../results/final-chess-run.md); the root README and report should
consume the frozen values from there.
