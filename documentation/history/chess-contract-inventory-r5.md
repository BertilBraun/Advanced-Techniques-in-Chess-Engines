# R5 Chess Contract Inventory

This inventory records the concrete chess-specific assumptions traced during R5,
their canonical owner after extraction, and the converted call sites.

## Native C++

| Assumption | Canonical owner | Converted call sites |
| --- | --- | --- |
| Chess position lifecycle starts from the standard initial board and may replay bounded UCI history. | `cpp/src/ChessGameContract.hpp` | `cpp/src/binding.cpp`, `cpp/test/TestChessGameContract.cpp` |
| Chess child transitions clone immutable search positions and apply one legal move. | `cpp/src/ChessGameContract.hpp` | `cpp/test/TestChessGameContract.cpp` |
| Terminal detection and terminal value use chess checkmate/stalemate semantics from `Board`. | `cpp/src/ChessGameContract.hpp` | `cpp/src/MCTS/DirectSelfPlaySearch.cpp`, `cpp/src/MCTS/MCTS.cpp`, `cpp/test/TestChessGameContract.cpp` |
| Action mapping is the dense chess move encoding already implemented by `MoveEncoding`. | `cpp/src/ChessGameContract.hpp` | `cpp/src/binding.cpp`, `cpp/src/MCTS/DirectSelfPlaySearch.cpp`, `cpp/src/MCTS/MCTS.cpp`, `cpp/test/TestChessGameContract.cpp` |
| Input encoding is the 8x8, 29-channel chess tensor with compressed binary and scalar planes. | `cpp/src/ChessGameContract.hpp` | `cpp/src/binding.cpp`, `cpp/src/MCTS/DirectSelfPlaySearch.cpp`, `cpp/src/InferenceClient.cpp`, `cpp/test/TestChessGameContract.cpp` |
| Representation dimensions are fixed for chess and must remain consistent with move/action counts. | `cpp/src/ChessGameContract.hpp` | `cpp/test/TestChessGameContract.cpp` |

## Python

| Assumption | Canonical owner | Converted call sites |
| --- | --- | --- |
| Chess action-space size, representation dimensions, canonical board encoding, and move encode/decode are owned by one chess state contract. | `py/src/games/chess/contract.py` | `py/src/Encoding.py`, `py/src/Network.py`, `py/src/cluster/InferenceClient.py`, `py/src/self_play/SelfPlay.py`, `py/src/train/ChessReplay.py`, `py/src/eval/ModelEvaluation.py` |
| Each research runtime resolves an explicit chess or Go experiment file instead of installing process-global game settings. | `py/src/experiment/configuration.py` + `py/configs/*-experiment.yaml` | `py/train.py` |
| Shared network, self-play, trainer, topology, lifecycle, limits, retention, and scheduling settings form one canonical immutable `TrainingArgs` hierarchy of validated `FrozenModel` types embedded directly in the experiment. | `py/src/train/TrainingArgs.py` | `py/src/experiment/configuration.py`, `py/train.py`, cluster and trainer entry points |
| Network architecture has one canonical immutable `NetworkParams` type at configuration and runtime boundaries, with explicit game dimensions at every model boundary. | `py/src/train/TrainingArgs.py` | `py/src/experiment/configuration.py`, `py/src/network.py` |
| Chess network shape, self-play temperature schedule, self-play root noise, and retained-tree discounting are shared runtime component settings selected by the chess experiment. | `py/src/train/TrainingArgs.py` | `py/src/settings.py`, `py/train.py` |
| Evaluation scheduling is shared; chess match budgets, openings, fixed dataset, model ladder, random opponent, and Stockfish settings are passed separately as chess-owned evaluation configuration. | `py/src/experiment/configuration.py` | `py/src/eval/ModelEvaluation.py`, `py/src/cluster/EvaluationProcess.py`, `py/src/cluster/CreditEvaluationScheduler.py` |
| Run approval, model-zero preparation, resolved manifests, and publication provenance bind the complete experiment rather than a projected legacy schema. | `py/src/experiment/run.py` | `py/train.py`, `py/src/train/CreditPublication.py`, `py/src/cluster/EvaluationProcess.py` |
| Creating the fixed chess evaluation dataset is a chess-owned operation rather than a static-settings side effect. | `py/src/games/chess/evaluation_dataset.py` | `py/tools/prepare_chess_evaluation_dataset.py` |
| Compact chess replay materialization and batch construction still use chess-specific data structures and are intentionally not generalized yet. | `py/src/train/ChessReplay.py` | Existing replay maintainer and trainer call sites |

## Deferred duplication

R5 intentionally preserves the following duplicated chess logic across Python and
C++ because both implementations are current production owners and Go is not
yet wired in:

- dense chess move/action encoding exists in both `py/src/games/chess/ChessGame.py` and `cpp/src/MoveEncoding.cpp`;
- canonical chess tensor encoding exists in both `py/src/games/chess/ChessGame.py` and `cpp/src/BoardEncoding.cpp`;
- terminal result scoring exists in both `py/src/Encoding.py` and `cpp/src/BoardEncoding.cpp`;
- bounded-history replay for repetition-aware roots exists in both Python orchestration and native bindings.

These were recorded for later review instead of being deduplicated during R5.

## Configuration migration

The historical JSON `RunConfiguration`, its mutable `TrainingArgs` adapter, and
the converted static training defaults were removed. Training, evaluation,
dataset preparation, benchmarks, approval, resolved manifests, and publication
provenance now consume the YAML experiment and its canonical component types
directly. Tool-specific overrides create immutable replacements and do not
mutate shared settings.

The configuration audit removed unused worker count, initial-evaluation,
top-level evaluation-concurrency, disconnected general artifact-retention, and
inference-cache settings. Chess self-play and MCTS evaluation use the measured
direct prepared-batch inference path. The retired Python and C++ cached clients,
cache containers, bindings, statistics, diagnostics, and cache-only benchmarks
were removed; the queued fallback performs no caching.
