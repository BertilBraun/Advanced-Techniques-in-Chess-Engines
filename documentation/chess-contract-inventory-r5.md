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
| Input encoding is the 8x8, 29-channel chess tensor with compressed binary and scalar planes. | `cpp/src/ChessGameContract.hpp` | `cpp/src/binding.cpp`, `cpp/src/MCTS/DirectSelfPlaySearch.cpp`, `cpp/src/NonCachingInferenceClient.cpp`, `cpp/test/TestChessGameContract.cpp` |
| Representation dimensions are fixed for chess and must remain consistent with move/action counts. | `cpp/src/ChessGameContract.hpp` | `cpp/test/TestChessGameContract.cpp` |

## Python

| Assumption | Canonical owner | Converted call sites |
| --- | --- | --- |
| Chess action-space size, representation dimensions, canonical board encoding, and move encode/decode are owned by one chess state contract. | `py/src/games/chess/contract.py` | `py/src/Encoding.py`, `py/src/Network.py`, `py/src/cluster/InferenceClient.py`, `py/src/cluster/NonCachingInferenceClient.py`, `py/src/self_play/SelfPlay.py`, `py/src/train/ChessReplay.py`, `py/src/eval/ModelEvaluation.py` |
| The default research runtime is a chess experiment resolved from one YAML file instead of static Python constants. | `py/src/experiment/chess_experiment.py` + `py/configs/chess-default-experiment.yaml` | `py/src/settings.py`, `py/src/games/chess/ChessSettings.py`, `py/train.py` |
| Trainer, self-play, evaluation, and host resource topology are separate typed subtrees used directly by their component builders. | `py/src/experiment/chess_experiment.py` | `py/src/settings.py`, `py/train.py` |
| Network architecture has one canonical immutable `NetworkParams` type at configuration and runtime boundaries. | `py/src/train/TrainingArgs.py` | `py/src/experiment/chess_experiment.py`, `py/src/Network.py` |
| Chess network shape, self-play temperature schedule, self-play root noise, and retained-tree discounting are chess experiment settings. | `py/src/experiment/chess_experiment.py` | `py/src/settings.py`, `py/train.py` |
| Evaluation match budgets and scheduling are shared settings; chess openings, fixed dataset, model ladder, random opponent, and Stockfish settings are chess-owned. | `py/src/experiment/chess_experiment.py` | `py/src/eval/ModelEvaluation.py`, `py/src/cluster/EvaluationProcess.py` |
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

## Remaining R5 integration

The paid-run manifest preparer still owns the historical JSON `RunConfiguration`
schema. The YAML entry point must persist and approve the complete resolved
`ChessExperimentConfiguration` directly before it can replace that path. This
requires changing manifest and publication provenance ownership; projecting the
hierarchical experiment back into the flat schema is intentionally avoided.
