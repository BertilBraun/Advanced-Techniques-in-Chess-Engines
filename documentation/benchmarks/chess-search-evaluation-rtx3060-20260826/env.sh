# Sourced by every search-evaluation command on the node.
set -euo pipefail
ulimit -Sn 65536
export SEARCH_EVAL_ROOT=/workspace/search-eval
export SEARCH_EVAL_SOURCE=$SEARCH_EVAL_ROOT/source
export SEARCH_EVAL_PYTHON=/workspace/alphazero-engine-venv/bin/python
export SEARCH_EVAL_STOCKFISH=/workspace/alphazero-engine/engines/stockfish-13
export SEARCH_EVAL_OPENINGS=/workspace/evaluation-artifacts/chess/chess-elite-2025-11-balanced-4moves-200-v1-openings.json
export SEARCH_EVAL_EXPERIMENT=$SEARCH_EVAL_SOURCE/py/configs/validation/vast-chess-4day-production-v9.yaml
export SEARCH_EVAL_RUN_STATE=$SEARCH_EVAL_ROOT/run-state
export SEARCH_EVAL_GENERATION=162
export ENGINE_SOURCE_REVISION=$(cd "$SEARCH_EVAL_SOURCE" && git rev-parse HEAD)
nvidia_library_path=$(find /workspace/alphazero-engine-venv/lib/python3.12/site-packages/nvidia -mindepth 2 -maxdepth 2 -type d -name lib -print | paste -sd:)
export LD_LIBRARY_PATH="${nvidia_library_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd "$SEARCH_EVAL_SOURCE/py"
