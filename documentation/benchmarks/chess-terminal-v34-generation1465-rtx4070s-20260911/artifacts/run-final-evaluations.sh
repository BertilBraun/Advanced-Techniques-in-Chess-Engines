#!/bin/bash
set -euo pipefail
cd /workspace/alphazero-engine/py
export PYTHONPATH=/workspace/alphazero-engine/py
exec /workspace/alphazero-engine-venv/bin/python -m tools.run_stockfish_terminal_evaluations \
  --experiment /workspace/alphazero-engine-v34-lr-001/py/training_data/production/vast-chess-8gpu-integrated-v34/resolved-experiment.json \
  --run-directory /workspace/alphazero-engine-v34-lr-001/py/training_data/production/vast-chess-8gpu-integrated-v34 \
  --checkpoint-generation 1465 \
  --opening-manifest /workspace/alphazero-engine/py/reference/chess-elite-2025-11-balanced-4moves-200-v1-openings.json \
  --stockfish-executable /workspace/alphazero-engine/engines/stockfish-13 \
  --output-root /workspace/postrun/v34-terminal-g1465/final \
  --devices 0 1 2 3 4 5 6 7 \
  --policy-only-stockfish-nodes 1000 \
  --shallow-stockfish-nodes 5000 \
  --deep-stockfish-nodes 100000 \
  --very-deep-stockfish-nodes 50000
