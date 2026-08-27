#!/usr/bin/env bash
# Does parallelism stay strength-neutral at the 150-visit fast-search budget?
# Concurrency equals the arm count so every arm runs under identical GPU contention for its whole
# life; per-arm wall-clock from an unevenly drained pool is not a valid throughput comparison.
set -euo pipefail

source /workspace/search-eval/env.sh

OUTPUT=${SEARCH_EVAL_ROOT}/output/lowvisit-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "${OUTPUT}"
ARMS=${SEARCH_EVAL_SOURCE}/py/configs/evaluation/chess-search-arms-lowvisits-v1.json
VISITS=150
OPENING_PAIRS=${OPENING_PAIRS:-100}

echo "output: ${OUTPUT}"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "${OUTPUT}/gpu-tenants-at-start.txt" || true

# The 600-visit rung of 3500 nodes would floor a 150-visit engine, so the rung is recalibrated here.
echo "=== calibration at ${VISITS} visits ==="
"${SEARCH_EVAL_PYTHON}" -m tools.run_stockfish_ladder \
    --experiment "${SEARCH_EVAL_EXPERIMENT}" \
    --run-directory "${SEARCH_EVAL_RUN_STATE}" \
    --checkpoint-generation "${SEARCH_EVAL_GENERATION}" \
    --opening-manifest "${SEARCH_EVAL_OPENINGS}" \
    --stockfish-executable "${SEARCH_EVAL_STOCKFISH}" \
    --stockfish-node-ladder 100 300 600 1000 2100 \
    --probe-games 10 \
    --opening-selection-seed 20260827 \
    --match-random-seed 20260828 \
    --devices 0 \
    --model-searches "${VISITS}" \
    --parallel-searches 4 \
    --exploration-constant 1.5 \
    --first-play-urgency reduced_parent_value \
    --first-play-urgency-reduction 0.2 \
    --virtual-loss-weight 1.0 \
    --search-value-discount-per-ply 0.99 \
    --inference-batch-size 64 \
    --output-directory "${OUTPUT}/calibration" \
    > "${OUTPUT}/calibration.log" 2>&1

RUNG=$("${SEARCH_EVAL_PYTHON}" -c "
import json
print(json.load(open('${OUTPUT}/calibration/ladder-result.json'))['closest_stockfish_nodes'])
")
echo "chosen rung: ${RUNG} nodes"
echo "${RUNG}" > "${OUTPUT}/chosen-rung.txt"

echo "=== arm matrix at ${VISITS} visits, rung ${RUNG} ==="
"${SEARCH_EVAL_PYTHON}" -m tools.run_search_arm_matrix \
    --matrix "${ARMS}" \
    --experiment "${SEARCH_EVAL_EXPERIMENT}" \
    --run-directory "${SEARCH_EVAL_RUN_STATE}" \
    --checkpoint-generation "${SEARCH_EVAL_GENERATION}" \
    --opening-manifest "${SEARCH_EVAL_OPENINGS}" \
    --stockfish-executable "${SEARCH_EVAL_STOCKFISH}" \
    --stockfish-nodes "${RUNG}" \
    --opening-pairs "${OPENING_PAIRS}" \
    --opening-selection-seed 20260827 \
    --match-random-seed 20260828 \
    --device 0 \
    --concurrency 6 \
    --inference-batch-size 64 \
    --output-directory "${OUTPUT}/arm-matrix" \
    2>&1 | tee "${OUTPUT}/arms.log"

echo "=== complete: ${OUTPUT} ==="
