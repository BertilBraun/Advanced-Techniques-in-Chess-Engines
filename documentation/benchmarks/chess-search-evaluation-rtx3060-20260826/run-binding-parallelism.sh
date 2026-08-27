#!/usr/bin/env bash
# Parallelism and virtual loss measured where they bind: batch 1024 against ~100 active trees.
# Concurrency equals the arm count so every arm runs under identical GPU contention for its whole
# life; per-arm wall-clock from an unevenly drained pool is not a valid throughput comparison.
set -euo pipefail

source /workspace/search-eval/env.sh

OUTPUT=${SEARCH_EVAL_ROOT}/output/binding-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "${OUTPUT}"
ARMS=${SEARCH_EVAL_SOURCE}/py/configs/evaluation/chess-search-arms-binding-v1.json
VISITS=600
OPENING_PAIRS=${OPENING_PAIRS:-100}

echo "output: ${OUTPUT}"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "${OUTPUT}/gpu-tenants-at-start.txt" || true

RUNG=3500
echo "pinned rung:  nodes (from the overnight calibration)"
echo "" > "/chosen-rung.txt"

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
    --opening-selection-seed 20260826 \
    --match-random-seed 20260827 \
    --device 0 \
    --concurrency 3 \
    --inference-batch-size 1024 \
    --output-directory "${OUTPUT}/arm-matrix" \
    2>&1 | tee "${OUTPUT}/arms.log"

echo "=== complete: ${OUTPUT} ==="
