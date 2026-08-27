#!/usr/bin/env bash
# Where does parallelism begin to cost strength at the 600-visit full-search budget?
# Concurrency equals the arm count so every arm runs under identical GPU contention for its whole
# life; per-arm wall-clock from an unevenly drained pool is not a valid throughput comparison.
set -euo pipefail

source /workspace/search-eval/env.sh

OUTPUT=${SEARCH_EVAL_ROOT}/output/highparallel-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "${OUTPUT}"
ARMS=${SEARCH_EVAL_SOURCE}/py/configs/evaluation/chess-search-arms-highparallel-v1.json
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
    --concurrency 5 \
    --inference-batch-size 64 \
    --output-directory "${OUTPUT}/arm-matrix" \
    2>&1 | tee "${OUTPUT}/arms.log"

echo "=== complete: ${OUTPUT} ==="
