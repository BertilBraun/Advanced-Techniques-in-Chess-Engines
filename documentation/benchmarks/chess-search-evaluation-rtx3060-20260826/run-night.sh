#!/usr/bin/env bash
# Overnight chess search evaluation: Family B fidelity, then Family A calibration and arm matrix.
# Start only on an explicit instruction. Stops on the first failure; every stage is resumable by
# deleting its output directory and rerunning.
set -euo pipefail

source /workspace/search-eval/env.sh

OUTPUT=${SEARCH_EVAL_ROOT}/output/night-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "${OUTPUT}"
GRID=${SEARCH_EVAL_SOURCE}/py/configs/evaluation/chess-search-stopping-grid-v1.json
ARMS=${SEARCH_EVAL_SOURCE}/py/configs/evaluation/chess-search-arms-v1.json
POSITIONS=${SEARCH_EVAL_ROOT}/output/positions-g162-v1.json

OPENING_PAIRS=${OPENING_PAIRS:-100}
CONCURRENCY=${CONCURRENCY:-6}
FIDELITY_POSITIONS=${FIDELITY_POSITIONS:-3000}

echo "output: ${OUTPUT}"
echo "revision: ${ENGINE_SOURCE_REVISION}"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "${OUTPUT}/gpu-tenants-at-start.txt" || true

# Stage 1 — Family B. Serial descent, where the checkpoint-trace replay is exact.
echo "=== Family B: policy-target fidelity ==="
"${SEARCH_EVAL_PYTHON}" -m tools.measure_policy_target_fidelity \
    --configuration "${SEARCH_EVAL_EXPERIMENT}" \
    --model "${SEARCH_EVAL_RUN_STATE}/model_162.jit.pt" \
    --positions "${POSITIONS}" \
    --grid "${GRID}" \
    --output "${OUTPUT}/fidelity-g162-v1.json" \
    --generation "${SEARCH_EVAL_GENERATION}" \
    --device 0 \
    --reference-visits 10000 \
    --observation-interval 50 \
    --parallel-searches 1 \
    --chunk-positions 64 \
    --inference-batch-size 128 \
    --position-limit "${FIDELITY_POSITIONS}" \
    2>&1 | tee "${OUTPUT}/family-b.log"

# Stage 2 — locate the Stockfish rung nearest 50% at the baseline arm's settings.
echo "=== Family A: rung calibration ==="
"${SEARCH_EVAL_PYTHON}" -m tools.run_stockfish_ladder \
    --experiment "${SEARCH_EVAL_EXPERIMENT}" \
    --run-directory "${SEARCH_EVAL_RUN_STATE}" \
    --checkpoint-generation "${SEARCH_EVAL_GENERATION}" \
    --opening-manifest "${SEARCH_EVAL_OPENINGS}" \
    --stockfish-executable "${SEARCH_EVAL_STOCKFISH}" \
    --stockfish-node-ladder 1000 2100 3500 6500 11000 20000 \
    --probe-games 10 \
    --opening-selection-seed 20260826 \
    --match-random-seed 20260827 \
    --devices 0 \
    --model-searches 600 \
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
import json, sys
result = json.load(open('${OUTPUT}/calibration/ladder-result.json'))
print(result['closest_stockfish_nodes'])
")
echo "chosen Stockfish rung: ${RUNG} nodes"
echo "${RUNG}" > "${OUTPUT}/chosen-rung.txt"

# Stage 3 — every arm against that one rung, on the same openings, colours and seed.
echo "=== Family A: arm matrix (${OPENING_PAIRS} pairs, concurrency ${CONCURRENCY}) ==="
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
    --concurrency "${CONCURRENCY}" \
    --inference-batch-size 64 \
    --output-directory "${OUTPUT}/arm-matrix" \
    2>&1 | tee "${OUTPUT}/family-a.log"

nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "${OUTPUT}/gpu-tenants-at-end.txt" || true
echo "=== complete: ${OUTPUT} ==="
