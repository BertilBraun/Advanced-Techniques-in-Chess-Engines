#!/usr/bin/env bash
# The four float16 refit templates the attention configuration names; publications refit them with real weights.
set -uo pipefail
cd /workspace/alphazero-engine/py
mkdir -p /workspace/tensorrt /workspace/matrix/logs
build() {
    local gpu=$1 size=$2 batch=$3
    CUDA_VISIBLE_DEVICES=${gpu} PYTHONPATH=. /workspace/alphazero-engine-venv/bin/python tools/build_tensorrt_refit_template.py \
        --model "/workspace/matrix/models/attention-${size}.pt" \
        --output "/workspace/tensorrt/lc0-attention-${size}-fp16-b${batch}.engine" --batch-size "${batch}" \
        > "/workspace/matrix/logs/production-template-${size}-b${batch}.log" 2>&1
    echo "PRODUCTION_TEMPLATE ${size} b${batch} EXIT $?"
}
(build 5 8x160 320; build 5 8x160 64) &
(build 7 10x192 320; build 7 10x192 64) &
wait
ls -la /workspace/tensorrt/
echo PRODUCTION_TEMPLATES_DONE
