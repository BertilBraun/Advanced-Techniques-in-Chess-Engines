#!/usr/bin/env bash
# Native self-play at the live settings; the benchmark takes the float16 ONNX publication form of each model.
set -uo pipefail
repository=/workspace/alphazero-engine
python=/workspace/alphazero-engine-venv/bin/python
matrix=/workspace/matrix
export PATH=/workspace/alphazero-engine-venv/bin:${PATH}
cd "${repository}/py"
names="attention-8x160 attention-10x192 attention-12x192 attention-10x224 cnn-14x160"
PYTHONPATH=. "${python}" - ${names} <<'PYTHON'
import sys
from pathlib import Path
from tools.publish_tensorrt_engine import export_onnx
for name in sys.argv[1:]:
    export_onnx(Path(f'/workspace/matrix/models/{name}.pt'), Path(f'/workspace/matrix/models/{name}.fp16.onnx'), (320, 52, 8, 8))
    print('ONNX', name)
PYTHON
rm -rf "${matrix}/self-play"
gpu=0
for name in ${names}; do
    (
        CUDA_VISIBLE_DEVICES=${gpu} PYTHON_BINARY="${python}" BENCHMARK_OUTPUT_ROOT="${matrix}/self-play/${name}" \
            GPU_COUNT=1 PROCESSES_PER_GPU=4 PARALLEL_GAMES_PER_PROCESS=512 MEASUREMENT_DURATION_SECONDS=60 \
            BENCHMARK_GENERATION=1026 INFERENCE_BACKEND=tensorrt \
            TENSORRT_TEMPLATE_ENGINE="${matrix}/templates/${name}-b320.engine" \
            bash tools/run_self_play_search_benchmark.sh "${matrix}/models/${name}.fp16.onnx" \
            configs/production/vast-chess-8gpu-final-attention.yaml "${repository}" \
            > "${matrix}/logs/self-play-${name}.log" 2>&1
        echo "SELF_PLAY ${name} EXIT $?"
    ) &
    gpu=$((gpu + 1))
done
wait
echo SELF_PLAY_MATRIX_DONE
