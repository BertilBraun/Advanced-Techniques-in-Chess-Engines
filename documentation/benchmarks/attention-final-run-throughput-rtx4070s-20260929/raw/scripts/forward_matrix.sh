#!/usr/bin/env bash
# Forward-only matrix: float16, FP8 and weight-product INT8 engines per architecture at batch 320, one GPU each.
set -uo pipefail
repository=/workspace/alphazero-engine
python=/workspace/alphazero-engine-venv/bin/python
matrix=/workspace/matrix
export PATH=/workspace/alphazero-engine-venv/bin:${PATH}
cd "${repository}/py"
mkdir -p "${matrix}/results" "${matrix}/logs"

"${python}" - > "${matrix}/dataset.txt" 2> "${matrix}/logs/dataset.log" <<'EOF'
import subprocess
from pathlib import Path
from src.evaluation.preparation import prepare_evaluation_artifacts
from src.experiment.configuration import load_experiment_configuration
from src.games.composition import create_game_implementation
root = Path('/workspace/alphazero-engine')
configuration = load_experiment_configuration(Path('configs/production/vast-chess-8gpu-final-attention.yaml'))
revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
artifacts = prepare_evaluation_artifacts(configuration, create_game_implementation(configuration), root, revision)
print(artifacts.dataset_path)
EOF
dataset=$(tail -n 1 "${matrix}/dataset.txt")
echo "DATASET ${dataset}"
[[ -f "${dataset}" ]] || { echo "DATASET_MISSING"; exit 1; }

PYTHONPATH=. "${python}" "${matrix}/export_matrix_models.py" "${matrix}/models" > "${matrix}/logs/export.log" 2>&1 || {
    echo "EXPORT_FAILED"; tail -n 20 "${matrix}/logs/export.log"; exit 1;
}
cat "${matrix}/logs/export.log" | grep parameters

gpu=0
for name in attention-8x160 attention-10x192 attention-12x192 attention-10x224 cnn-14x160 trained-10x192; do
    (
        started=$(date +%s)
        CUDA_VISIBLE_DEVICES=${gpu} PYTHONPATH=. "${python}" tools/compare_tensorrt_low_precision.py \
            --inference-model "${matrix}/models/${name}.pt" --dataset "${dataset}" \
            --precision fp8 --precision int8 \
            --fidelity-position-count 196 --calibration-position-count 320 \
            --artifact-directory "${matrix}/engines/${name}" --output "${matrix}/results/${name}.json" \
            --gpu-id 0 --acknowledge-gpu-load > "${matrix}/logs/${name}.log" 2>&1
        echo "MATRIX ${name} EXIT $? after $(( $(date +%s) - started )) s"
    ) &
    gpu=$((gpu + 1))
done
wait
echo FORWARD_MATRIX_DONE
