#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 6 ]]; then
    echo "Usage: sgd_replay_screen_arm.sh ARM GPU_0 GPU_1 OUTPUT LAYERS HIDDEN_SIZE" >&2
    exit 2
fi

arm="$1"
gpu_0="$2"
gpu_1="$3"
output="$4"
layers="$5"
hidden_size="$6"
repository="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
virtual_environment="/workspace/alphazero-engine-venv"
replay_root="/workspace/alphazero-engine-v34-lr-001/py/training_data/production/vast-chess-8gpu-integrated-v34"
python_nvidia_library_path="$(
    "${virtual_environment}/bin/python" -c \
        'from pathlib import Path; import site; root = Path(site.getsitepackages()[0]) / "nvidia"; print(":".join(str(path) for path in sorted(root.glob("*/lib")) if path.is_dir()))'
)"
if [[ -n "${python_nvidia_library_path}" ]]; then
    export LD_LIBRARY_PATH="${python_nvidia_library_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
export PYTHONPATH="${repository}/py"
export OMP_NUM_THREADS=1
cd "${repository}/py"

exec "${virtual_environment}/bin/torchrun" --standalone --nproc-per-node=2 \
    -m tools.run_sgd_replay_screen \
    --arm "${arm}" \
    --replay-store "${replay_root}/replay.bin" \
    --replay-experiment "${repository}/py/configs/production/vast-chess-8gpu-progressive-v39-int8.yaml" \
    --replay-sha256 d668a18e07be099538a79a70b2c65c275a724b7dba56eddeb5629d0a7df52a83 \
    --output "${output}" \
    --gpu-ids "${gpu_0}" "${gpu_1}" \
    --random-seed 20260913 \
    --layers "${layers}" \
    --hidden-size "${hidden_size}" \
    --time-budget-seconds 1200 \
    --maximum-optimizer-steps 12000 \
    --held-out-positions 4096 \
    --holdout-fraction 0.02
