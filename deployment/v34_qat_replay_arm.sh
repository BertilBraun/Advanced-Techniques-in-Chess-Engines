#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: v34_qat_replay_arm.sh CELL GPU_ID OUTPUT" >&2
    exit 2
fi

cell="$1"
gpu_id="$2"
output="$3"
repository="${ENGINE_REPOSITORY_DIRECTORY:?ENGINE_REPOSITORY_DIRECTORY must identify the prepared worktree}"
virtual_environment="${ENGINE_VIRTUAL_ENVIRONMENT:-/workspace/alphazero-engine-venv}"
replay_root="${V34_QAT_REPLAY_ROOT:?V34_QAT_REPLAY_ROOT must identify the frozen production run directory}"
replay_experiment="${V34_QAT_REPLAY_EXPERIMENT:?V34_QAT_REPLAY_EXPERIMENT must identify the replay experiment YAML}"
replay_sha256="${V34_QAT_REPLAY_SHA256:?V34_QAT_REPLAY_SHA256 must identify the frozen replay store}"
initial_model="${V34_QAT_INITIAL_MODEL:?V34_QAT_INITIAL_MODEL must identify the common V34 state dictionary}"
steps="${V34_QAT_STEPS:-2000}"
learning_rate="${V34_QAT_LEARNING_RATE:-0.008}"
warmup_steps="${V34_QAT_WARMUP_STEPS:-200}"
final_fidelity_positions="${V34_QAT_FIDELITY_POSITIONS:-4096}"

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

arguments=(
    -m tools.run_int8_architecture_screen
    --replay-store "${replay_root}/replay.bin"
    --replay-experiment "${replay_experiment}"
    --replay-sha256 "${replay_sha256}"
    --output "${output}"
    --cell "${cell}"
    --random-seed 20260916
    --device-id "${gpu_id}"
    --steps "${steps}"
    --batch-size 1024
    --learning-rate "${learning_rate}"
    --warmup-steps "${warmup_steps}"
    --evaluate-every 500
    --final-fidelity-positions "${final_fidelity_positions}"
    --initial-model "${initial_model}"
)
if [[ "${cell}" == "post_qat" || "${cell}" == "post_shared_qat" ]]; then
    arguments+=(--fold-post-activation-batch-norm)
fi

exec "${virtual_environment}/bin/python" "${arguments[@]}"
