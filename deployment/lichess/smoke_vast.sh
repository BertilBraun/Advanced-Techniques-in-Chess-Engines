#!/usr/bin/env bash
set -euo pipefail

engine_repository_root="${ENGINE_REPOSITORY_ROOT:-/workspace/alphazero-engine}"
virtual_environment_root="${ENGINE_VENV_ROOT:-/workspace/alphazero-venv}"
model_path="${MODEL_PATH:-}"

if [[ -z "${model_path}" ]]; then
    model_repository="${HF_MODEL_REPO:-BertilBraun/alphazero-chess}"
    model_file="${HF_MODEL_FILE:-latest.jit.pt}"
    model_revision="${HF_MODEL_REVISION:-}"
    download_arguments=(
        --repository "${model_repository}"
        --filename "${model_file}"
        --output-directory /data/models
        --provenance /data/model-source.txt
    )
    if [[ -n "${model_revision}" ]]; then
        download_arguments+=(--revision "${model_revision}")
    fi
    mkdir -p /data/models
    "${virtual_environment_root}/bin/python" "${engine_repository_root}/py/tools/fetch_hf_model.py" \
        "${download_arguments[@]}"
    model_path="/data/models/${model_file}"
fi

if [[ ! -f "${model_path}" ]]; then
    echo "Model does not exist: ${model_path}" >&2
    exit 1
fi

export ENGINE_PYTHON="${virtual_environment_root}/bin/python"
export ENGINE_REPOSITORY_ROOT="${engine_repository_root}"
export MODEL_PATH="${model_path}"

echo "CUDA device:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"${ENGINE_PYTHON}" -c \
    "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"

smoke_output="$(mktemp)"
trap 'rm -f "${smoke_output}"' EXIT
{
    echo uci
    echo isready
    echo ucinewgame
    echo "position startpos moves e2e4 e7e5"
    echo "go movetime 1000"
    sleep 3
    echo stop
    sleep 1
    echo quit
} | /usr/local/bin/alphazero-uci | tee "${smoke_output}"

"${ENGINE_PYTHON}" "${engine_repository_root}/py/tools/validate_uci_transcript.py" \
    --transcript "${smoke_output}" \
    --move e2e4 \
    --move e7e5
echo "UCI/CUDA/model smoke test passed with a legal bestmove."
