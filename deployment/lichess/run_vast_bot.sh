#!/usr/bin/env bash
set -euo pipefail

: "${LICHESS_BOT_TOKEN:?Set the bot:play bearer token in LICHESS_BOT_TOKEN}"

engine_repository_root="${ENGINE_REPOSITORY_ROOT:-/workspace/alphazero-engine}"
virtual_environment_root="${ENGINE_VENV_ROOT:-/workspace/alphazero-venv}"
lichess_bot_root="${LICHESS_BOT_ROOT:-/workspace/lichess-bot}"
source_config="${LICHESS_BOT_CONFIG_SOURCE:-${engine_repository_root}/deployment/lichess/config.yml}"
runtime_config="${LICHESS_BOT_RUNTIME_CONFIG:-/data/lichess-bot-config.yml}"
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
if [[ ! -f "${source_config}" ]]; then
    echo "Lichess configuration does not exist: ${source_config}" >&2
    exit 1
fi
if [[ "${source_config}" == "${runtime_config}" ]]; then
    echo "LICHESS_BOT_RUNTIME_CONFIG must differ from the source config." >&2
    exit 1
fi

mkdir -p "$(dirname "${runtime_config}")" /data/logs /data/pgn
sed \
    "s|^  working_dir:.*|  working_dir: \"${engine_repository_root}/py\"|" \
    "${source_config}" > "${runtime_config}"

export ENGINE_PYTHON="${virtual_environment_root}/bin/python"
export ENGINE_REPOSITORY_ROOT="${engine_repository_root}"
export MODEL_PATH="${model_path}"

echo "Engine checkout: ${engine_repository_root}"
echo "Model: ${model_path}"
echo "Runtime config: ${runtime_config}"
echo "PGNs: /data/pgn"
echo "Log: /data/logs/lichess-bot.log"

cd "${lichess_bot_root}"
exec "${virtual_environment_root}/bin/python" ./lichess-bot.py \
    --config "${runtime_config}" \
    --logfile /data/logs/lichess-bot.log
