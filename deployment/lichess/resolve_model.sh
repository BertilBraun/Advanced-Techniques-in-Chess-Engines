# Sourced by run_vast_bot.sh and smoke_vast.sh: resolves model_path, fetching from Hugging Face when unset.
# Expects engine_repository_root and virtual_environment_root; sets model_path.

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
