#!/usr/bin/env bash

set -euo pipefail

repository_url="${ENGINE_REPOSITORY_URL:-https://github.com/BertilBraun/Advanced-Techniques-in-Chess-Engines.git}"
repository_ref="${ENGINE_REPOSITORY_REF:-master}"
repository_directory="${ENGINE_REPOSITORY_DIRECTORY:-/workspace/alphazero-engine}"
virtual_environment="${ENGINE_VIRTUAL_ENVIRONMENT:-${repository_directory}-venv}"
python_command="${ENGINE_PYTHON:-python3}"
uv_version="${ENGINE_UV_VERSION:-0.11.14}"
open_file_soft_limit="${ENGINE_OPEN_FILE_SOFT_LIMIT:-65536}"

if [[ $# -eq 0 ]]; then
    echo "Usage: setup_remote.sh COMMAND [ARGUMENT ...]" >&2
    echo "The command runs from the cloned repository with ENGINE_SOURCE_REVISION exported." >&2
    exit 2
fi

for required_command in git cmake "${python_command}"; do
    if ! command -v "${required_command}" >/dev/null 2>&1; then
        echo "Required command is unavailable: ${required_command}" >&2
        exit 1
    fi
done

if ! [[ "${open_file_soft_limit}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ENGINE_OPEN_FILE_SOFT_LIMIT must be a positive integer." >&2
    exit 1
fi
if ! ulimit -Sn "${open_file_soft_limit}"; then
    echo "Cannot raise the open-file soft limit to ${open_file_soft_limit}." >&2
    exit 1
fi

if [[ -e "${repository_directory}" ]]; then
    echo "Repository destination already exists: ${repository_directory}" >&2
    exit 1
fi
if [[ -e "${virtual_environment}" ]]; then
    echo "Virtual-environment destination already exists: ${virtual_environment}" >&2
    exit 1
fi

mkdir -p "$(dirname "${repository_directory}")"
git clone --branch "${repository_ref}" --single-branch "${repository_url}" "${repository_directory}"

"${python_command}" -m venv "${virtual_environment}"
virtual_environment_python="${virtual_environment}/bin/python"
virtual_environment_uv="${virtual_environment}/bin/uv"

"${virtual_environment_python}" -m pip install --disable-pip-version-check "uv==${uv_version}"
"${virtual_environment_uv}" pip install \
    --python "${virtual_environment_python}" \
    --require-hashes \
    --torch-backend cu126 \
    --requirements "${repository_directory}/py/requirements-training.lock"

python_nvidia_library_path="$(
    "${virtual_environment_python}" -c \
        'from pathlib import Path; import site; root = Path(site.getsitepackages()[0]) / "nvidia"; print(":".join(str(path) for path in sorted(root.glob("*/lib")) if path.is_dir()))'
)"
if [[ -n "${python_nvidia_library_path}" ]]; then
    export LD_LIBRARY_PATH="${python_nvidia_library_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

cmake \
    -S "${repository_directory}/cpp" \
    -B "${repository_directory}/cpp/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython3_EXECUTABLE="${virtual_environment_python}"
cmake --build "${repository_directory}/cpp/build" --parallel

ENGINE_EVALUATION_DIRECTORY="${repository_directory}/engines" \
    "${repository_directory}/deployment/install_evaluation_engines.sh"
ENGINE_EVALUATION_DIRECTORY="${repository_directory}/engines" \
    "${repository_directory}/deployment/smoke_evaluation_engines.sh"

if [[ -n "$(git -C "${repository_directory}" status --porcelain)" ]]; then
    echo "The checkout changed while installing or building; refusing to start the runner." >&2
    git -C "${repository_directory}" status --short >&2
    exit 1
fi

export ENGINE_SOURCE_REVISION
ENGINE_SOURCE_REVISION="$(git -C "${repository_directory}" rev-parse HEAD)"
export EVALUATION_STOCKFISH_EXECUTABLE="${repository_directory}/engines/stockfish"
export EVALUATION_KATAGO_EXECUTABLE="${repository_directory}/engines/katago"
export EVALUATION_KATAGO_MODEL="${repository_directory}/engines/katago.bin.gz"
export EVALUATION_KATAGO_CONFIGURATION="${repository_directory}/engines/katago-analysis.cfg"
export PATH="${virtual_environment}/bin:${PATH}"

cd "${repository_directory}"
exec "$@"
