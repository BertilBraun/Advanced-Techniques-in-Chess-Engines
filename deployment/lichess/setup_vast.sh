#!/usr/bin/env bash
set -euo pipefail

engine_repository_url="${ENGINE_REPOSITORY_URL:-https://github.com/BertilBraun/Advanced-Techniques-in-Chess-Engines.git}"
engine_revision="${ENGINE_REVISION:-master}"
engine_repository_root="${ENGINE_REPOSITORY_ROOT:-/workspace/alphazero-engine}"
lichess_bot_root="${LICHESS_BOT_ROOT:-/workspace/lichess-bot}"
lichess_bot_revision="${LICHESS_BOT_REVISION:-36ba4575b3299646956c911b11487b1c348f4b0c}"
virtual_environment_root="${ENGINE_VENV_ROOT:-/workspace/alphazero-venv}"
build_root="${ENGINE_BUILD_ROOT:-${engine_repository_root}/cpp/build}"
build_jobs="${ENGINE_BUILD_JOBS:-$(nproc)}"

if [[ "$(id -u)" -ne 0 ]]; then
    echo "Run setup_vast.sh as root in the Vast SSH instance." >&2
    exit 1
fi

echo "[1/8] Checking the GPU and base environment"
command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi is unavailable." >&2; exit 1; }
nvidia-smi

echo "[2/8] Installing system build dependencies"
apt-get update
apt-get install --yes --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    git \
    ninja-build \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip
mkdir -p /workspace /data

echo "[3/8] Cloning or updating the engine repository"
if [[ -d "${engine_repository_root}/.git" ]]; then
    git -C "${engine_repository_root}" fetch --tags origin
elif [[ -e "${engine_repository_root}" ]]; then
    echo "${engine_repository_root} exists but is not a Git checkout." >&2
    exit 1
else
    git clone "${engine_repository_url}" "${engine_repository_root}"
fi
if [[ "${engine_revision}" == "master" ]]; then
    git -C "${engine_repository_root}" checkout --detach origin/master
else
    git -C "${engine_repository_root}" checkout "${engine_revision}"
fi
resolved_engine_revision="$(git -C "${engine_repository_root}" rev-parse HEAD)"
echo "Engine revision: ${resolved_engine_revision}"

echo "[4/8] Creating the Python environment"
if [[ ! -x "${virtual_environment_root}/bin/python" ]]; then
    python3.10 -m venv "${virtual_environment_root}"
fi
"${virtual_environment_root}/bin/python" -m pip install --no-cache-dir uv==0.11.14
"${virtual_environment_root}/bin/uv" pip install \
    --python "${virtual_environment_root}/bin/python" \
    --require-hashes \
    --torch-backend cu128 \
    --requirements "${engine_repository_root}/py/requirements-training.lock"
"${virtual_environment_root}/bin/uv" pip install \
    --python "${virtual_environment_root}/bin/python" \
    backoff==2.2.1 \
    huggingface-hub==0.34.4 \
    pytest==8.4.2 \
    PyYAML==6.0.2 \
    requests==2.34.2 \
    rich==15.0.0

echo "[5/8] Verifying CUDA from Python"
"${virtual_environment_root}/bin/python" -c \
    "import torch; assert torch.cuda.is_available(), 'PyTorch cannot access CUDA'; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"

echo "[6/8] Compiling AlphaZeroCpp on this machine"
export PATH="${virtual_environment_root}/bin:${PATH}"
cmake -S "${engine_repository_root}/cpp" -B "${build_root}" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_NATIVE_ARCH=ON \
    -DPYTHON_EXECUTABLE="${virtual_environment_root}/bin/python"
cmake --build "${build_root}" --parallel "${build_jobs}"

echo "[7/8] Running native and Python tests"
ctest --test-dir "${build_root}" --output-on-failure
cd "${engine_repository_root}/py"
"${virtual_environment_root}/bin/python" -m pytest --import-mode=importlib \
    ./test/test_interactive_engine.py \
    ./test/test_uci_server.py \
    ./test/test_fetch_hf_model.py \
    ./test/test_run_cutechess_gauntlet.py \
    ./test/test_validate_uci_transcript.py \
    -q
"${virtual_environment_root}/bin/python" -m src.uci --help >/dev/null

echo "[8/8] Installing the UCI launcher and pinned lichess-bot"
install -m 0755 \
    "${engine_repository_root}/deployment/lichess/alphazero-uci" \
    /usr/local/bin/alphazero-uci
if [[ -d "${lichess_bot_root}/.git" ]]; then
    git -C "${lichess_bot_root}" fetch --tags origin
elif [[ -e "${lichess_bot_root}" ]]; then
    echo "${lichess_bot_root} exists but is not a Git checkout." >&2
    exit 1
else
    git clone https://github.com/lichess-bot-devs/lichess-bot.git "${lichess_bot_root}"
fi
git -C "${lichess_bot_root}" checkout "${lichess_bot_revision}"

mkdir -p /data/logs /data/models /data/pgn
cat > /data/alphazero-install.txt <<EOF
ENGINE_REPOSITORY_ROOT=${engine_repository_root}
ENGINE_REVISION=${resolved_engine_revision}
ENGINE_VENV_ROOT=${virtual_environment_root}
ENGINE_BUILD_ROOT=${build_root}
LICHESS_BOT_ROOT=${lichess_bot_root}
LICHESS_BOT_REVISION=${lichess_bot_revision}
EOF

echo
echo "Setup complete. Installation details: /data/alphazero-install.txt"
echo "Next: download/set MODEL_PATH, run smoke_vast.sh, then run_vast_bot.sh."
