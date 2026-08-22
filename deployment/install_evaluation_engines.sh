#!/usr/bin/env bash

set -euo pipefail

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_directory="$(cd -- "${script_directory}/.." && pwd)"
engine_directory="${ENGINE_EVALUATION_DIRECTORY:-${repository_directory}/engines}"

stockfish_version="18"
stockfish_archive_name="stockfish-ubuntu-x86-64.tar"
stockfish_archive_url="${ENGINE_STOCKFISH_ARCHIVE_URL:-https://github.com/official-stockfish/Stockfish/releases/download/sf_18/${stockfish_archive_name}}"
stockfish_archive_sha256="${ENGINE_STOCKFISH_ARCHIVE_SHA256:-5c6f38b02a4da5f3ffe763f27da6c3e743eebefd92b50cb3661623b96696adff}"

# The sf_13 GitHub release ships no binary assets; the official stockfishchess.org download of that era
# survives only in the Web Archive. The pinned sha256 is the integrity anchor, not the host.
stockfish13_version="13"
stockfish13_archive_name="stockfish_13_linux_x64_bmi2.zip"
stockfish13_archive_url="${ENGINE_STOCKFISH13_ARCHIVE_URL:-https://web.archive.org/web/20210422110408id_/https://stockfishchess.org/files/${stockfish13_archive_name}}"
stockfish13_archive_sha256="${ENGINE_STOCKFISH13_ARCHIVE_SHA256:-7b6a46564c71745ac64b546fa9b8ce8c726d8b6f754f3bc8a1eebe2f1ce7e5a8}"

katago_version="1.17.1"
default_katago_backend="cuda12.8-cudnn9.8.0"
default_katago_archive_name="katago-v${katago_version}-${default_katago_backend}-linux-x64.zip"
default_katago_archive_url="https://github.com/lightvector/KataGo/releases/download/v${katago_version}/${default_katago_archive_name}"
default_katago_archive_sha256="458d226c2c8533600251bba3b2ee612d3aee0c796f592a2b53839a6a05b0826e"
katago_backend="${ENGINE_KATAGO_BACKEND:-${default_katago_backend}}"
katago_archive_url="${ENGINE_KATAGO_ARCHIVE_URL:-${default_katago_archive_url}}"
katago_archive_sha256="${ENGINE_KATAGO_ARCHIVE_SHA256:-${default_katago_archive_sha256}}"

katago_model_name="b10c384h6nbttflrs.bin.gz"
katago_model_url="${ENGINE_KATAGO_MODEL_URL:-https://github.com/lightvector/KataGo/releases/download/v${katago_version}/${katago_model_name}}"
katago_model_sha256="${ENGINE_KATAGO_MODEL_SHA256:-0ba27eced5180b3e3d0b898b280c541112989765e789d1eb6cd0d31b2b2c1229}"

if [[ -e "${engine_directory}" ]]; then
    echo "Evaluation-engine destination already exists: ${engine_directory}" >&2
    exit 1
fi

for required_command in curl sha256sum tar unzip; do
    if ! command -v "${required_command}" >/dev/null 2>&1; then
        echo "Required command is unavailable: ${required_command}" >&2
        exit 1
    fi
done

if [[ -n "${ENGINE_KATAGO_BACKEND:-}" ]] ||
    [[ -n "${ENGINE_KATAGO_ARCHIVE_URL:-}" ]] ||
    [[ -n "${ENGINE_KATAGO_ARCHIVE_SHA256:-}" ]]; then
    if [[ -z "${ENGINE_KATAGO_BACKEND:-}" ]] ||
        [[ -z "${ENGINE_KATAGO_ARCHIVE_URL:-}" ]] ||
        [[ -z "${ENGINE_KATAGO_ARCHIVE_SHA256:-}" ]]; then
        echo "A KataGo archive override requires ENGINE_KATAGO_BACKEND, ENGINE_KATAGO_ARCHIVE_URL, and ENGINE_KATAGO_ARCHIVE_SHA256 together." >&2
        exit 1
    fi
fi
if [[ "${katago_backend}" != cuda* ]]; then
    echo "KataGo must use a CUDA/cuDNN backend; CPU, OpenCL, and TensorRT builds are unsupported for this project." >&2
    exit 1
fi

temporary_directory="$(mktemp -d)"
cleanup() {
    rm -rf -- "${temporary_directory}"
}
trap cleanup EXIT

download_verified() {
    local url="$1"
    local expected_sha256="$2"
    local destination="$3"

    curl --fail --location --retry 3 --output "${destination}" "${url}"
    printf '%s  %s\n' "${expected_sha256}" "${destination}" | sha256sum --check --status
}

stockfish_archive="${temporary_directory}/${stockfish_archive_name}"
stockfish13_archive="${temporary_directory}/${stockfish13_archive_name}"
katago_archive="${temporary_directory}/katago.zip"
katago_model="${temporary_directory}/${katago_model_name}"

download_verified "${stockfish_archive_url}" "${stockfish_archive_sha256}" "${stockfish_archive}"
download_verified "${stockfish13_archive_url}" "${stockfish13_archive_sha256}" "${stockfish13_archive}"
download_verified "${katago_archive_url}" "${katago_archive_sha256}" "${katago_archive}"
download_verified "${katago_model_url}" "${katago_model_sha256}" "${katago_model}"

mkdir -p "${temporary_directory}/stockfish" "${temporary_directory}/stockfish13" "${temporary_directory}/katago"
tar -xf "${stockfish_archive}" -C "${temporary_directory}/stockfish"
unzip -q "${stockfish13_archive}" -d "${temporary_directory}/stockfish13"
unzip -q "${katago_archive}" -d "${temporary_directory}/katago"

stockfish_executable="${temporary_directory}/stockfish/stockfish/stockfish-ubuntu-x86-64"
stockfish13_executable="${temporary_directory}/stockfish13/stockfish_13_linux_x64_bmi2/stockfish_13_linux_x64_bmi2"
katago_executable="${temporary_directory}/katago/katago"
if [[ ! -f "${stockfish_executable}" ]]; then
    echo "Pinned Stockfish archive did not contain the expected executable." >&2
    exit 1
fi
if [[ ! -f "${stockfish13_executable}" ]]; then
    echo "Pinned Stockfish 13 archive did not contain the expected executable." >&2
    exit 1
fi
if [[ ! -f "${katago_executable}" ]]; then
    echo "Pinned KataGo archive did not contain the expected executable." >&2
    exit 1
fi

katago_runtime_relative_path="katago"
if (
    cd "${temporary_directory}/katago"
    "${katago_executable}" --appimage-extract >/dev/null
); then
    mv "${temporary_directory}/katago/squashfs-root" "${temporary_directory}/katago/appimage-root"
    katago_runtime_relative_path="appimage-root/AppRun"
fi

mkdir -p "${engine_directory}/upstream"
mv "${temporary_directory}/stockfish/stockfish" "${engine_directory}/upstream/stockfish-${stockfish_version}"
mv "${temporary_directory}/stockfish13/stockfish_13_linux_x64_bmi2" "${engine_directory}/upstream/stockfish-${stockfish13_version}"
mv "${temporary_directory}/katago" "${engine_directory}/upstream/katago-${katago_version}-${katago_backend}"
mv "${katago_model}" "${engine_directory}/${katago_model_name}"
cp "${script_directory}/katago-analysis.cfg" "${engine_directory}/katago-analysis.cfg"

ln -s "upstream/stockfish-${stockfish_version}/stockfish-ubuntu-x86-64" "${engine_directory}/stockfish"
ln -s "upstream/stockfish-${stockfish13_version}/stockfish_13_linux_x64_bmi2" "${engine_directory}/stockfish-13"
ln -s "upstream/katago-${katago_version}-${katago_backend}/${katago_runtime_relative_path}" "${engine_directory}/katago"
ln -s "${katago_model_name}" "${engine_directory}/katago.bin.gz"
chmod +x "${engine_directory}/stockfish" "${engine_directory}/stockfish-13" "${engine_directory}/katago"

cat >"${engine_directory}/ARTIFACTS.sha256" <<EOF
${stockfish_archive_sha256}  ${stockfish_archive_name}
${stockfish13_archive_sha256}  ${stockfish13_archive_name}
${katago_archive_sha256}  katago-v${katago_version}-${katago_backend}-linux-x64.zip
${katago_model_sha256}  ${katago_model_name}
EOF

cat >"${engine_directory}/INSTALLATION.txt" <<EOF
Stockfish version: ${stockfish_version}
Stockfish archive: ${stockfish_archive_url}
Stockfish 13 version: ${stockfish13_version}
Stockfish 13 archive: ${stockfish13_archive_url}
KataGo version: ${katago_version}
KataGo backend: ${katago_backend}
KataGo archive: ${katago_archive_url}
KataGo runtime path: ${katago_runtime_relative_path}
KataGo model: ${katago_model_url}
EOF

# Protocol smoke lives in smoke_evaluation_engines.sh, which provisioning runs right after this script.
echo "Installed evaluation engines in ${engine_directory}"
