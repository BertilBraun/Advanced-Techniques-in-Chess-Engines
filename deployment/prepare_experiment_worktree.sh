#!/usr/bin/env bash

set -euo pipefail

script_repository="$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel)"
source_root="${ENGINE_SOURCE_ROOT:-${script_repository}}"
control_repository="${ENGINE_CONTROL_REPOSITORY:-/workspace/alphazero-engine}"
virtual_environment_python="${ENGINE_VIRTUAL_ENVIRONMENT_PYTHON:-/workspace/alphazero-engine-venv/bin/python}"
engine_directory="${ENGINE_EVALUATION_DIRECTORY:-${control_repository}/engines}"
build_cache_root="${ENGINE_BUILD_CACHE_ROOT:-/workspace/experiment-build-cache}"
shared_dependency_root="${ENGINE_SHARED_DEPENDENCY_ROOT:-${control_repository}/cpp/build/_deps}"
build_jobs="${ENGINE_BUILD_JOBS:-$(nproc)}"
trust_existing_native_build="${ENGINE_TRUST_EXISTING_NATIVE_BUILD:-0}"

for required_command in cmake flock git sha256sum; do
    if ! command -v "${required_command}" >/dev/null 2>&1; then
        echo "Required command is unavailable: ${required_command}" >&2
        exit 1
    fi
done
if [[ ! -x "${virtual_environment_python}" ]]; then
    echo "Training Python is unavailable: ${virtual_environment_python}" >&2
    exit 1
fi
if [[ ! -d "${engine_directory}" ]]; then
    echo "Evaluation engines are unavailable: ${engine_directory}" >&2
    exit 1
fi
for dependency in pybind11-src stockfish-src nlohmann_json-src; do
    if [[ ! -d "${shared_dependency_root}/${dependency}" ]]; then
        echo "Shared CMake dependency is unavailable: ${shared_dependency_root}/${dependency}" >&2
        exit 1
    fi
done

engine_projection="${source_root}/engines"
if [[ -L "${engine_projection}" ]]; then
    if [[ "$(readlink -f "${engine_projection}")" != "$(readlink -f "${engine_directory}")" ]]; then
        echo "Existing engine projection targets the wrong directory: ${engine_projection}" >&2
        exit 1
    fi
elif [[ -e "${engine_projection}" ]]; then
    echo "Engine projection already exists and is not a symlink: ${engine_projection}" >&2
    exit 1
else
    cmake -E create_symlink "${engine_directory}" "${engine_projection}"
fi

cpp_tree="$(git -C "${source_root}" rev-parse HEAD:cpp)"
runtime_identity="$(${virtual_environment_python} -c 'import platform, sysconfig, torch; print("|".join((platform.machine(), sysconfig.get_config_var("SOABI"), torch.__version__, str(torch.version.cuda), str(torch._C._GLIBCXX_USE_CXX11_ABI))))')"
compiler_identity="$(c++ --version | head -n 1)"
cache_key="$(printf '%s\n%s\n%s\n' "${cpp_tree}" "${runtime_identity}" "${compiler_identity}" | sha256sum | cut -d ' ' -f 1)"
native_cache_directory="${build_cache_root}/native/${cache_key}"
lock_path="${build_cache_root}/locks/${cache_key}.lock"
mkdir -p "$(dirname "${lock_path}")" "$(dirname "${native_cache_directory}")"

exec 9>"${lock_path}"
flock 9

install_cached_artifacts() {
    cp "${native_cache_directory}/AlphaZeroCpp.so" "${source_root}/py/AlphaZeroCpp.so"
    cp "${native_cache_directory}/AlphaZeroCpp.pyi" "${source_root}/py/AlphaZeroCpp.pyi"
}

if [[ -f "${native_cache_directory}/AlphaZeroCpp.so" && -f "${native_cache_directory}/AlphaZeroCpp.pyi" ]]; then
    install_cached_artifacts
    echo "Reused native build cache ${cache_key}."
    exit 0
fi

if [[ "${trust_existing_native_build}" != 1 || ! -f "${source_root}/py/AlphaZeroCpp.so" || ! -f "${source_root}/py/AlphaZeroCpp.pyi" ]]; then
    cmake \
        -S "${source_root}/cpp" \
        -B "${source_root}/cpp/build" \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DPython3_EXECUTABLE="${virtual_environment_python}" \
        -DFETCHCONTENT_SOURCE_DIR_PYBIND11="${shared_dependency_root}/pybind11-src" \
        -DFETCHCONTENT_SOURCE_DIR_STOCKFISH="${shared_dependency_root}/stockfish-src" \
        -DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON="${shared_dependency_root}/nlohmann_json-src"
    cmake --build "${source_root}/cpp/build" --parallel "${build_jobs}"
fi

temporary_cache_directory="$(mktemp -d "${build_cache_root}/native/.${cache_key}.XXXXXX")"
trap 'rm -rf -- "${temporary_cache_directory}"' EXIT
cp "${source_root}/py/AlphaZeroCpp.so" "${temporary_cache_directory}/AlphaZeroCpp.so"
cp "${source_root}/py/AlphaZeroCpp.pyi" "${temporary_cache_directory}/AlphaZeroCpp.pyi"
printf '%s\n' "${cpp_tree}" >"${temporary_cache_directory}/cpp-tree.txt"
printf '%s\n' "${runtime_identity}" >"${temporary_cache_directory}/runtime-identity.txt"
printf '%s\n' "${compiler_identity}" >"${temporary_cache_directory}/compiler-identity.txt"
mv "${temporary_cache_directory}" "${native_cache_directory}"
trap - EXIT
echo "Stored native build cache ${cache_key}."
