#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 || "$#" -gt 3 ]]; then
    echo "Usage: $0 MODEL_PATH RUN_CONFIG [SOURCE_ROOT]"
    exit 1
fi

model=$1
run_config=$2
script_directory=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_root=${3:-$(git -C "${script_directory}" rev-parse --show-toplevel)}
python_root="${source_root}/py"
python_binary=${PYTHON_BINARY:-python3}
output_root=${BENCHMARK_OUTPUT_ROOT:-"${source_root}/benchmark-artifacts"}
gpu_count=${GPU_COUNT:-4}
processes_per_gpu=${PROCESSES_PER_GPU:-1}
parallel_games=${PARALLEL_GAMES_PER_PROCESS:-64}
warmup_batches=${WARMUP_BATCHES:-2}
duration_seconds=${MEASUREMENT_DURATION_SECONDS:-60}

for value in "${gpu_count}" "${processes_per_gpu}" "${parallel_games}" "${duration_seconds}"; do
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "GPU, process, game, and duration values must be positive integers."
        exit 1
    fi
done
if [[ ! "${warmup_batches}" =~ ^[0-9]+$ ]]; then
    echo "WARMUP_BATCHES must be a nonnegative integer."
    exit 1
fi
if [[ ! -f "${model}" || ! -f "${run_config}" ]]; then
    echo "The benchmark model and run configuration must exist."
    exit 1
fi
model=$(realpath "${model}")
run_config=$(realpath "${run_config}")
if [[ "$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l)" -lt "${gpu_count}" ]]; then
    echo "The requested GPU count is not visible."
    exit 1
fi

run_timestamp=$(date -u +%Y%m%dT%H%M%SZ)
game=$(
    cd "${python_root}"
    "${python_binary}" -c \
        'import sys; from pathlib import Path; from src.experiment.configuration import load_experiment_configuration; print(load_experiment_configuration(Path(sys.argv[1])).game)' \
        "${run_config}"
)
output_directory="${output_root}/self-play-search-${game}-gpus${gpu_count}-processes${processes_per_gpu}-games${parallel_games}-${duration_seconds}s-${run_timestamp}"
mkdir -p "${output_directory}"

jq -n \
    --arg timestamp "${run_timestamp}" \
    --arg revision "$(git -C "${source_root}" rev-parse HEAD)" \
    --arg model "$(realpath "${model}")" \
    --arg model_sha256 "$(sha256sum "${model}" | awk '{print $1}')" \
    --arg run_config "$(realpath "${run_config}")" \
    --arg game "${game}" \
    --argjson gpu_count "${gpu_count}" \
    --argjson processes_per_gpu "${processes_per_gpu}" \
    --argjson parallel_games "${parallel_games}" \
    --argjson warmup_batches "${warmup_batches}" \
    --argjson duration_seconds "${duration_seconds}" \
    '{
        timestamp_utc: $timestamp,
        source_revision: $revision,
        game: $game,
        model: {path: $model, sha256: $model_sha256},
        run_config: $run_config,
        topology: {
            gpu_count: $gpu_count,
            processes_per_gpu: $processes_per_gpu,
            parallel_games_per_process: $parallel_games,
            total_parallel_games: ($gpu_count * $processes_per_gpu * $parallel_games)
        },
        workload: {warmup_batches: $warmup_batches, duration_seconds: $duration_seconds}
    }' >"${output_directory}/manifest.json"

cd "${python_root}"
export PYTHONPATH=.
barrier="${output_directory}/start"
processes=()
worker_index=0
for ((device = 0; device < gpu_count; device++)); do
    for ((replica = 0; replica < processes_per_gpu; replica++)); do
        "${python_binary}" tools/benchmark_self_play_search.py \
            --run-config "${run_config}" \
            --model "${model}" \
            --device "${device}" \
            --worker-id "${worker_index}" \
            --inference-device cuda \
            --games "${parallel_games}" \
            --warmup-batches "${warmup_batches}" \
            --duration-seconds "${duration_seconds}" \
            --ready-file "${output_directory}/ready-${worker_index}" \
            --start-barrier "${barrier}" \
            >"${output_directory}/worker-${worker_index}.json" \
            2>"${output_directory}/worker-${worker_index}.stderr" &
        processes+=("$!")
        worker_index=$((worker_index + 1))
    done
done

expected_workers=$((gpu_count * processes_per_gpu))
deadline=$((SECONDS + 600))
while [[ "$(find "${output_directory}" -maxdepth 1 -name 'ready-*' | wc -l)" -ne "${expected_workers}" ]]; do
    for process in "${processes[@]}"; do
        if ! kill -0 "${process}" 2>/dev/null; then
            kill "${processes[@]}" 2>/dev/null || true
            echo "A benchmark worker failed during warm-up."
            exit 1
        fi
    done
    if [[ "${SECONDS}" -ge "${deadline}" ]]; then
        kill "${processes[@]}" 2>/dev/null || true
        echo "Timed out waiting for benchmark warm-up."
        exit 1
    fi
    sleep 1
done
touch "${barrier}"

while kill -0 "${processes[0]}" 2>/dev/null; do
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used \
        --format=csv,noheader,nounits >>"${output_directory}/gpu.csv"
    sleep 1
done
for process in "${processes[@]}"; do
    wait "${process}"
done

for worker in "${output_directory}"/worker-*.json; do
    tail -n 1 "${worker}"
done >"${output_directory}/workers.jsonl"

jq -s \
    --slurpfile manifest "${output_directory}/manifest.json" \
    '(. | map(.elapsed_seconds) | max) as $makespan
    | (. | map(.searches_completed) | add) as $searches
    | (. | map(.inference_model_calls) | add) as $calls
    | (. | map(.inference_model_positions) | add) as $positions
    | {
        manifest: $manifest[0],
        measurement: {
            makespan_seconds: $makespan,
            searches_completed: $searches,
            searches_per_second: ($searches / $makespan),
            completed_games: (map(.completed_games) | add),
            search_batches: (map(.search_batches) | add)
        },
        inference: {
            evaluations: (map(.inference_evaluations) | add),
            model_calls: $calls,
            model_positions: $positions,
            average_batch_size: (if $calls == 0 then 0 else $positions / $calls end)
        },
        resources: {
            aggregate_process_cpu_percent: (map(.process_cpu_percent) | add),
            summed_peak_rss_mib: (map(.peak_rss_mib) | add)
        },
        workers: .
    }' "${output_directory}/workers.jsonl" | tee "${output_directory}/summary.json"

echo "${output_directory}"
