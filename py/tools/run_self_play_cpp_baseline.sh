#!/bin/bash
set -euo pipefail

script_directory=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
git_source_root=$(git -C "${script_directory}" rev-parse --show-toplevel)
source_root=${4:-${git_source_root}}
python_root="${source_root}/py"
python_binary=${PYTHON_BINARY:-python3}
openings="${python_root}/reference/main-monitoring-openings-50.tsv"
if [[ "$#" -lt 1 ]]; then
    echo "Usage: $0 MODEL_PATH [WARMUP_STEPS] [MEASUREMENT_STEPS] [SOURCE_ROOT]"
    exit 1
fi
model=$1
warmup_steps=${2:-2}
measurement_steps=${3:-10}
output_root=${BASELINE_OUTPUT_ROOT:-"${source_root}/benchmark-artifacts"}

gpu_count=4
processes_per_gpu=8
mcts_threads_per_process=3
parallel_games_per_process=96
searches_per_ply=600
parallel_searches=4
maximum_batch_size=256
inference_timeout_microseconds=500
cache_capacity=1500000
telemetry_interval_seconds=2
expected_processes=$((gpu_count * processes_per_gpu))
total_mcts_threads=$((expected_processes * mcts_threads_per_process))

cd "${python_root}"
export PYTHONPATH=.

if [[ ! -f "${model}" ]]; then
    echo "Benchmark model does not exist: ${model}"
    exit 1
fi

visible_gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l)
logical_cpu_count=$(nproc)
if [[ "${visible_gpu_count}" -ne "${gpu_count}" ]]; then
    echo "Expected ${gpu_count} visible GPUs, found ${visible_gpu_count}"
    exit 1
fi
if [[ "${logical_cpu_count}" -lt "${total_mcts_threads}" ]]; then
    echo "Expected at least ${total_mcts_threads} logical CPUs, found ${logical_cpu_count}"
    exit 1
fi

cmake_cache="${source_root}/cpp/build/CMakeCache.txt"
compile_commands="${source_root}/cpp/build/compile_commands.json"
if [[ ! -f "${cmake_cache}" || ! -f "${compile_commands}" ]]; then
    echo "Release build metadata is missing from ${source_root}/cpp/build"
    exit 1
fi

build_type=$(awk -F= '/^CMAKE_BUILD_TYPE:STRING=/{print $2}' "${cmake_cache}")
timing_instrumentation=$(awk -F= '/^ENABLE_TIMING_INSTRUMENTATION:BOOL=/{print $2}' "${cmake_cache}")
if [[ "${build_type}" != "Release" || "${timing_instrumentation}" != "OFF" ]]; then
    echo "Expected Release with timing instrumentation OFF; found ${build_type}/${timing_instrumentation}"
    exit 1
fi
if ! grep -q -- '-O3' "${compile_commands}"; then
    echo "The recorded compile commands do not contain -O3"
    exit 1
fi
if grep -q -- '-DENABLE_TIMING' "${compile_commands}"; then
    echo "The recorded compile commands enable timing instrumentation"
    exit 1
fi

run_timestamp=$(date -u +%Y%m%dT%H%M%SZ)
output_directory="${output_root}/self-play-cpp-baseline-4x8x3x96-${run_timestamp}"
mkdir -p "${output_directory}"
exec > >(tee -a "${output_directory}/console.log") 2>&1

source_revision=$(git -C "${source_root}" rev-parse HEAD)
source_worktree_clean=true
if [[ -n "$(git -C "${source_root}" status --short)" ]]; then
    source_worktree_clean=false
fi
model_sha256=$(sha256sum "${model}" | awk '{print $1}')
module_sha256=$(sha256sum "${python_root}/AlphaZeroCpp.so" | awk '{print $1}')
torch_version=$("${python_binary}" -c 'import torch; print(torch.__version__)')
cuda_version=$("${python_binary}" -c 'import torch; print(torch.version.cuda)')
gpu_inventory=$(nvidia-smi \
    --query-gpu=index,name,memory.total,driver_version \
    --format=csv,noheader,nounits)

jq -n \
    --arg run_timestamp_utc "${run_timestamp}" \
    --arg source_revision "${source_revision}" \
    --argjson source_worktree_clean "${source_worktree_clean}" \
    --arg model_path "${model}" \
    --arg model_sha256 "${model_sha256}" \
    --arg module_sha256 "${module_sha256}" \
    --arg build_type "${build_type}" \
    --arg timing_instrumentation "${timing_instrumentation}" \
    --arg torch_version "${torch_version}" \
    --arg cuda_version "${cuda_version}" \
    --arg gpu_inventory "${gpu_inventory}" \
    --argjson gpu_count "${gpu_count}" \
    --argjson logical_cpu_count "${logical_cpu_count}" \
    --argjson processes_per_gpu "${processes_per_gpu}" \
    --argjson mcts_threads_per_process "${mcts_threads_per_process}" \
    --argjson parallel_games_per_process "${parallel_games_per_process}" \
    --argjson warmup_steps "${warmup_steps}" \
    --argjson measurement_steps "${measurement_steps}" \
    --argjson searches_per_ply "${searches_per_ply}" \
    --argjson parallel_searches "${parallel_searches}" \
    --argjson maximum_batch_size "${maximum_batch_size}" \
    --argjson inference_timeout_microseconds "${inference_timeout_microseconds}" \
    --argjson cache_capacity "${cache_capacity}" \
    '{
        run_timestamp_utc: $run_timestamp_utc,
        source: {
            revision: $source_revision,
            worktree_clean: $source_worktree_clean,
            alpha_zero_cpp_sha256: $module_sha256
        },
        build: {
            type: $build_type,
            optimization: "-O3",
            timing_instrumentation: $timing_instrumentation
        },
        runtime: {
            torch_version: $torch_version,
            cuda_version: $cuda_version
        },
        hardware: {
            gpu_count: $gpu_count,
            logical_cpu_count: $logical_cpu_count,
            gpu_inventory: ($gpu_inventory | split("\n"))
        },
        topology: {
            processes_per_gpu: $processes_per_gpu,
            total_processes: ($gpu_count * $processes_per_gpu),
            mcts_threads_per_process: $mcts_threads_per_process,
            total_mcts_threads: ($gpu_count * $processes_per_gpu * $mcts_threads_per_process),
            parallel_games_per_process: $parallel_games_per_process,
            total_parallel_games: ($gpu_count * $processes_per_gpu * $parallel_games_per_process)
        },
        workload: {
            model_path: $model_path,
            model_sha256: $model_sha256,
            warmup_steps: $warmup_steps,
            measurement_steps: $measurement_steps,
            target_searches_per_ply: $searches_per_ply,
            parallel_searches: $parallel_searches,
            maximum_batch_size: $maximum_batch_size,
            inference_timeout_microseconds: $inference_timeout_microseconds,
            cache_capacity_per_process: $cache_capacity
        }
    }' > "${output_directory}/manifest.json"

start_barrier="${output_directory}/start"
child_processes=()
process_index=0
for device in 0 1 2 3; do
    for _ in $(seq 1 "${processes_per_gpu}"); do
        "${python_binary}" tools/benchmark_repetition_mcts.py \
            --model "${model}" \
            --openings "${openings}" \
            --device "${device}" \
            --games "${parallel_games_per_process}" \
            --warmup-steps "${warmup_steps}" \
            --steps "${measurement_steps}" \
            --searches "${searches_per_ply}" \
            --parallel-searches "${parallel_searches}" \
            --threads "${mcts_threads_per_process}" \
            --maximum-batch-size "${maximum_batch_size}" \
            --inference-timeout-microseconds "${inference_timeout_microseconds}" \
            --cache-capacity "${cache_capacity}" \
            --gpu-sampling-interval-seconds 0 \
            --ready-file "${output_directory}/ready-${process_index}" \
            --start-barrier "${start_barrier}" \
            > "${output_directory}/worker-${process_index}.json" \
            2> "${output_directory}/worker-${process_index}.stderr" &
        child_processes+=($!)
        process_index=$((process_index + 1))
        sleep 1
    done
done

ready_deadline=$((SECONDS + 600))
while [[ "$(find "${output_directory}" -maxdepth 1 -name 'ready-*' | wc -l)" -ne "${expected_processes}" ]]; do
    if [[ "${SECONDS}" -ge "${ready_deadline}" ]]; then
        kill "${child_processes[@]}" || true
        echo "Timed out waiting for ${expected_processes} warmed benchmark workers"
        exit 1
    fi
    sleep 1
done

measurement_started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
touch "${start_barrier}"

while true; do
    running=0
    for child_process in "${child_processes[@]}"; do
        if kill -0 "${child_process}" 2>/dev/null; then
            running=1
            break
        fi
    done
    if [[ "${running}" -eq 0 ]]; then
        break
    fi

    timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    host_ram_percent=$(free | awk '/^Mem:/ {printf "%.3f", 100 * $3 / $2}')
    while IFS=',' read -r gpu_index gpu_utilization gpu_memory; do
        jq -cn \
            --arg timestamp_utc "${timestamp}" \
            --argjson host_ram_percent "${host_ram_percent}" \
            --argjson gpu_index "${gpu_index// /}" \
            --argjson gpu_utilization_percent "${gpu_utilization// /}" \
            --argjson gpu_memory_used_mib "${gpu_memory// /}" \
            '{
                timestamp_utc: $timestamp_utc,
                host_ram_percent: $host_ram_percent,
                gpu_index: $gpu_index,
                gpu_utilization_percent: $gpu_utilization_percent,
                gpu_memory_used_mib: $gpu_memory_used_mib
            }' >> "${output_directory}/resource-telemetry.jsonl"
    done < <(
        nvidia-smi \
            --query-gpu=index,utilization.gpu,memory.used \
            --format=csv,noheader,nounits
    )

    if awk "BEGIN {exit !(${host_ram_percent} >= 95.0)}"; then
        kill "${child_processes[@]}" || true
        echo "Benchmark reached the 95% host-RAM guard"
        exit 1
    fi
    sleep "${telemetry_interval_seconds}"
done
measurement_finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)

exit_status=0
for child_process in "${child_processes[@]}"; do
    wait "${child_process}" || exit_status=$?
done
if [[ "${exit_status}" -ne 0 ]]; then
    echo "Benchmark failed with status ${exit_status}"
    exit "${exit_status}"
fi

jq -s \
    '{
        samples_per_gpu: (length / 4),
        peak_host_ram_percent: (map(.host_ram_percent) | max),
        gpus: (
            group_by(.gpu_index)
            | map({
                index: .[0].gpu_index,
                mean_utilization_percent: (map(.gpu_utilization_percent) | add / length),
                peak_memory_used_mib: (map(.gpu_memory_used_mib) | max)
            })
        )
    }' "${output_directory}/resource-telemetry.jsonl" \
    > "${output_directory}/resource-summary.json"

jq -s \
    --arg measurement_started_at "${measurement_started_at}" \
    --arg measurement_finished_at "${measurement_finished_at}" \
    --argjson logical_cpu_count "${logical_cpu_count}" \
    --slurpfile manifest "${output_directory}/manifest.json" \
    --slurpfile resources "${output_directory}/resource-summary.json" \
    '
        . as $workers
        | ($workers | map(.elapsed_seconds) | max) as $makespan
        | ($workers | map(.completed_game_plies) | add) as $plies
        | ($workers | map(.searches_completed) | add) as $searches
        | ($workers | map(.inference_model_calls) | add) as $model_calls
        | ($workers | map(.inference_model_positions) | add) as $model_positions
        | {
            manifest: $manifest[0],
            measurement: {
                started_at_utc: $measurement_started_at,
                finished_at_utc: $measurement_finished_at,
                makespan_seconds: $makespan,
                completed_game_plies: $plies,
                completed_game_plies_per_second: ($plies / $makespan),
                searches_completed: $searches,
                searches_per_second: ($searches / $makespan)
            },
            inference: {
                evaluations: ($workers | map(.inference_evaluations) | add),
                cache_hits: ($workers | map(.inference_cache_hits) | add),
                cache_hit_rate_percent: (
                    100
                    * ($workers | map(.inference_cache_hits) | add)
                    / ($workers | map(.inference_evaluations) | add)
                ),
                model_calls: $model_calls,
                model_positions: $model_positions,
                average_batch_size: ($model_positions / $model_calls),
                batch_size_distribution: (
                    [$workers[].inference_batch_size_distribution[]]
                    | group_by(.batch_size)
                    | map({
                        batch_size: .[0].batch_size,
                        calls: (map(.calls) | add)
                    })
                ),
                cache_entries_at_end: ($workers | map(.inference_unique_positions) | add),
                cache_size_mib_at_end: ($workers | map(.inference_cache_size_mib) | add),
                fingerprint_collisions: (
                    $workers
                    | map(.inference_cache_fingerprint_collisions)
                    | add
                )
            },
            resources: (
                $resources[0]
                + {
                    aggregate_process_cpu_percent: (
                        $workers
                        | map(.process_cpu_percent)
                        | add
                    ),
                    estimated_host_cpu_capacity_percent: (
                        ($workers | map(.process_cpu_percent) | add)
                        / $logical_cpu_count
                    ),
                    summed_worker_peak_rss_mib: ($workers | map(.peak_rss_mib) | add)
                }
            ),
            workers: {
                count: ($workers | length),
                minimum_elapsed_seconds: ($workers | map(.elapsed_seconds) | min),
                maximum_elapsed_seconds: $makespan
            }
        }
    ' "${output_directory}"/worker-*.json \
    | tee "${output_directory}/summary.json"

echo "${output_directory}"
