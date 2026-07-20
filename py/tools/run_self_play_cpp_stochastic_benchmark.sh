#!/bin/bash
set -euo pipefail

script_directory=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
git_source_root=$(git -C "${script_directory}" rev-parse --show-toplevel)
source_root=${2:-${git_source_root}}
python_root="${source_root}/py"
python_binary=${PYTHON_BINARY:-python3}
build_directory=${BUILD_DIRECTORY:-"${source_root}/cpp/build"}
if [[ "$#" -lt 1 ]]; then
    echo "Usage: $0 MODEL_PATH [SOURCE_ROOT]"
    echo "Topology overrides: GPU_COUNT, PROCESSES_PER_GPU, MCTS_THREADS_PER_PROCESS, PARALLEL_GAMES_PER_PROCESS"
    echo "Workload overrides: WARMUP_STEPS, MEASUREMENT_DURATION_SECONDS, SEARCHES_PER_PLY, FAST_SEARCHES_PER_PLY, PARALLEL_SEARCHES, MAXIMUM_BATCH_SIZE, INFERENCE_TIMEOUT_MICROSECONDS, CACHE_CAPACITY, USE_INFERENCE_CACHE, SEED_BASE"
    echo "Affinity override: PIN_WORKERS_TO_CPUS=0|1"
    exit 1
fi
model=$1
output_root=${STOCHASTIC_BENCHMARK_OUTPUT_ROOT:-${BASELINE_OUTPUT_ROOT:-"${source_root}/benchmark-artifacts"}}

gpu_count=${GPU_COUNT:-4}
processes_per_gpu=${PROCESSES_PER_GPU:-8}
mcts_threads_per_process=${MCTS_THREADS_PER_PROCESS:-3}
parallel_games_per_process=${PARALLEL_GAMES_PER_PROCESS:-96}
warmup_steps=${WARMUP_STEPS:-1}
measurement_duration_seconds=${MEASUREMENT_DURATION_SECONDS:-120}
searches_per_ply=${SEARCHES_PER_PLY:-600}
fast_searches_per_ply=${FAST_SEARCHES_PER_PLY:-0}
parallel_searches=${PARALLEL_SEARCHES:-4}
maximum_batch_size=${MAXIMUM_BATCH_SIZE:-256}
inference_timeout_microseconds=${INFERENCE_TIMEOUT_MICROSECONDS:-500}
cache_capacity=${CACHE_CAPACITY:-1500000}
use_inference_cache=${USE_INFERENCE_CACHE:-1}
seed_base=${SEED_BASE:-1}
pin_workers_to_cpus=${PIN_WORKERS_TO_CPUS:-0}
maximum_host_ram_percent=${MAXIMUM_HOST_RAM_PERCENT:-95}
telemetry_interval_seconds=2

require_positive_integer() {
    local setting_name=$1
    local setting_value=$2
    if [[ ! "${setting_value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${setting_name} must be a positive integer, found: ${setting_value}"
        exit 1
    fi
}

require_nonnegative_integer() {
    local setting_name=$1
    local setting_value=$2
    if [[ ! "${setting_value}" =~ ^[0-9]+$ ]]; then
        echo "${setting_name} must be a nonnegative integer, found: ${setting_value}"
        exit 1
    fi
}

require_positive_integer GPU_COUNT "${gpu_count}"
require_positive_integer PROCESSES_PER_GPU "${processes_per_gpu}"
require_positive_integer MCTS_THREADS_PER_PROCESS "${mcts_threads_per_process}"
require_positive_integer PARALLEL_GAMES_PER_PROCESS "${parallel_games_per_process}"
require_nonnegative_integer WARMUP_STEPS "${warmup_steps}"
require_positive_integer MEASUREMENT_DURATION_SECONDS "${measurement_duration_seconds}"
require_positive_integer SEARCHES_PER_PLY "${searches_per_ply}"
require_nonnegative_integer FAST_SEARCHES_PER_PLY "${fast_searches_per_ply}"
require_positive_integer PARALLEL_SEARCHES "${parallel_searches}"
require_positive_integer MAXIMUM_BATCH_SIZE "${maximum_batch_size}"
require_positive_integer INFERENCE_TIMEOUT_MICROSECONDS "${inference_timeout_microseconds}"
require_positive_integer CACHE_CAPACITY "${cache_capacity}"
require_nonnegative_integer SEED_BASE "${seed_base}"
require_positive_integer MAXIMUM_HOST_RAM_PERCENT "${maximum_host_ram_percent}"
if [[ "${maximum_host_ram_percent}" -gt 100 ]]; then
    echo "MAXIMUM_HOST_RAM_PERCENT must not exceed 100, found: ${maximum_host_ram_percent}"
    exit 1
fi
if [[ "${pin_workers_to_cpus}" != "0" && "${pin_workers_to_cpus}" != "1" ]]; then
    echo "PIN_WORKERS_TO_CPUS must be 0 or 1, found: ${pin_workers_to_cpus}"
    exit 1
fi
if [[ "${use_inference_cache}" != "0" && "${use_inference_cache}" != "1" ]]; then
    echo "USE_INFERENCE_CACHE must be 0 or 1, found: ${use_inference_cache}"
    exit 1
fi
if [[ "${fast_searches_per_ply}" -gt 0 ]] && {
    [[ "${fast_searches_per_ply}" -le "${parallel_searches}" ]] ||
        [[ "${fast_searches_per_ply}" -ge "${searches_per_ply}" ]]
}; then
    echo "FAST_SEARCHES_PER_PLY must exceed PARALLEL_SEARCHES and remain below SEARCHES_PER_PLY"
    exit 1
fi

expected_processes=$((gpu_count * processes_per_gpu))
total_mcts_threads=$((expected_processes * mcts_threads_per_process))

cd "${python_root}"
export PYTHONPATH=.

if [[ ! -f "${model}" ]]; then
    echo "Benchmark model does not exist: ${model}"
    exit 1
fi

gpu_indices=
for ((device = 0; device < gpu_count; device++)); do
    if [[ -n "${gpu_indices}" ]]; then
        gpu_indices+=,
    fi
    gpu_indices+=${device}
done
visible_gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l)
logical_cpu_count=$(nproc)
maximum_mcts_threads=$((logical_cpu_count * 3))
if [[ "${visible_gpu_count}" -lt "${gpu_count}" ]]; then
    echo "Requested ${gpu_count} GPUs, but only ${visible_gpu_count} are visible"
    exit 1
fi
if [[ "${total_mcts_threads}" -gt "${maximum_mcts_threads}" ]]; then
    echo "MCTS threads ${total_mcts_threads} exceed the 3x CPU limit ${maximum_mcts_threads}"
    exit 1
fi

affinity_mode=disabled
worker_cpu_lists=()
gpu_cpu_affinity_specs=()

expand_cpu_affinity() {
    local affinity_spec=$1
    local segment
    local range_start
    local range_end
    local cpu_index
    local -a affinity_segments
    expanded_cpu_ids=()
    IFS=',' read -r -a affinity_segments <<< "${affinity_spec}"
    for segment in "${affinity_segments[@]}"; do
        if [[ "${segment}" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            range_start=${BASH_REMATCH[1]}
            range_end=${BASH_REMATCH[2]}
            if [[ "${range_start}" -gt "${range_end}" ]]; then
                return 1
            fi
            for ((cpu_index = range_start; cpu_index <= range_end; cpu_index++)); do
                expanded_cpu_ids+=("${cpu_index}")
            done
        elif [[ "${segment}" =~ ^[0-9]+$ ]]; then
            expanded_cpu_ids+=("${segment}")
        else
            return 1
        fi
    done
    [[ "${#expanded_cpu_ids[@]}" -gt 0 ]]
}

if [[ "${pin_workers_to_cpus}" -eq 1 ]]; then
    affinity_mode=taskset
    if ! command -v taskset >/dev/null 2>&1; then
        echo "PIN_WORKERS_TO_CPUS=1 requires taskset"
        exit 1
    fi
    if [[ "${logical_cpu_count}" -lt "${expected_processes}" ]]; then
        echo "CPU pinning requires at least one logical CPU per process; found ${logical_cpu_count} CPUs for ${expected_processes} processes"
        exit 1
    fi

    gpu_topology=$(nvidia-smi topo -m)
    declare -A affinity_group_gpu_counts=()
    for ((device = 0; device < gpu_count; device++)); do
        cpu_affinity=$(
            awk -v gpu="GPU${device}" '
                $1 == gpu {
                    for (field = 2; field <= NF; field++) {
                        if ($field ~ /^[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*$/) {
                            print $field
                            exit
                        }
                    }
                }
            ' <<< "${gpu_topology}"
        )
        if [[ -z "${cpu_affinity}" ]]; then
            echo "Could not parse CPU Affinity for GPU${device} from nvidia-smi topo -m"
            exit 1
        fi
        if ! expand_cpu_affinity "${cpu_affinity}"; then
            echo "Invalid CPU Affinity for GPU${device}: ${cpu_affinity}"
            exit 1
        fi
        gpu_cpu_affinity_specs[device]=${cpu_affinity}
        group_gpu_count=${affinity_group_gpu_counts["${cpu_affinity}"]:-0}
        affinity_group_gpu_counts["${cpu_affinity}"]=$((group_gpu_count + 1))
    done

    declare -A affinity_group_seen_gpus=()
    for ((device = 0; device < gpu_count; device++)); do
        cpu_affinity=${gpu_cpu_affinity_specs[device]}
        expand_cpu_affinity "${cpu_affinity}"
        group_cpu_count=${#expanded_cpu_ids[@]}
        group_gpu_count=${affinity_group_gpu_counts["${cpu_affinity}"]}
        group_worker_count=$((group_gpu_count * processes_per_gpu))
        if [[ "${group_cpu_count}" -lt "${group_worker_count}" ]]; then
            echo "CPU Affinity ${cpu_affinity} has ${group_cpu_count} CPUs for ${group_worker_count} workers"
            exit 1
        fi
        gpu_rank_in_group=${affinity_group_seen_gpus["${cpu_affinity}"]:-0}
        for ((gpu_process_index = 0; gpu_process_index < processes_per_gpu; gpu_process_index++)); do
            process_index=$((device * processes_per_gpu + gpu_process_index))
            group_worker_index=$((gpu_rank_in_group * processes_per_gpu + gpu_process_index))
            cpu_slice_start=$((group_worker_index * group_cpu_count / group_worker_count))
            cpu_slice_end=$(((group_worker_index + 1) * group_cpu_count / group_worker_count - 1))
            cpu_slice_length=$((cpu_slice_end - cpu_slice_start + 1))
            cpu_slice=("${expanded_cpu_ids[@]:cpu_slice_start:cpu_slice_length}")
            printf -v cpu_list '%s,' "${cpu_slice[@]}"
            worker_cpu_lists[process_index]=${cpu_list%,}
        done
        affinity_group_seen_gpus["${cpu_affinity}"]=$((gpu_rank_in_group + 1))
    done
fi

affinity_assignments_json=$(
    for ((worker_index = 0; worker_index < expected_processes; worker_index++)); do
        if [[ "${pin_workers_to_cpus}" -eq 0 ]]; then
            break
        fi
        device=$((worker_index / processes_per_gpu))
        jq -cn \
            --argjson process_index "${worker_index}" \
            --argjson device "${device}" \
            --arg gpu_cpu_affinity "${gpu_cpu_affinity_specs[device]}" \
            --arg cpu_list "${worker_cpu_lists[worker_index]}" \
            '{
                process_index: $process_index,
                device: $device,
                gpu_cpu_affinity: $gpu_cpu_affinity,
                cpu_list: $cpu_list
            }'
    done | jq -s .
)

cmake_cache="${build_directory}/CMakeCache.txt"
compile_commands="${build_directory}/compile_commands.json"
if [[ ! -f "${cmake_cache}" || ! -f "${compile_commands}" ]]; then
    echo "Release build metadata is missing from ${build_directory}"
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
topology_name="g${gpu_count}-pg${processes_per_gpu}-t${mcts_threads_per_process}-games${parallel_games_per_process}"
workload_name="s${searches_per_ply}-fs${fast_searches_per_ply}-ps${parallel_searches}-b${maximum_batch_size}-cache${cache_capacity}"
if [[ "${use_inference_cache}" -eq 1 ]]; then
    inference_client_name=cached
else
    inference_client_name=uncached
fi
workload_name="${workload_name}-${inference_client_name}"
output_directory="${output_root}/self-play-cpp-stochastic-${topology_name}-${workload_name}-duration${measurement_duration_seconds}s-timeout${inference_timeout_microseconds}us-affinity${affinity_mode}-${run_timestamp}"
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
gpu_inventory=$(nvidia-smi --id="${gpu_indices}" \
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
    --argjson visible_gpu_count "${visible_gpu_count}" \
    --argjson logical_cpu_count "${logical_cpu_count}" \
    --argjson processes_per_gpu "${processes_per_gpu}" \
    --argjson mcts_threads_per_process "${mcts_threads_per_process}" \
    --argjson parallel_games_per_process "${parallel_games_per_process}" \
    --argjson warmup_steps "${warmup_steps}" \
    --argjson measurement_duration_seconds "${measurement_duration_seconds}" \
    --argjson searches_per_ply "${searches_per_ply}" \
    --argjson fast_searches_per_ply "${fast_searches_per_ply}" \
    --argjson parallel_searches "${parallel_searches}" \
    --argjson maximum_batch_size "${maximum_batch_size}" \
    --argjson inference_timeout_microseconds "${inference_timeout_microseconds}" \
    --argjson cache_capacity "${cache_capacity}" \
    --argjson use_inference_cache "${use_inference_cache}" \
    --argjson seed_base "${seed_base}" \
    --argjson maximum_host_ram_percent "${maximum_host_ram_percent}" \
    --arg affinity_mode "${affinity_mode}" \
    --argjson pin_workers_to_cpus "${pin_workers_to_cpus}" \
    --argjson affinity_assignments "${affinity_assignments_json}" \
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
            visible_gpu_count: $visible_gpu_count,
            logical_cpu_count: $logical_cpu_count,
            gpu_inventory: ($gpu_inventory | split("\n"))
        },
        topology: {
            processes_per_gpu: $processes_per_gpu,
            total_processes: ($gpu_count * $processes_per_gpu),
            mcts_threads_per_process: $mcts_threads_per_process,
            total_mcts_threads: ($gpu_count * $processes_per_gpu * $mcts_threads_per_process),
            parallel_games_per_process: $parallel_games_per_process,
            total_parallel_games: ($gpu_count * $processes_per_gpu * $parallel_games_per_process),
            affinity: {
                mode: $affinity_mode,
                pin_workers_to_cpus: ($pin_workers_to_cpus == 1),
                worker_assignments: $affinity_assignments
            }
        },
        workload: {
            model_path: $model_path,
            model_sha256: $model_sha256,
            warmup_steps: $warmup_steps,
            measurement_duration_seconds: $measurement_duration_seconds,
            target_searches_per_ply: $searches_per_ply,
            target_fast_searches_per_ply_override: $fast_searches_per_ply,
            parallel_searches: $parallel_searches,
            maximum_batch_size: $maximum_batch_size,
            inference_timeout_microseconds: $inference_timeout_microseconds,
            cache_capacity_per_process: $cache_capacity,
            use_inference_cache: ($use_inference_cache == 1),
            seed_base: $seed_base
        },
        safety: {
            maximum_host_ram_percent: $maximum_host_ram_percent
        }
    }' > "${output_directory}/manifest.json"

start_barrier="${output_directory}/start"
child_processes=()
process_index=0
for ((device = 0; device < gpu_count; device++)); do
    for ((gpu_process_index = 0; gpu_process_index < processes_per_gpu; gpu_process_index++)); do
        worker_seed=$((seed_base + process_index))
        worker_command=(
            "${python_binary}" tools/benchmark_stochastic_self_play_cpp.py
            --model "${model}"
            --device "${device}"
            --games "${parallel_games_per_process}"
            --duration-seconds "${measurement_duration_seconds}"
            --warmup-steps "${warmup_steps}"
            --searches "${searches_per_ply}"
            --parallel-searches "${parallel_searches}"
            --threads "${mcts_threads_per_process}"
            --maximum-batch-size "${maximum_batch_size}"
            --inference-timeout-microseconds "${inference_timeout_microseconds}"
            --cache-capacity "${cache_capacity}"
            --seed "${worker_seed}"
            --ready-file "${output_directory}/ready-${process_index}"
            --start-barrier "${start_barrier}"
        )
        if [[ "${use_inference_cache}" -eq 0 ]]; then
            worker_command+=(--no-inference-cache)
        fi
        if [[ "${fast_searches_per_ply}" -gt 0 ]]; then
            worker_command+=(--fast-searches "${fast_searches_per_ply}")
        fi
        if [[ "${pin_workers_to_cpus}" -eq 1 ]]; then
            taskset --cpu-list "${worker_cpu_lists[process_index]}" "${worker_command[@]}" \
                > "${output_directory}/worker-${process_index}.json" \
                2> "${output_directory}/worker-${process_index}.stderr" &
        else
            "${worker_command[@]}" \
                > "${output_directory}/worker-${process_index}.json" \
                2> "${output_directory}/worker-${process_index}.stderr" &
        fi
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
        nvidia-smi --id="${gpu_indices}" \
            --query-gpu=index,utilization.gpu,memory.used \
            --format=csv,noheader,nounits
    )

    if awk "BEGIN {exit !(${host_ram_percent} >= ${maximum_host_ram_percent})}"; then
        kill "${child_processes[@]}" || true
        echo "Benchmark reached the ${maximum_host_ram_percent}% host-RAM guard"
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

worker_results="${output_directory}/worker-results.jsonl"
for worker_output in "${output_directory}"/worker-*.json; do
    tail -n 1 "${worker_output}"
done > "${worker_results}"

jq -s \
    --argjson gpu_count "${gpu_count}" \
    '{
        samples_per_gpu: (length / $gpu_count),
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
        | ($workers | map(.completed_games) | add) as $completed_games
        | ($workers | map(.generated_samples) | add) as $generated_samples
        | ($workers | map(.retained_samples) | add) as $retained_samples
        | ($workers | map(.self_play_steps) | add) as $self_play_steps
        | ($workers | map(.game_updates) | add) as $game_updates
        | ($workers | map(.completed_game_plies) | add) as $completed_game_plies
        | ($workers | map(.inference_model_calls) | add) as $model_calls
        | ($workers | map(.inference_model_positions) | add) as $model_positions
        | ($workers | map(.inference_evaluations) | add) as $evaluations
        | ($workers | map(.inference_cache_hits) | add) as $cache_hits
        | {
            manifest: $manifest[0],
            measurement: {
                started_at_utc: $measurement_started_at,
                finished_at_utc: $measurement_finished_at,
                makespan_seconds: $makespan,
                completed_games: $completed_games,
                completed_games_per_second: ($completed_games / $makespan),
                generated_samples: $generated_samples,
                generated_samples_per_second: ($generated_samples / $makespan),
                retained_samples: $retained_samples,
                retained_samples_per_second: ($retained_samples / $makespan),
                self_play_steps: $self_play_steps,
                self_play_steps_per_second: ($self_play_steps / $makespan),
                game_updates: $game_updates,
                game_updates_per_second: ($game_updates / $makespan),
                milliseconds_per_game_update: (1000 * $makespan / $game_updates),
                completed_game_plies: $completed_game_plies,
                completed_game_plies_per_second: ($completed_game_plies / $makespan)
            },
            inference: {
                evaluations: $evaluations,
                cache_hits: $cache_hits,
                cache_hit_rate_percent: (
                    if $evaluations == 0 then 0 else 100 * $cache_hits / $evaluations end
                ),
                model_calls: $model_calls,
                model_positions: $model_positions,
                average_batch_size: (
                    if $model_calls == 0 then 0 else $model_positions / $model_calls end
                ),
                batch_size_distribution: (
                    [$workers[].inference_batch_size_distribution[]]
                    | group_by(.batch_size)
                    | map({
                        batch_size: .[0].batch_size,
                        calls: (map(.calls) | add)
                    })
                ),
                unique_positions: ($workers | map(.inference_unique_positions) | add),
                cache_size_mib_at_end: ($workers | map(.inference_cache_size_mib) | add),
                cache_evictions: ($workers | map(.inference_cache_evictions) | add),
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
            search_trees: {
                live_materialized_nodes_at_end: (
                    $workers
                    | map(.live_materialized_nodes)
                    | add
                ),
                total_child_records_at_end: (
                    $workers
                    | map(.total_child_records)
                    | add
                ),
                arena_capacity_per_game: ($workers[0].arena_capacity_per_game)
            },
            workers: {
                count: ($workers | length),
                minimum_elapsed_seconds: ($workers | map(.elapsed_seconds) | min),
                maximum_elapsed_seconds: $makespan
            }
        }
    ' "${worker_results}" \
    | tee "${output_directory}/summary.json"

echo "${output_directory}"
