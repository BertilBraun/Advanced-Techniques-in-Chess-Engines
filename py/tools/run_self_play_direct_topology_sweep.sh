#!/bin/bash
set -euo pipefail

if [[ "$#" -lt 1 || "$#" -gt 3 ]]; then
    echo "Usage: $0 MODEL_PATH [SOURCE_ROOT] [OUTPUT_ROOT]"
    exit 1
fi

model_path=$1
source_root=${2:-$(git rev-parse --show-toplevel)}
output_root=${3:-${source_root}/benchmark-artifacts/direct-self-play-topology-sweep}
benchmark_runner="${source_root}/py/tools/run_self_play_cpp_stochastic_benchmark.sh"

run_case() {
    local case_name=$1
    local processes_per_gpu=$2
    local mcts_threads=$3
    local games_per_process=$4
    local parallel_searches=$5
    local direct_workers=$6
    local direct_batch_size=$7
    local outstanding_batches=$8
    local seed_base=$9

    echo "Running ${case_name}"
    env \
        GPU_COUNT=4 \
        PROCESSES_PER_GPU="${processes_per_gpu}" \
        MCTS_THREADS_PER_PROCESS="${mcts_threads}" \
        PARALLEL_GAMES_PER_PROCESS="${games_per_process}" \
        WARMUP_STEPS=1 \
        MEASUREMENT_DURATION_SECONDS=30 \
        SEARCHES_PER_PLY=600 \
        FAST_SEARCHES_PER_PLY=150 \
        PARALLEL_SEARCHES="${parallel_searches}" \
        MAXIMUM_BATCH_SIZE=256 \
        INFERENCE_TIMEOUT_MICROSECONDS=500 \
        CACHE_CAPACITY=1 \
        USE_INFERENCE_CACHE=0 \
        DIRECT_INFERENCE_WORKERS="${direct_workers}" \
        DIRECT_INFERENCE_BATCH_SIZE="${direct_batch_size}" \
        DIRECT_OUTSTANDING_BATCHES_PER_WORKER="${outstanding_batches}" \
        SEED_BASE="${seed_base}" \
        PIN_WORKERS_TO_CPUS=0 \
        MAXIMUM_HOST_RAM_PERCENT=95 \
        STOCHASTIC_BENCHMARK_OUTPUT_ROOT="${output_root}/${case_name}" \
        bash "${benchmark_runner}" "${model_path}" "${source_root}"
}

# Current production topology, rebuilt with exact completed-search accounting.
run_case legacy-p10-t3-g96-p4 10 3 96 4 0 64 1 1000

# Direct candidates. Every direct search uses one outstanding leaf per tree.
run_case direct-p1-w2-g512-b64 1 1 512 1 2 64 1 2000
run_case direct-p2-w2-g512-b64 2 1 512 1 2 64 1 3000
run_case direct-p2-w3-g768-b64 2 1 768 1 3 64 1 4000
run_case direct-p3-w2-g512-b64 3 1 512 1 2 64 1 5000
