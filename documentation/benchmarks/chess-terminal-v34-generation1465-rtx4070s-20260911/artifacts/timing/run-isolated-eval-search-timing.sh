#!/usr/bin/env bash
set -euo pipefail

source_root=/workspace/alphazero-engine
python_binary=/workspace/alphazero-engine-venv/bin/python
run_directory=/workspace/alphazero-engine-v34-lr-001/py/training_data/production/vast-chess-8gpu-integrated-v34
experiment=${run_directory}/resolved-experiment.json
openings=${source_root}/py/reference/chess-elite-2025-11-balanced-4moves-200-v1-openings.json
output_root=/workspace/postrun/v34-terminal-g1465/isolated-evaluation-search-timing
script=${output_root}/isolated_eval_search_timing.py
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
output_directory=${output_root}/${timestamp}
mkdir -p "${output_directory}"
processes=()
for device in $(seq 0 7); do
    (
        cd "${source_root}/py"
        PYTHONPATH=. "${python_binary}" "${script}" \
            --device "${device}" \
            --experiment "${experiment}" \
            --run-directory "${run_directory}" \
            --generation 1465 \
            --openings "${openings}" \
            --position-count 50 \
            --searches 80000 \
            --warmup-searches 1024 \
            >"${output_directory}/gpu-${device}.json" \
            2>"${output_directory}/gpu-${device}.stderr"
    ) &
    processes+=("$!")
done
for process in "${processes[@]}"; do
    wait "${process}"
done

"${python_binary}" - "${output_directory}" <<'PYTHON'
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
workers = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(root.glob("gpu-*.json"))]
seconds = [worker["amortized_seconds_per_position"] for worker in workers]
summary = {
    "worker_count": len(workers),
    "total_positions": sum(worker["position_count"] for worker in workers),
    "searches_per_position": 80_000,
    "parallel_searches": 8,
    "inference_workers": 1,
    "inference_batch_size": 64,
    "outstanding_batches_per_worker": 1,
    "amortized_seconds_per_position_mean": statistics.mean(seconds),
    "amortized_seconds_per_position_median": statistics.median(seconds),
    "amortized_seconds_per_position_minimum": min(seconds),
    "amortized_seconds_per_position_maximum": max(seconds),
    "per_gpu": workers,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PYTHON

(
    cd "${output_directory}"
    find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS
    sha256sum --check --quiet SHA256SUMS
)
echo "${output_directory}"
