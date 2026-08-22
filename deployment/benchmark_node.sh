#!/usr/bin/env bash
# Node comparison benchmark: hardware facts, inference, self-play search throughput, trainer throughput.
# Run ON the node, from the repository root, inside the locked venv, after deployment/setup_remote.sh has built
# the Release extension. Results land in ${BENCHMARK_OUTPUT_ROOT:-/workspace/node-benchmark-<timestamp>}.
#
#   NODE_LABEL=rtx3090 HOURLY_PRICE=0.21 bash deployment/benchmark_node.sh
#
# Every measurement is repeated for two reference networks built from the committed production configs:
#   cnn   = py/configs/production/vast-chess-8gpu-cnn-reference.yaml   (CNN 12x144, 76-plane head)
#   att   = py/configs/production/vast-chess-8gpu-optimal.yaml         (attention 8x128, first progressive model)
# Throughput ratios between nodes are what matter; absolute numbers will change with the WP1 head rework.
set -euo pipefail

label=${NODE_LABEL:?set NODE_LABEL, e.g. rtx3090}
price=${HOURLY_PRICE:?set HOURLY_PRICE in USD per hour for the whole instance}
root=$(git rev-parse --show-toplevel)
stamp=$(date -u +%Y%m%dT%H%M%SZ)
out=${BENCHMARK_OUTPUT_ROOT:-/workspace/node-benchmark-${label}-${stamp}}
mkdir -p "${out}"
python=${PYTHON_BINARY:-python3}
gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
cpus=$(nproc)
duration=${MEASUREMENT_DURATION_SECONDS:-60}

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "${out}/benchmark.log"; }

# 1. Hardware facts -------------------------------------------------------------------------------------------
log "hardware facts"
{
  echo "label=${label}"; echo "hourly_price=${price}"; echo "revision=$(git rev-parse HEAD)"
  echo "gpu_count=${gpu_count}"; echo "nproc=${cpus}"
  echo "cpu_quota=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo unknown)"
  echo "mem_total_gib=$(awk '/MemTotal/ {printf "%.1f", $2/1048576}' /proc/meminfo)"
  echo "mem_limit=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo unknown)"
  echo "disk_free=$(df -h /workspace 2>/dev/null | tail -1 | awk '{print $4}')"
  lscpu | grep -E 'Model name|Socket|Thread|Core|NUMA' || true
  nvidia-smi --query-gpu=index,name,memory.total,driver_version,pcie.link.gen.current,pcie.link.width.current,clocks.max.sm --format=csv
  nvidia-smi topo -m 2>/dev/null || true
  "${python}" -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "cudnn", torch.backends.cudnn.version())'
} >"${out}/hardware.txt" 2>&1
cat "${out}/hardware.txt" | tee -a "${out}/benchmark.log"

# idle check: refuse to benchmark a loaded node
busy=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | awk '$1>5' | wc -l)
if [[ "${busy}" -gt 0 ]]; then log "WARNING: ${busy} GPU(s) not idle; results will be pessimistic"; fi
# disk + memory bandwidth quick probes
( dd if=/dev/zero of="${out}/dd.tmp" bs=1M count=2048 oflag=direct 2>&1 | tail -1; rm -f "${out}/dd.tmp" ) >"${out}/disk.txt" 2>&1 || true
"${python}" - <<'EOF' >"${out}/host_bandwidth.txt" 2>&1 || true
import time, numpy as np
a=np.ones(256<<20,dtype=np.uint8); t=time.perf_counter(); b=a.copy(); dt=time.perf_counter()-t
print(f"host_memcpy_gib_s={0.25/dt:.1f}")
import torch
if torch.cuda.is_available():
    x=torch.empty(256<<20,dtype=torch.uint8).pin_memory(); y=torch.empty_like(x,device='cuda')
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(8): y.copy_(x,non_blocking=True)
    torch.cuda.synchronize(); dt=time.perf_counter()-t; print(f"h2d_pinned_gib_s={2.0/dt:.1f}")
    t=time.perf_counter()
    for _ in range(8): x.copy_(y,non_blocking=True)
    torch.cuda.synchronize(); dt=time.perf_counter()-t; print(f"d2h_pinned_gib_s={2.0/dt:.1f}")
EOF
cat "${out}/disk.txt" "${out}/host_bandwidth.txt" | tee -a "${out}/benchmark.log"

# 2. Reference models ----------------------------------------------------------------------------------------
cd "${root}/py"; export PYTHONPATH=.
declare -A cfg=( [cnn]=configs/production/vast-chess-8gpu-cnn-reference.yaml [att]=configs/production/vast-chess-8gpu-optimal.yaml )
for m in cnn att; do
  log "prepare model ${m}"
  "${python}" tools/prepare_benchmark_model.py --run-config "${cfg[$m]}" --output "${out}/model-${m}.jit.pt" --seed 0 \
    >"${out}/prepare-${m}.log" 2>&1
done

# 3. Raw inference latency/throughput per GPU (batch 1/16/64/256, scripted) ----------------------------------
for g in $(seq 0 $((gpu_count-1))); do
  log "inference benchmark gpu ${g}"
  "${python}" tools/benchmark_chess_inference.py --gpu-id "${g}" --output "${out}/inference-gpu${g}.json" \
    --batch-sizes 1,16,64,256 --duration-seconds 5 --acknowledge-gpu-load >"${out}/inference-gpu${g}.log" 2>&1 || \
    log "inference benchmark failed on gpu ${g} (see log)"
done

# 4. Self-play search throughput (the number that matters) ---------------------------------------------------
# processes per GPU x parallel_searches grid; 512 games per process; generation 0 (200 full visits, 50 fast).
cd "${root}"
for m in cnn att; do
  for ppg in 2 3; do
    for ps in 1 4; do
      log "self-play ${m} processes_per_gpu=${ppg} parallel_searches=${ps}"
      BENCHMARK_OUTPUT_ROOT="${out}/selfplay" GPU_COUNT="${gpu_count}" PROCESSES_PER_GPU="${ppg}" \
      PARALLEL_GAMES_PER_PROCESS=512 MEASUREMENT_DURATION_SECONDS="${duration}" WARMUP_BATCHES=2 \
      CPUS_PER_PROCESS=$(( cpus / (gpu_count*ppg) > 0 ? cpus / (gpu_count*ppg) : 1 )) PARALLEL_SEARCHES="${ps}" \
      bash py/tools/run_self_play_search_benchmark.sh "${out}/model-${m}.jit.pt" "py/${cfg[$m]}" \
        >"${out}/selfplay-${m}-ppg${ppg}-ps${ps}.log" 2>&1 || log "self-play benchmark failed (see log)"
      # tag the newest summary with the grid point
      latest=$(ls -td "${out}"/selfplay/self-play-search-* 2>/dev/null | head -1)
      [[ -n "${latest}" ]] && cp "${latest}/summary.json" "${out}/selfplay-${m}-ppg${ppg}-ps${ps}.json"
    done
  done
done

# 5. Trainer throughput (fixed batch, production objective, bf16, compiled as configured) --------------------
# benchmark_training_overfit needs an existing run directory with a replay.bin; without one the stage cannot
# produce a number and must say so instead of silently emitting nothing.
if [[ -z "${TRAINER_RUN_DIRECTORY:-}" ]]; then
  log "TRAINER_RUN_DIRECTORY is not set; skipping the trainer stage (needs a run directory with replay.bin)"
else
  [[ -f "${TRAINER_RUN_DIRECTORY}/replay.bin" ]] || { log "ERROR: ${TRAINER_RUN_DIRECTORY}/replay.bin does not exist"; exit 1; }
  cd "${root}/py"
  for m in cnn att; do
    log "trainer throughput ${m}"
    "${python}" -c "
from pathlib import Path
from src.experiment.configuration import load_experiment_configuration, write_resolved_experiment
write_resolved_experiment(Path('${out}/resolved-${m}.json'), load_experiment_configuration(Path('${cfg[$m]}')))
" >"${out}/resolve-${m}.log" 2>&1 || { log "ERROR: could not resolve ${m} (see log)"; exit 1; }
    "${python}" tools/benchmark_training_overfit.py --resolved-configuration "${out}/resolved-${m}.json" \
      --run-directory "${TRAINER_RUN_DIRECTORY}" --device-id 0 --batch-size 256 --maximum-optimizer-steps 300 \
      --report-interval 25 --model-generation 0 --output-path "${out}/trainer-${m}.json" >"${out}/trainer-${m}.log" 2>&1 || \
      { log "ERROR: trainer benchmark failed for ${m} (see log)"; exit 1; }
  done
fi

# 6. Summary -------------------------------------------------------------------------------------------------
"${python}" - "${out}" "${label}" "${price}" "${gpu_count}" <<'EOF' | tee "${out}/SUMMARY.md"
import json, sys, glob, os
out, label, price, gpus = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
print(f"# Node benchmark {label} — {gpus} GPU(s), ${price}/h instance\n")
print("| model | procs/GPU | parallel_searches | searches/s (node) | searches/s per GPU | searches per $ (M/h) | avg batch |")
print("|---|---:|---:|---:|---:|---:|---:|")
for f in sorted(glob.glob(f"{out}/selfplay-*.json")):
    d = json.load(open(f)); m = d["measurement"]; name = os.path.basename(f)[9:-5]
    model, ppg, ps = name.split("-")[0], name.split("ppg")[1].split("-")[0], name.split("ps")[1]
    sps = m.get("searches_per_second", 0); batch = d.get("inference", {}).get("average_batch_size", "")
    print(f"| {model} | {ppg} | {ps} | {sps:,.0f} | {sps/gpus:,.0f} | {sps*3600/price/1e6:,.1f} | {batch} |")
print()
for f in sorted(glob.glob(f"{out}/trainer-*.json")):
    try:
        d = json.load(open(f)); print(f"trainer {os.path.basename(f)}: ", {k: d[k] for k in d if 'per_second' in k or 'steps' in k})
    except Exception as e: print(f"trainer {f}: unreadable ({e})")
for f in sorted(glob.glob(f"{out}/inference-gpu*.json")):
    try:
        d = json.load(open(f)); print(f"inference {os.path.basename(f)}: ", json.dumps(d)[:600])
    except Exception as e: print(f"inference {f}: unreadable ({e})")
EOF
log "done: ${out}"
