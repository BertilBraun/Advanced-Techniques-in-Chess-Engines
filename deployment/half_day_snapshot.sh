#!/usr/bin/env bash
# Archive the live run once per half day of evaluation time, from the two-day mark onwards.
#
# The grid is evaluation seconds, not optimizer steps, so a snapshot always lines up with a
# TensorBoard step and two runs can be compared at the same wall-clock point. Each snapshot is a
# plain run_control preserve, which already captures every stage's weights, optimizer, QAT state and
# ONNX without the replay store.
set -uo pipefail

RUN_CONTROL_ROOT="${RUN_CONTROL_ROOT:-/workspace/run-control}"
SNAPSHOT_MARKER_ROOT="${SNAPSHOT_MARKER_ROOT:-/workspace/half-day-snapshots}"
FIRST_EVALUATION_SECONDS="${FIRST_EVALUATION_SECONDS:-172800}"
SNAPSHOT_INTERVAL_SECONDS="${SNAPSHOT_INTERVAL_SECONDS:-43200}"
# One evaluation boundary of grace, so the evaluation at the threshold has been recorded first.
SNAPSHOT_GRACE_SECONDS="${SNAPSHOT_GRACE_SECONDS:-1200}"
POLL_SECONDS="${POLL_SECONDS:-300}"
PYTHON="${ENGINE_VIRTUAL_ENVIRONMENT:-/workspace/alphazero-engine-venv}/bin/python"

mkdir -p "${SNAPSHOT_MARKER_ROOT}"

running_run_name() {
    supervisorctl status 2>/dev/null | awk '$2 == "RUNNING" && $1 ~ /^vast-/ { print $1; exit }'
}

elapsed_evaluation_seconds() {
    local save_path="$1"
    "${PYTHON}" - "$save_path" <<'PY' 2>/dev/null
import json, sys
from pathlib import Path
state = Path(sys.argv[1]) / 'evaluations' / 'manager-state.json'
print(int(json.loads(state.read_text(encoding='utf-8'))['accumulated_elapsed_seconds']))
PY
}

while true; do
    run_name="$(running_run_name)"
    registry="${RUN_CONTROL_ROOT}/runs/${run_name}.env"
    if [[ -n "${run_name}" && -f "${registry}" ]]; then
        # shellcheck disable=SC1090
        source "${registry}"
        elapsed="$(elapsed_evaluation_seconds "${SAVE_PATH}")"
        if [[ -n "${elapsed}" ]] && ((elapsed >= FIRST_EVALUATION_SECONDS + SNAPSHOT_GRACE_SECONDS)); then
            crossed=$(((elapsed - SNAPSHOT_GRACE_SECONDS - FIRST_EVALUATION_SECONDS) / SNAPSHOT_INTERVAL_SECONDS))
            threshold=$((FIRST_EVALUATION_SECONDS + crossed * SNAPSHOT_INTERVAL_SECONDS))
            marker="${SNAPSHOT_MARKER_ROOT}/${threshold}"
            if [[ ! -f "${marker}" ]]; then
                echo "$(date -u +%FT%TZ) snapshotting ${run_name} at evaluation second ${threshold} (elapsed ${elapsed})"
                if bash "${ENGINE_REPOSITORY_DIRECTORY}/deployment/run_control.sh" preserve "${run_name}"; then
                    archive="$(ls -dt "${ENGINE_REPOSITORY_DIRECTORY}/.codex-diagnostics/${run_name}-"*/ 2>/dev/null | head -1)"
                    printf 'evaluation_seconds=%s\nrun_name=%s\nelapsed_at_snapshot=%s\narchive=%s\n' \
                        "${threshold}" "${run_name}" "${elapsed}" "${archive}" > "${marker}"
                    echo "$(date -u +%FT%TZ) snapshot recorded: ${archive}"
                else
                    echo "$(date -u +%FT%TZ) preserve failed for ${run_name}; will retry next poll"
                fi
            fi
        fi
    fi
    sleep "${POLL_SECONDS}"
done
