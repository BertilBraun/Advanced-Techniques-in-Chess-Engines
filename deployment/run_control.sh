#!/usr/bin/env bash
# Run-control interface and evidence preservation (recovery plan WP8).
#
# The only supported way to start, stop, inspect and archive a production or test run:
#
#   run_control.sh start <config.yaml>            validate approval + clean checkout, install and start the
#                                                  supervisor program, print run name / TensorBoard / state paths
#   run_control.sh stop <run-name>                 touch the manual stop file, wait for a checkpoint-safe exit,
#                                                  then preserve
#   run_control.sh status <run-name>               supervisor state, last generation, last evaluation result,
#                                                  inbox/staging depth, GPU utilisation
#   run_control.sh preserve <run-name>             archive TensorBoard, run-state manifests, resolved config,
#                                                  credit ledger, run log and evaluation results into
#                                                  .codex-diagnostics/<run-name>-<UTC timestamp>/ with SHA256SUMS;
#                                                  idempotent
#   run_control.sh fetch <run-name> <local-dir>    from the workstation: rsync the node archives into a local
#                                                  .codex-diagnostics/ and verify SHA256SUMS
#
# A test run without a fetched .codex-diagnostics archive did not happen.
# See documentation/operations/run-control.md.

set -euo pipefail

script_path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
repository_directory="${ENGINE_REPOSITORY_DIRECTORY:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
virtual_environment="${ENGINE_VIRTUAL_ENVIRONMENT:-${repository_directory}-venv}"
run_control_root="${RUN_CONTROL_ROOT:-/workspace/run-control}"
approval_directory="${RUN_CONTROL_APPROVAL_DIRECTORY:-/workspace/approvals}"
tensorboard_root="${RUN_CONTROL_TENSORBOARD_ROOT:-/workspace/tensorboard}"
archive_root="${RUN_CONTROL_ARCHIVE_ROOT:-${repository_directory}/.codex-diagnostics}"
supervisor_conf_dir="${RUN_CONTROL_SUPERVISOR_CONF_DIR:-/etc/supervisor/conf.d}"
stop_wait_seconds="${RUN_CONTROL_STOPWAITSECS:-600}"
stop_timeout_seconds="${RUN_CONTROL_STOP_TIMEOUT_SECONDS:-1800}"
registry_directory="${run_control_root}/runs"

usage() {
    echo "Usage: run_control.sh start <config.yaml> | stop <run-name> | status <run-name>" >&2
    echo "                      | preserve <run-name> | fetch <run-name> <local-dir>" >&2
    exit 2
}

fail() {
    echo "run_control: $*" >&2
    exit 1
}

registry_path() {
    echo "${registry_directory}/$1.env"
}

load_registry() {
    local run_name="$1"
    local path
    path="$(registry_path "${run_name}")"
    [[ -f "${path}" ]] || fail "unknown run '${run_name}': ${path} does not exist (was it started through run_control.sh?)"
    # shellcheck source=/dev/null
    source "${path}"
}

venv_python() {
    echo "${virtual_environment}/bin/python"
}

# ---------------------------------------------------------------- start ----

describe_configuration() {
    local config_path="$1"
    (
        cd "${repository_directory}/py"
        PYTHONPATH=. "$(venv_python)" - "${config_path}" <<'PYTHON'
import sys
from pathlib import Path

from src.experiment.configuration import experiment_configuration_sha256, load_experiment_configuration

experiment = load_experiment_configuration(Path(sys.argv[1]).resolve())
training = experiment.training
if training.limits.manual_stop_file is None:
    raise SystemExit('run_control requires training.limits.manual_stop_file to be configured.')
print(f'RUN_NAME={experiment.run.run_name}')
print(f'TENSORBOARD_RUN_DIRECTORY={experiment.run.tensorboard_run_directory}')
print(f'SAVE_PATH={training.save_path}')
print(f'STOP_FILE={training.limits.manual_stop_file}')
print(f'RUNTIME_IMAGE={experiment.run.environment.runtime_image}')
print(f'OPEN_FILE_SOFT_LIMIT={experiment.run.environment.minimum_open_file_soft_limit}')
print(f'CONFIGURATION_SHA256={experiment_configuration_sha256(experiment)}')
PYTHON
    )
}

validate_approval() {
    local approval_file="$1" configuration_sha256="$2" source_revision="$3"
    [[ -f "${approval_file}" ]] || fail "approval file does not exist: ${approval_file}"
    "$(venv_python)" - "${approval_file}" "${configuration_sha256}" "${source_revision}" <<'PYTHON'
import json
import sys

approval = json.load(open(sys.argv[1], encoding='utf-8'))
expected = {'configuration_sha256': sys.argv[2], 'source_revision': sys.argv[3]}
for key, value in expected.items():
    if approval.get(key) != value:
        raise SystemExit(f'approval mismatch on {key}: approval has {approval.get(key)!r}, run has {value!r}')
PYTHON
}

command_start() {
    local config_argument="$1"
    local config_path
    config_path="$(cd "$(dirname "${config_argument}")" && pwd)/$(basename "${config_argument}")"
    [[ -f "${config_path}" ]] || fail "run configuration does not exist: ${config_path}"

    command -v supervisorctl >/dev/null || fail "supervisorctl is not installed on this node"
    [[ -d "${supervisor_conf_dir}" ]] || fail "supervisor configuration directory does not exist: ${supervisor_conf_dir}"

    [[ -z "$(git -C "${repository_directory}" status --porcelain)" ]] \
        || fail "refusing to start from a dirty source working tree: $(git -C "${repository_directory}" status --short | head -5)"
    local source_revision
    source_revision="$(git -C "${repository_directory}" rev-parse HEAD)"

    local description
    description="$(describe_configuration "${config_path}")"
    local run_name tensorboard_run_directory save_path stop_file runtime_image configuration_sha256 open_file_soft_limit
    run_name="$(echo "${description}" | sed -n 's/^RUN_NAME=//p')"
    tensorboard_run_directory="$(echo "${description}" | sed -n 's/^TENSORBOARD_RUN_DIRECTORY=//p')"
    save_path="$(echo "${description}" | sed -n 's/^SAVE_PATH=//p')"
    stop_file="$(echo "${description}" | sed -n 's/^STOP_FILE=//p')"
    runtime_image="$(echo "${description}" | sed -n 's/^RUNTIME_IMAGE=//p')"
    open_file_soft_limit="$(echo "${description}" | sed -n 's/^OPEN_FILE_SOFT_LIMIT=//p')"
    configuration_sha256="$(echo "${description}" | sed -n 's/^CONFIGURATION_SHA256=//p')"
    [[ -n "${run_name}" && -n "${configuration_sha256}" ]] || fail "could not resolve the run configuration"
    # The registry below single-quotes these values for later `source`; a quote would corrupt it.
    [[ "${run_name}${config_path}${save_path}${runtime_image}" != *"'"* ]] \
        || fail "run names and paths must not contain single quotes"
    [[ "${save_path}" == /* ]] || save_path="${repository_directory}/${save_path}"

    local approval_file
    approval_file="${approval_directory}/$(basename "${config_path%.*}").json"
    validate_approval "${approval_file}" "${configuration_sha256}" "${source_revision}"

    [[ -e "${stop_file}" ]] && fail "stale manual stop file exists: ${stop_file} — inspect and remove it before starting"

    mkdir -p "${registry_directory}" "${run_control_root}/logs/${run_name}" "$(dirname "${stop_file}")" "${tensorboard_root}"
    local registry
    registry="$(registry_path "${run_name}")"
    if [[ -e "${registry}" ]]; then
        echo "run_control: registry entry already exists and is replaced: ${registry}" >&2
    fi
    cat >"${registry}" <<REGISTRY
RUN_NAME='${run_name}'
RUN_CONFIG='${config_path}'
SAVE_PATH='${save_path}'
TENSORBOARD_DIRECTORY='${tensorboard_root}/${tensorboard_run_directory}'
LOG_DIRECTORY='${run_control_root}/logs/${run_name}'
STOP_FILE='${stop_file}'
APPROVAL_FILE='${approval_file}'
RUNTIME_IMAGE='${runtime_image}'
OPEN_FILE_SOFT_LIMIT='${open_file_soft_limit}'
SOURCE_REVISION='${source_revision}'
CONFIGURATION_SHA256='${configuration_sha256}'
CREATED_AT_UTC='$(date -u +%Y-%m-%dT%H:%M:%SZ)'
REGISTRY

    cat >"${supervisor_conf_dir}/${run_name}.conf" <<SUPERVISOR
[program:${run_name}]
command=/bin/bash ${script_path} supervised-runner ${run_name}
directory=${repository_directory}
autostart=false
autorestart=unexpected
exitcodes=0
startsecs=30
stopsignal=TERM
stopwaitsecs=${stop_wait_seconds}
stopasgroup=false
killasgroup=true
stdout_logfile=${run_control_root}/logs/${run_name}/supervisor-stdout.log
stderr_logfile=${run_control_root}/logs/${run_name}/supervisor-stderr.log
environment=ENGINE_REPOSITORY_DIRECTORY="${repository_directory}",ENGINE_VIRTUAL_ENVIRONMENT="${virtual_environment}",RUN_CONTROL_ROOT="${run_control_root}",RUN_CONTROL_TENSORBOARD_ROOT="${tensorboard_root}",RUN_CONTROL_ARCHIVE_ROOT="${archive_root}",RUN_CONTROL_APPROVAL_DIRECTORY="${approval_directory}"
SUPERVISOR

    supervisorctl reread >/dev/null
    supervisorctl update >/dev/null
    supervisorctl start "${run_name}"

    echo "run name:              ${run_name}"
    echo "tensorboard directory: ${tensorboard_root}/${tensorboard_run_directory}"
    echo "state directory:       ${save_path}"
}

# ------------------------------------------------------ supervised runner ----

command_supervised_runner() {
    local run_name="$1"
    load_registry "${run_name}"

    ulimit -Sn "${OPEN_FILE_SOFT_LIMIT}" \
        || echo "run_control: could not raise the open-file soft limit to ${OPEN_FILE_SOFT_LIMIT}" >&2

    local python_nvidia_library_path
    python_nvidia_library_path="$(
        "$(venv_python)" -c \
            'from pathlib import Path; import site; root = Path(site.getsitepackages()[0]) / "nvidia"; print(":".join(str(path) for path in sorted(root.glob("*/lib")) if path.is_dir()))'
    )"
    if [[ -n "${python_nvidia_library_path}" ]]; then
        export LD_LIBRARY_PATH="${python_nvidia_library_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
    export PATH="${virtual_environment}/bin:${PATH}"
    export TRAINING_RUNTIME_IMAGE="${RUNTIME_IMAGE}"
    export TRAINING_TENSORBOARD_LOG_PATH="${tensorboard_root}"
    export TRAINING_LOG_PATH="${LOG_DIRECTORY}"

    # A cancelled run must still be archived: preserve on every exit path.
    preserve_on_exit() {
        command_preserve "${run_name}" || echo "run_control: preservation on exit failed for ${run_name}" >&2
    }
    trap preserve_on_exit EXIT

    local child_pid=0
    request_stop() {
        echo "run_control: termination requested; touching ${STOP_FILE} for a checkpoint-safe stop" >&2
        touch "${STOP_FILE}"
    }
    trap request_stop TERM INT

    cd "${repository_directory}"
    "$(venv_python)" py/run_approved_experiment.py \
        --approval-directory "${approval_directory}" \
        --run-config "${RUN_CONFIG}" &
    child_pid=$!
    local exit_status=0
    while kill -0 "${child_pid}" 2>/dev/null; do
        wait "${child_pid}" && exit_status=0 || exit_status=$?
    done
    return "${exit_status}"
}

# ----------------------------------------------------------------- stop ----

command_stop() {
    local run_name="$1"
    load_registry "${run_name}"

    echo "run_control: requesting checkpoint-safe stop via ${STOP_FILE}"
    touch "${STOP_FILE}"

    local waited=0
    while true; do
        local state
        state="$(supervisorctl status "${run_name}" 2>/dev/null | awk '{print $2}')"
        case "${state}" in
            RUNNING|STOPPING|STARTING)
                if (( waited >= stop_timeout_seconds )); then
                    echo "run_control: run did not exit within ${stop_timeout_seconds}s; preserving anyway." >&2
                    echo "run_control: escalate manually with 'supervisorctl stop ${run_name}' if needed." >&2
                    command_preserve "${run_name}"
                    exit 1
                fi
                sleep 10
                waited=$((waited + 10))
                ;;
            *)
                echo "run_control: supervisor state is '${state:-gone}' after ${waited}s"
                break
                ;;
        esac
    done
    command_preserve "${run_name}"
}

# --------------------------------------------------------------- status ----

command_status() {
    local run_name="$1"
    load_registry "${run_name}"

    echo "== supervisor =="
    supervisorctl status "${run_name}" || true

    echo "== progress =="
    "$(venv_python)" - "${SAVE_PATH}" <<'PYTHON'
import json
import sys
from pathlib import Path

save_path = Path(sys.argv[1])
ledger_path = save_path / 'credit-ledger.json'
resolved_path = save_path / 'resolved-experiment.json'
if ledger_path.exists() and resolved_path.exists():
    ledger = json.loads(ledger_path.read_text(encoding='utf-8'))
    resolved = json.loads(resolved_path.read_text(encoding='utf-8'))
    steps_per_quantum = resolved['training']['lifecycle']['credit']['optimizer_steps_per_quantum']
    steps = ledger['completed_optimizer_steps']
    print(f'generation:            {steps // steps_per_quantum} ({steps} optimizer steps)')
    print(f'available credits:     {float(ledger["earned_credits"]) - float(ledger["consumed_credits"]):.0f}')
else:
    print('no credit ledger / resolved configuration yet')
outcome_path = save_path / 'run-outcome.json'
if outcome_path.exists():
    outcome = json.loads(outcome_path.read_text(encoding='utf-8'))
    print(f'run outcome:           {outcome.get("status")} ({outcome.get("stop_reason")})')

evaluations = save_path / 'evaluations'
results = (
    [
        path
        for path in evaluations.glob('*.json')
        if path.name not in ('manager-state.json', 'reference-checkpoints.json')
    ]
    if evaluations.is_dir()
    else []
)
if results:
    latest = max(results, key=lambda path: path.stat().st_mtime)
    result = json.loads(latest.read_text(encoding='utf-8'))
    job = result.get('job', {})
    definition = job.get('definition', {}).get('definition_id', '?')
    generation = job.get('candidate', {}).get('generation', '?')
    aggregate = result.get('aggregate') or {}
    detail = (
        f'score={aggregate.get("score")} W/D/L={aggregate.get("wins")}/{aggregate.get("draws")}/{aggregate.get("losses")}'
        if aggregate
        else f'accuracy={result.get("top_action_accuracy")} cross_entropy={result.get("policy_cross_entropy")}'
    )
    print(f'last evaluation:       {definition} generation {generation}: {detail} ({latest.name})')
else:
    print('last evaluation:       none yet')

for stage in ('inbox', 'staging'):
    directory = save_path / 'completed-games' / stage
    count = sum(1 for _ in directory.iterdir()) if directory.is_dir() else 0
    print(f'{stage} depth:'.ljust(23) + str(count))
PYTHON

    echo "== gpus =="
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null \
        || echo "nvidia-smi unavailable"
}

# ------------------------------------------------------------- preserve ----

archive_copy() {
    local source="$1" destination="$2"
    if [[ -d "${source}" ]]; then
        mkdir -p "${destination}"
        cp -a "${source}/." "${destination}/"
    elif [[ -f "${source}" ]]; then
        mkdir -p "$(dirname "${destination}")"
        cp -a "${source}" "${destination}"
    fi
}

command_preserve() {
    local run_name="$1"
    load_registry "${run_name}"

    mkdir -p "${archive_root}"
    local timestamp partial
    timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
    partial="${archive_root}/.partial-${run_name}-${timestamp}"
    rm -rf "${partial}"
    mkdir -p "${partial}"

    archive_copy "${TENSORBOARD_DIRECTORY}" "${partial}/tensorboard"
    archive_copy "${LOG_DIRECTORY}" "${partial}/logs"
    archive_copy "${RUN_CONFIG}" "${partial}/config/$(basename "${RUN_CONFIG}")"
    archive_copy "${APPROVAL_FILE}" "${partial}/config/$(basename "${APPROVAL_FILE}")"
    archive_copy "$(registry_path "${run_name}")" "${partial}/config/${run_name}.env"
    local artifact
    for artifact in run_manifest.json run_manifests resolved-experiment.json run-outcome.json \
        resource-telemetry.jsonl credit-ledger.json evaluations; do
        archive_copy "${SAVE_PATH}/${artifact}" "${partial}/run/${artifact}"
    done
    local checkpoint_manifest
    for checkpoint_manifest in "${SAVE_PATH}"/checkpoint_*.json; do
        [[ -f "${checkpoint_manifest}" ]] && archive_copy "${checkpoint_manifest}" "${partial}/run/$(basename "${checkpoint_manifest}")"
    done

    (
        cd "${partial}"
        find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS
        sha256sum --check --quiet SHA256SUMS
    )

    # Idempotence: if the newest existing archive of this run has identical content, keep it and discard this one.
    local newest
    newest="$(find "${archive_root}" -maxdepth 1 -type d -name "${run_name}-*" | sort | tail -1)"
    if [[ -n "${newest}" && -f "${newest}/SHA256SUMS" ]] && cmp -s "${newest}/SHA256SUMS" "${partial}/SHA256SUMS"; then
        rm -rf "${partial}"
        echo "run_control: archive is up to date: ${newest}"
        return 0
    fi

    local destination="${archive_root}/${run_name}-${timestamp}"
    mv "${partial}" "${destination}"
    echo "run_control: preserved ${destination} ($(find "${destination}" -type f | wc -l) files)"
}

# ---------------------------------------------------------------- fetch ----

command_fetch() {
    local run_name="$1" local_directory="$2"
    local destination="${local_directory%/}/.codex-diagnostics"
    local ssh_destination="${RUN_CONTROL_SSH_DESTINATION:?set RUN_CONTROL_SSH_DESTINATION, e.g. root@HOST}"
    local ssh_port="${RUN_CONTROL_SSH_PORT:-22}"
    local remote_archive_root="${RUN_CONTROL_REMOTE_ARCHIVE_ROOT:-/workspace/alphazero-engine/.codex-diagnostics}"
    local ssh_command="ssh -p ${ssh_port}"
    if [[ -n "${RUN_CONTROL_SSH_KEY:-}" ]]; then
        ssh_command="${ssh_command} -i ${RUN_CONTROL_SSH_KEY}"
    fi

    mkdir -p "${destination}"
    rsync -a --info=stats1 -e "${ssh_command}" \
        "${ssh_destination}:${remote_archive_root}/${run_name}-*" "${destination}/"

    local archive verified=0
    for archive in "${destination}/${run_name}"-*/; do
        [[ -d "${archive}" ]] || continue
        (
            cd "${archive}"
            sha256sum --check --quiet SHA256SUMS
        ) || fail "SHA256SUMS verification failed in ${archive}"
        echo "run_control: verified ${archive}"
        verified=$((verified + 1))
    done
    (( verified > 0 )) || fail "no archive for '${run_name}' was fetched from ${remote_archive_root}"
}

# ----------------------------------------------------------------- main ----

[[ $# -ge 1 ]] || usage
command="$1"
shift
case "${command}" in
    start) [[ $# -eq 1 ]] || usage; command_start "$1" ;;
    stop) [[ $# -eq 1 ]] || usage; command_stop "$1" ;;
    status) [[ $# -eq 1 ]] || usage; command_status "$1" ;;
    preserve) [[ $# -eq 1 ]] || usage; command_preserve "$1" ;;
    fetch) [[ $# -eq 2 ]] || usage; command_fetch "$1" "$2" ;;
    supervised-runner) [[ $# -eq 1 ]] || usage; command_supervised_runner "$1" ;;
    *) usage ;;
esac
