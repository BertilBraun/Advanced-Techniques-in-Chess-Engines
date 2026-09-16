#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 5 ]]; then
    echo "Usage: v34_qat_replay_campaign.sh REPLAY_ROOT REPLAY_EXPERIMENT REPLAY_SHA256 INITIAL_MODEL OUTPUT_ROOT" >&2
    exit 2
fi

export V34_QAT_REPLAY_ROOT="$1"
export V34_QAT_REPLAY_EXPERIMENT="$2"
export V34_QAT_REPLAY_SHA256="$3"
export V34_QAT_INITIAL_MODEL="$4"
output_root="$5"
repository="${ENGINE_REPOSITORY_DIRECTORY:?ENGINE_REPOSITORY_DIRECTORY must identify the prepared worktree}"
arm_script="${repository}/deployment/v34_qat_replay_arm.sh"
training_steps="${V34_QAT_CAMPAIGN_STEPS:-2000}"

mkdir -p "${output_root}"

V34_QAT_STEPS=0 "${arm_script}" post_qat 0 "${output_root}/ptq-only" \
    >"${output_root}/ptq-only.log" 2>&1
test -f "${output_root}/ptq-only/report.json"

pids=()
for definition in 'post_float:0:fp-control' 'post_qat:1:qat-per-block' 'post_shared_qat:2:qat-shared-trunk'; do
    IFS=: read -r cell gpu_id output_name <<<"${definition}"
    V34_QAT_STEPS="${training_steps}" "${arm_script}" "${cell}" "${gpu_id}" "${output_root}/${output_name}" \
        >"${output_root}/${output_name}.log" 2>&1 &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        failed=1
    fi
done
if [[ "${failed}" -ne 0 ]]; then
    echo 'At least one V34 replay diagnostic failed.' >&2
    exit 1
fi

for report in \
    "${output_root}/fp-control/report.json" \
    "${output_root}/qat-per-block/report.json" \
    "${output_root}/qat-shared-trunk/report.json"; do
    test -f "${report}"
done

"${ENGINE_VIRTUAL_ENVIRONMENT:-/workspace/alphazero-engine-venv}/bin/python" -c \
    'import json, pathlib, sys; root = pathlib.Path(sys.argv[1]); names = ("ptq-only", "fp-control", "qat-per-block", "qat-shared-trunk"); payload = {name: json.loads((root / name / "report.json").read_text(encoding="utf-8")) for name in names}; (root / "campaign-summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")' \
    "${output_root}"
