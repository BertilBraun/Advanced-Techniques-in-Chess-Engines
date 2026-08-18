# Second four-day chess run readiness

This run uses the canonical configuration
`py/configs/production/vast-chess-8gpu-optimal.yaml`. It is a single approved experiment, not a queue campaign. The
configuration owns the four-day wall-time cap, the RTX 3090 offer contract, output paths, progressive models, search
schedules, replay, auxiliary objectives, evaluation, and late-cap Syzygy adjudication.

## Revision and approval ownership

After the final `master` revision is pushed, record its full 40-character SHA as `APPROVED_REVISION`. Do not use a
branch name in the approval. On the selected node, require:

```bash
test "$(git -C /workspace/alphazero-engine rev-parse HEAD)" = "${APPROVED_REVISION}"
test -z "$(git -C /workspace/alphazero-engine status --porcelain)"
```

Create `/workspace/approvals/vast-chess-8gpu-optimal.json` outside Git only after the node hardware and exact revision
are fixed. The approval must contain:

```json
{
  "approved_by": "USER_NAME",
  "approved_at_utc": "UTC_TIMESTAMP_WITH_OFFSET",
  "run_name": "vast-chess-8gpu-optimal",
  "source_revision": "APPROVED_REVISION",
  "configuration_sha256": "RESOLVED_CONFIGURATION_SHA256",
  "provider_name": "vast.ai",
  "offer_id": "instance-48042270-machine-79780-host-399360",
  "hourly_price": 1.1244444,
  "maximum_cost": null,
  "maximum_wall_time_minutes": 5760
}
```

Compute `RESOLVED_CONFIGURATION_SHA256` with the locked environment from the exact checkout:

```bash
cd /workspace/alphazero-engine/py
/workspace/alphazero-engine-venv/bin/python -c "from pathlib import Path; from src.experiment.configuration import experiment_configuration_sha256, load_experiment_configuration; print(experiment_configuration_sha256(load_experiment_configuration(Path('configs/production/vast-chess-8gpu-optimal.yaml'))))"
```

The user owns approval creation. The repository owns configuration and validation. The runner validates both against
the clean checkout before creating any run state.

## Provision, verify, then launch later

Read `/etc/vast-agents-guide.md`, record the hardware/runtime facts required by
[`experiment-platform.md`](experiment-platform.md), and provision without starting training:

```bash
export ENGINE_REPOSITORY_REF=master
export ENGINE_REPOSITORY_DIRECTORY=/workspace/alphazero-engine
export ENGINE_VIRTUAL_ENVIRONMENT=/workspace/alphazero-engine-venv
deployment/setup_remote.sh /bin/true
```

Before launch, verify the approved revision, CUDA KataGo smokes, `/workspace/syzygy/wdl345`, immutable chess evaluation
artifacts, an empty `/workspace/alphazero-engine/py/training_data/production/vast-chess-8gpu-optimal` output path, and
an empty TensorBoard run directory. Install a supervisor definition outside Git whose command is:

```bash
/workspace/alphazero-engine-venv/bin/python py/run_approved_experiment.py \
  --approval-directory /workspace/approvals \
  --run-config py/configs/production/vast-chess-8gpu-optimal.yaml
```

Run it from `/workspace/alphazero-engine` with the bootstrap `LD_LIBRARY_PATH`, `TRAINING_RUNTIME_IMAGE`, and open-file
limit preserved. Start that supervisor program only after the user separately authorizes launch. This readiness step
does not create an approval, provision a node, or start training.
