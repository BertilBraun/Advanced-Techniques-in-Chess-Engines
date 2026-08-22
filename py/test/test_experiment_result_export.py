from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from src.evaluation.contracts import ElapsedCheckpointReference, EvaluationReferenceManifest
from src.experiment.configuration import experiment_configuration_sha256, load_experiment_configuration
from src.experiment.progress_telemetry import RunOutcome, RunOutcomeStatus
from src.experiment.run import ExperimentRunManifest
from src.experiment.run_contract import ApprovalRecord, ResolvedHardware
from src.games.representation import NetworkDimensions
from src.experiment_queue.configuration import (
    QueueConfiguration,
    QueuedExperiment,
    ResourceRequest,
    ResourceSlot,
    RunnerCommand,
)
from src.experiment_queue.result_export import ExportRequest, export_experiment_results
from src.experiment_queue.scheduler import create_assignment
from src.experiment_queue.state import (
    CompletedExperimentStatus,
    ExecutionIdentity,
    FailedExperimentStatus,
    QueueSummary,
    RunningExperimentStatus,
    write_queue_summary,
)
from src.experiment_queue.validation import queue_configuration_fingerprint
from src.training.checkpoint import CheckpointManifest, CheckpointReference
from src.training.network import GoPointPassPolicyHeadConfiguration, NetworkDefinition, NetworkParams


NETWORK_DEFINITION = NetworkDefinition(
    architecture=NetworkParams(num_layers=1, hidden_size=8, policy_head=GoPointPassPolicyHeadConfiguration()),
    dimensions=NetworkDimensions(channels=3, rows=3, columns=3, actions=10),
    auxiliary_heads=(),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_checkpoint(run_path: Path, generation: int, include_optimizer: bool = True) -> CheckpointReference:
    model_path = run_path / f'model_{generation}.pt'
    optimizer_path = run_path / f'optimizer_{generation}.pt'
    inference_path = run_path / f'model_{generation}.jit.pt'
    model_path.write_bytes(f'model-{generation}'.encode())
    if include_optimizer:
        optimizer_path.write_bytes(f'optimizer-{generation}'.encode())
    inference_path.write_bytes(f'inference-{generation}'.encode())
    manifest = CheckpointManifest(
        generation=generation,
        network=NETWORK_DEFINITION,
        model_path=model_path.name,
        model_sha256=_sha256(model_path),
        optimizer_path=optimizer_path.name,
        optimizer_sha256=_sha256(optimizer_path) if include_optimizer else '0' * 64,
        inference_model_path=inference_path.name,
        inference_model_sha256=_sha256(inference_path),
    )
    manifest_path = run_path / f'checkpoint_{generation}.json'
    manifest_path.write_text(manifest.model_dump_json(), encoding='utf-8')
    return CheckpointReference.from_manifest(run_path, manifest)


def _fixture(tmp_path: Path) -> tuple[ExportRequest, Path, Path]:
    working_directory = tmp_path / 'workspace'
    working_directory.mkdir()
    run_path = working_directory / 'training-data' / 'experiment'
    run_path.mkdir(parents=True)
    tensorboard_path = working_directory / 'logs' / 'export-test'
    tensorboard_path.mkdir(parents=True)
    (tensorboard_path / 'events.out.tfevents.test').write_bytes(b'tensorboard')

    template_path = Path('test/configs/go-7x7-experiment.yaml').resolve()
    template = load_experiment_configuration(template_path)
    run = template.run.validated_copy(update={'run_name': 'export-test', 'tensorboard_run_directory': 'export-test'})
    training = template.training.validated_copy(update={'save_path': 'training-data/experiment'})
    experiment = template.validated_copy(
        update={'run': run.model_dump(mode='json'), 'training': training.model_dump(mode='json')}
    )
    experiment_file = working_directory / 'experiment.json'
    experiment_file.write_text(experiment.model_dump_json(indent=2), encoding='utf-8')
    resolved_path = run_path / 'resolved-experiment.json'
    resolved_path.write_text(experiment.model_dump_json(indent=2), encoding='utf-8')

    evaluation_checkpoint = _write_checkpoint(run_path, 2, include_optimizer=False)
    latest_checkpoint = _write_checkpoint(run_path, 3)
    evaluations = run_path / 'evaluations'
    evaluations.mkdir()
    (evaluations / 'manager-state.json').write_text('{}', encoding='utf-8')
    (evaluations / '0000001200-summary.md').write_text('summary', encoding='utf-8')
    (evaluations / '0000001200-result.json').write_text('{"result": true}', encoding='utf-8')
    reference = EvaluationReferenceManifest(
        checkpoints=(
            ElapsedCheckpointReference(boundary_seconds=1200, checkpoint=evaluation_checkpoint),
            ElapsedCheckpointReference(boundary_seconds=2400, checkpoint=latest_checkpoint),
        )
    )
    (evaluations / 'reference-checkpoints.json').write_text(reference.model_dump_json(indent=2), encoding='utf-8')
    (run_path / 'resource-telemetry.jsonl').write_text('{"rss": 1}\n', encoding='utf-8')
    outcome = RunOutcome(
        status=RunOutcomeStatus.COMPLETED,
        reason=None,
        completed_at_utc=datetime.now(timezone.utc),
        elapsed_seconds=14_700.0,
        estimated_cost=1.0,
        latest_checkpoint_model_version=3,
    )
    (run_path / 'run-outcome.json').write_text(outcome.model_dump_json(indent=2), encoding='utf-8')
    configuration_sha256 = experiment_configuration_sha256(experiment)
    source_revision = '1' * 40
    approval = ApprovalRecord(
        approved_by='test',
        approved_at_utc=datetime.now(timezone.utc),
        run_name=experiment.run.run_name,
        source_revision=source_revision,
        configuration_sha256=configuration_sha256,
        provider_name=experiment.run.hardware.provider_name,
        offer_id=experiment.run.hardware.offer_id,
        hourly_price=experiment.training.limits.hourly_price,
        maximum_cost=experiment.training.limits.maximum_cost,
        maximum_wall_time_minutes=int(experiment.training.limits.maximum_wall_time_seconds / 60),
    )
    run_manifest = ExperimentRunManifest(
        experiment=experiment,
        approval=approval,
        resolved_hardware=ResolvedHardware(
            visible_gpu_names=('GPU',),
            visible_gpu_count=1,
            logical_cpu_count=1,
            total_ram_gib=1.0,
            free_disk_gib=1.0,
        ),
        source_revision=source_revision,
        source_worktree_clean=True,
        initial_model_sha256='2' * 64,
        evaluation_dataset_sha256='3' * 64,
        evaluation_dataset_manifest_sha256='4' * 64,
        opening_suite_manifest_sha256='5' * 64,
        evaluation_engine_artifact_sha256=('6' * 64,),
        open_file_soft_limit=32_768,
        torch_version='test',
        cuda_version='test',
    )
    (run_path / 'run_manifest.json').write_text(run_manifest.model_dump_json(indent=2), encoding='utf-8')
    (run_path / 'replay.bin').write_bytes(b'excluded replay')
    (run_path / 'completed-games').mkdir()
    (run_path / 'completed-games' / 'game.json').write_text('{}', encoding='utf-8')
    _write_checkpoint(run_path, 1)

    stdout_log = tmp_path / 'experiment.stdout.log'
    stderr_log = tmp_path / 'experiment.stderr.log'
    stdout_log.write_text('stdout', encoding='utf-8')
    stderr_log.write_text('stderr', encoding='utf-8')
    experiment_definition = QueuedExperiment(
        experiment_id='experiment',
        experiment_file=experiment_file,
        source_revision=source_revision,
        resources=ResourceRequest(cuda_device_count=0, cpu_core_count=1, ram_limit_bytes=1_000),
    )
    slot = ResourceSlot(
        slot_id='slot',
        cuda_devices=(),
        cpu_affinity=(0,),
        ram_capacity_bytes=2_000,
        log_directory=tmp_path,
    )
    queue = QueueConfiguration(
        runner=RunnerCommand(command=('python',)),
        repository_directory=working_directory,
        worktree_root=tmp_path / 'worktrees',
        runtime_directory=working_directory,
        tensorboard_log_directory=working_directory / 'logs',
        slots=(slot,),
        experiments=(experiment_definition,),
        summary_path=tmp_path / 'summary.json',
    )
    queue_path = tmp_path / 'queue.json'
    queue_path.write_text(queue.model_dump_json(indent=2), encoding='utf-8')
    timestamp = datetime.now(timezone.utc)
    execution = ExecutionIdentity(
        configuration_sha256=configuration_sha256,
        source_revision=source_revision,
        setup_commands=(),
        source_worktree=tmp_path / 'worktree',
        runtime_directory=working_directory,
        preserved_configuration_directory=tmp_path / 'evidence',
        preserved_experiment_file=experiment_file,
        command=('python', 'train.py'),
        assignment=create_assignment(experiment_definition, slot),
        started_at=timestamp,
        pid=100,
        process_group_id=100,
        stdout_log=stdout_log,
        stderr_log=stderr_log,
    )
    summary = QueueSummary(
        queue_fingerprint=queue_configuration_fingerprint(queue),
        created_at=timestamp,
        updated_at=timestamp,
        experiments=(
            CompletedExperimentStatus(experiment_id='experiment', execution=execution, finished_at=timestamp),
        ),
    )
    write_queue_summary(queue.summary_path, summary)
    request = ExportRequest(
        queue_configuration_path=queue_path,
        output_path=tmp_path / 'results.zip',
    )
    return request, run_path, queue.summary_path


def test_export_includes_evidence_and_only_selected_checkpoint_state(tmp_path: Path) -> None:
    request, run_path, _ = _fixture(tmp_path)

    manifest = export_experiment_results(request)

    with ZipFile(request.output_path) as archive:
        archive_names = archive.namelist()
        names = set(archive_names)
        archive_manifest = json.loads(archive.read('archive-manifest.json'))
    assert 'experiments/experiment/run/optimizer_3.pt' in names
    assert 'experiments/experiment/run/model_2.pt' in names
    assert 'experiments/experiment/run/model_2.jit.pt' in names
    assert 'experiments/experiment/run/checkpoint_2.json' in names
    assert 'experiments/experiment/run/evaluations/manager-state.json' in names
    assert 'experiments/experiment/tensorboard/events.out.tfevents.test' in names
    assert 'experiments/experiment/queue-logs/stdout.log' in names
    assert 'experiments/experiment/queue-logs/stderr.log' in names
    assert not any('optimizer_2.pt' in name for name in names)
    assert not any('model_1' in name or 'checkpoint_1' in name for name in names)
    assert not any('replay' in name or 'completed-games' in name for name in names)
    assert archive_names.count('experiments/experiment/run/model_3.pt') == 1
    latest_model = next(item for item in manifest.files if item.archive_path.endswith('/model_3.pt'))
    assert latest_model.selection_reasons == (
        'elapsed_evaluation_checkpoint_training_model',
        'latest_checkpoint_training_model',
    )
    assert archive_manifest['files'] == [item.model_dump(mode='json') for item in manifest.files]
    for item in manifest.files:
        assert item.sha256 == _sha256(Path(item.source_path))
    assert (run_path / 'replay.bin').read_bytes() == b'excluded replay'


def test_export_rejects_nonterminal_experiment(tmp_path: Path) -> None:
    request, _, summary_path = _fixture(tmp_path)
    summary = QueueSummary.model_validate_json(summary_path.read_text(encoding='utf-8'))
    completed = summary.experiments[0]
    assert isinstance(completed, CompletedExperimentStatus)
    running = RunningExperimentStatus(experiment_id=completed.experiment_id, execution=completed.execution)
    write_queue_summary(summary_path, summary.model_copy(update={'experiments': (running,)}))

    with pytest.raises(ValueError, match='not terminal'):
        export_experiment_results(request)
    assert not request.output_path.exists()


def test_export_rejects_missing_required_checkpoint_without_mutating_sources(tmp_path: Path) -> None:
    request, run_path, _ = _fixture(tmp_path)
    missing_path = run_path / 'model_2.jit.pt'
    missing_path.unlink()
    before = tuple(sorted(path.relative_to(run_path) for path in run_path.rglob('*')))

    with pytest.raises(ValueError, match='Required result artifact'):
        export_experiment_results(request)

    assert tuple(sorted(path.relative_to(run_path) for path in run_path.rglob('*'))) == before
    assert not request.output_path.exists()


def test_export_rejects_symlink_escape(tmp_path: Path) -> None:
    request, run_path, _ = _fixture(tmp_path)
    target = tmp_path / 'outside.txt'
    target.write_text('outside', encoding='utf-8')
    symlink = run_path / 'evaluations' / 'escaped.txt'
    try:
        symlink.symlink_to(target)
    except OSError:
        pytest.skip('Symbolic links are unavailable on this platform.')

    with pytest.raises(ValueError, match='Symbolic links'):
        export_experiment_results(request)


def test_export_degrades_missing_outcome_for_terminated_experiment(tmp_path: Path) -> None:
    import shutil

    request, run_path, summary_path = _fixture(tmp_path)
    (run_path / 'run-outcome.json').unlink()
    shutil.rmtree(run_path / 'evaluations')
    summary = QueueSummary.model_validate_json(summary_path.read_text(encoding='utf-8'))
    completed = summary.experiments[0]
    assert isinstance(completed, CompletedExperimentStatus)
    failed = FailedExperimentStatus(
        experiment_id=completed.experiment_id,
        execution=completed.execution,
        finished_at=completed.finished_at,
        exit_code=None,
        reason='terminated by SIGKILL',
    )
    write_queue_summary(summary_path, summary.validated_copy(update={'experiments': (failed,)}))

    manifest = export_experiment_results(request)

    assert manifest.experiments[0].queue_status == 'failed'
    assert manifest.experiments[0].latest_generation is None
    assert manifest.experiments[0].evaluation_checkpoint_generations == ()
    descriptions = {artifact.description for artifact in manifest.missing_optional_artifacts}
    assert 'No run outcome was recorded before termination.' in descriptions
    assert 'No evaluations were recorded before termination.' in descriptions
    assert request.output_path.exists()
