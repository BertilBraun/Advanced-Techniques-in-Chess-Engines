from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
from typing import Literal
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from pydantic import Field
from src.evaluation.contracts import EvaluationReferenceManifest
from src.experiment.configuration import (
    ExperimentConfiguration,
    experiment_configuration_sha256,
    experiment_configuration_source_paths,
    load_experiment_configuration,
)
from src.experiment.progress_telemetry import RunOutcome
from src.experiment.run import ExperimentRunManifest
from src.experiment_queue.configuration import QueueConfiguration, load_queue_configuration
from src.experiment_queue.state import (
    CompletedExperimentStatus,
    FailedExperimentStatus,
    QueueSummary,
    load_queue_summary,
)
from src.experiment_queue.validation import queue_configuration_fingerprint
from src.training.checkpoint import CheckpointReference
from src.training.checkpoint.contracts import read_checkpoint_manifest
from src.util.frozen_model import FrozenModel
from src.util.hashing import file_sha256


class ExplicitExperimentConfiguration(FrozenModel):
    experiment_id: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9_-]*$')
    experiment_file: Path


class ExportRequest(FrozenModel):
    queue_configuration_path: Path
    output_path: Path
    queue_summary_path: Path | None = None
    tensorboard_log_root: Path | None = None
    experiment_ids: tuple[str, ...] = ()
    explicit_experiments: tuple[ExplicitExperimentConfiguration, ...] = ()
    queue_stdout_log: Path | None = None
    queue_stderr_log: Path | None = None


class ExportedFile(FrozenModel):
    source_path: str
    archive_path: str
    byte_size: int = Field(ge=0)
    sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    experiment_ids: tuple[str, ...]
    selection_reasons: tuple[str, ...]


class MissingOptionalArtifact(FrozenModel):
    experiment_id: str | None
    description: str
    source_path: str | None


class ExportedExperiment(FrozenModel):
    experiment_id: str
    run_name: str
    queue_status: Literal['completed', 'failed']
    source_revision: str
    latest_generation: int | None = Field(ge=0)
    evaluation_checkpoint_generations: tuple[int, ...]


class ResultArchiveManifest(FrozenModel):
    schema_version: Literal[1] = 1
    queue_fingerprint: str
    experiments: tuple[ExportedExperiment, ...]
    files: tuple[ExportedFile, ...]
    missing_optional_artifacts: tuple[MissingOptionalArtifact, ...]


class _SelectedFile:
    def __init__(self, source_path: Path, archive_path: str, experiment_id: str | None, reason: str) -> None:
        self.source_path = source_path
        self.archive_path = archive_path
        self.experiment_ids = set(() if experiment_id is None else (experiment_id,))
        self.reasons = {reason}


class _ArchiveSelection:
    def __init__(self) -> None:
        self._by_archive_path: dict[str, _SelectedFile] = {}
        self._archive_path_by_source: dict[Path, str] = {}
        self._protected_roots: set[Path] = set()
        self.missing_optional: list[MissingOptionalArtifact] = []

    @property
    def files(self) -> tuple[_SelectedFile, ...]:
        return tuple(self._by_archive_path[path] for path in sorted(self._by_archive_path))

    def validate_output_path(self, output_path: Path) -> None:
        resolved_output = output_path.resolve()
        for root in self._protected_roots:
            if resolved_output.is_relative_to(root):
                raise ValueError(f'Result archive cannot be written inside a source artifact directory: {root}')

    def protect_root(self, path: Path) -> None:
        self._protected_roots.add(path.resolve())

    def add(self, source_path: Path, archive_path: str, experiment_id: str | None, reason: str) -> None:
        resolved_source = _require_regular_file(source_path)
        checked_archive_path = _validate_archive_path(archive_path)
        existing_archive_path = self._archive_path_by_source.get(resolved_source)
        if existing_archive_path is not None:
            existing = self._by_archive_path[existing_archive_path]
            existing.experiment_ids.update(() if experiment_id is None else (experiment_id,))
            existing.reasons.add(reason)
            return
        if checked_archive_path in self._by_archive_path:
            raise ValueError(f'Duplicate archive path: {checked_archive_path}')
        selected = _SelectedFile(resolved_source, checked_archive_path, experiment_id, reason)
        self._by_archive_path[checked_archive_path] = selected
        self._archive_path_by_source[resolved_source] = checked_archive_path

    def add_tree(self, source_root: Path, archive_root: str, experiment_id: str, reason: str) -> None:
        resolved_root = _require_directory(source_root)
        paths = sorted(path for path in resolved_root.rglob('*') if path.is_file() or path.is_symlink())
        if not paths:
            raise ValueError(f'Required artifact directory is empty: {resolved_root}')
        for path in paths:
            _require_contained_path(path, resolved_root)
            relative_path = path.relative_to(resolved_root).as_posix()
            self.add(path, f'{archive_root}/{relative_path}', experiment_id, reason)

    def optional_missing(self, experiment_id: str | None, description: str, source_path: Path | None) -> None:
        self.missing_optional.append(
            MissingOptionalArtifact(
                experiment_id=experiment_id,
                description=description,
                source_path=None if source_path is None else str(source_path),
            )
        )


def export_experiment_results(request: ExportRequest) -> ResultArchiveManifest:
    queue_path = _require_regular_file(request.queue_configuration_path)
    queue = load_queue_configuration(queue_path)
    summary_path = queue.summary_path if request.queue_summary_path is None else request.queue_summary_path.resolve()
    summary = load_queue_summary(_require_regular_file(summary_path))
    if summary.queue_fingerprint != queue_configuration_fingerprint(queue):
        raise ValueError('Queue summary does not match the queue configuration runtime identity.')
    selected_statuses = _select_terminal_statuses(summary, request.experiment_ids)
    experiment_files = _experiment_files(queue, request.explicit_experiments)
    selection = _ArchiveSelection()
    selection.add(queue_path, f'queue/configuration/{queue_path.name}', None, 'queue_configuration')
    selection.add(summary_path, f'queue/summary/{summary_path.name}', None, 'durable_queue_summary')
    _add_optional_queue_log(selection, request.queue_stdout_log, 'stdout')
    _add_optional_queue_log(selection, request.queue_stderr_log, 'stderr')

    tensorboard_log_root = (
        queue.tensorboard_log_directory if request.tensorboard_log_root is None else request.tensorboard_log_root
    )
    exported_experiments = tuple(
        _select_experiment(status, experiment_files, tensorboard_log_root, selection) for status in selected_statuses
    )
    return _write_archive(request.output_path, summary, exported_experiments, selection)


def _select_terminal_statuses(
    summary: QueueSummary,
    requested_ids: tuple[str, ...],
) -> tuple[CompletedExperimentStatus | FailedExperimentStatus, ...]:
    if len(set(requested_ids)) != len(requested_ids):
        raise ValueError('Requested experiment IDs must be unique.')
    statuses = {status.experiment_id: status for status in summary.experiments}
    selected_ids = requested_ids or tuple(statuses)
    unknown = tuple(experiment_id for experiment_id in selected_ids if experiment_id not in statuses)
    if unknown:
        raise ValueError(f'Queue summary does not contain requested experiments: {unknown}.')
    selected: list[CompletedExperimentStatus | FailedExperimentStatus] = []
    for experiment_id in selected_ids:
        status = statuses[experiment_id]
        if not isinstance(status, CompletedExperimentStatus | FailedExperimentStatus):
            raise ValueError(f'Experiment {experiment_id!r} is not terminal: {status.status}.')
        selected.append(status)
    return tuple(selected)


def _experiment_files(
    queue: QueueConfiguration,
    explicit: tuple[ExplicitExperimentConfiguration, ...],
) -> dict[str, Path]:
    files = {experiment.experiment_id: experiment.experiment_file for experiment in queue.experiments}
    for binding in explicit:
        existing = files.get(binding.experiment_id)
        resolved = binding.experiment_file.resolve()
        if existing is not None and existing != resolved:
            raise ValueError(f'Explicit configuration conflicts with queue experiment {binding.experiment_id!r}.')
        files[binding.experiment_id] = resolved
    return files


def _select_experiment(
    status: CompletedExperimentStatus | FailedExperimentStatus,
    experiment_files: dict[str, Path],
    tensorboard_log_root: Path,
    selection: _ArchiveSelection,
) -> ExportedExperiment:
    experiment_id = status.experiment_id
    configured_experiment_file = experiment_files.get(experiment_id)
    if status.execution.preserved_experiment_file.is_file():
        experiment_file = status.execution.preserved_experiment_file
    else:
        retained_worktree_file = (
            status.execution.source_worktree
            / status.execution.preserved_experiment_file.relative_to(status.execution.preserved_configuration_directory)
        )
        experiment_file = retained_worktree_file if retained_worktree_file.is_file() else configured_experiment_file
    if experiment_file is None:
        raise ValueError(f'No authored configuration is available for experiment {experiment_id!r}.')
    experiment = load_experiment_configuration(_require_regular_file(experiment_file))
    configuration_sha256 = experiment_configuration_sha256(experiment)
    if configuration_sha256 != status.execution.configuration_sha256:
        raise ValueError(f'Queue summary configuration hash does not match experiment {experiment_id!r}.')

    archive_prefix = f'experiments/{experiment_id}'
    for index, source_path in enumerate(experiment_configuration_source_paths(experiment_file)):
        selection.add(
            source_path,
            f'{archive_prefix}/configuration/{index:02d}-{source_path.name}',
            experiment_id,
            'authored_experiment_configuration',
        )
    selection.add(
        status.execution.stdout_log,
        f'{archive_prefix}/queue-logs/stdout.log',
        experiment_id,
        'experiment_stdout',
    )
    selection.add(
        status.execution.stderr_log,
        f'{archive_prefix}/queue-logs/stderr.log',
        experiment_id,
        'experiment_stderr',
    )

    working_directory = _require_directory(status.execution.runtime_directory)
    run_path = _runtime_path(experiment.training.save_path, working_directory)
    _require_directory(run_path)
    selection.protect_root(run_path)
    manifest_path = run_path / 'run_manifest.json'
    resolved_path = run_path / 'resolved-experiment.json'
    outcome_path = run_path / 'run-outcome.json'
    telemetry_path = run_path / 'resource-telemetry.jsonl'
    run_manifest = ExperimentRunManifest.model_validate_json(
        _require_regular_file(manifest_path).read_text(encoding='utf-8')
    )
    resolved_experiment = load_experiment_configuration(_require_regular_file(resolved_path))
    outcome: RunOutcome | None
    if outcome_path.is_file():
        outcome = RunOutcome.model_validate_json(outcome_path.read_text(encoding='utf-8'))
    elif status.status == 'failed':
        # A hard-killed run never reaches its outcome write; export the rest of its evidence anyway.
        selection.optional_missing(experiment_id, 'No run outcome was recorded before termination.', outcome_path)
        outcome = None
    else:
        raise ValueError(f'Completed experiment {experiment_id!r} is missing its run outcome.')
    _validate_run_identity(experiment_id, experiment, resolved_experiment, run_manifest, outcome)
    required_files = [
        (manifest_path, 'run_manifest'),
        (resolved_path, 'resolved_experiment_configuration'),
        (telemetry_path, 'resource_telemetry'),
    ]
    if outcome is not None:
        required_files.append((outcome_path, 'run_outcome'))
    for path, reason in required_files:
        _require_contained_path(path, run_path)
        selection.add(path, f'{archive_prefix}/run/{path.name}', experiment_id, reason)

    historical_manifests = run_path / 'run_manifests'
    if historical_manifests.exists():
        selection.add_tree(
            historical_manifests,
            f'{archive_prefix}/run/run_manifests',
            experiment_id,
            'historical_run_manifests',
        )
    else:
        selection.optional_missing(experiment_id, 'No historical run manifests were present.', historical_manifests)

    evaluation_path = run_path / 'evaluations'
    reference_path = evaluation_path / 'reference-checkpoints.json'
    if outcome is None and not reference_path.is_file():
        selection.optional_missing(experiment_id, 'No evaluations were recorded before termination.', evaluation_path)
        evaluation_generations: tuple[int, ...] = ()
        reference_manifest = None
    else:
        selection.add_tree(evaluation_path, f'{archive_prefix}/run/evaluations', experiment_id, 'evaluation_artifacts')
        reference_manifest = EvaluationReferenceManifest.model_validate_json(
            _require_regular_file(reference_path).read_text(encoding='utf-8')
        )
        evaluation_generations = tuple(
            dict.fromkeys(entry.checkpoint.generation for entry in reference_manifest.checkpoints)
        )
    if outcome is not None and any(
        generation > outcome.latest_checkpoint_model_version for generation in evaluation_generations
    ):
        raise ValueError(
            f'Evaluation manifest references a checkpoint newer than the run outcome for {experiment_id!r}.'
        )
    for entry in reference_manifest.checkpoints if reference_manifest is not None else ():
        checkpoint_manifest = read_checkpoint_manifest(entry.checkpoint.generation, run_path)
        expected_reference = CheckpointReference.from_manifest(run_path, checkpoint_manifest)
        reference = entry.checkpoint
        matching_reference = (
            reference.generation == expected_reference.generation
            and reference.inference_model_sha256 == expected_reference.inference_model_sha256
            and _runtime_path(reference.manifest_path, working_directory) == expected_reference.manifest_path.resolve()
            and _runtime_path(reference.model_path, working_directory) == expected_reference.model_path.resolve()
            and _runtime_path(reference.optimizer_path, working_directory)
            == expected_reference.optimizer_path.resolve()
            and _runtime_path(reference.inference_model_path, working_directory)
            == expected_reference.inference_model_path.resolve()
        )
        if not matching_reference:
            raise ValueError(f'Evaluation checkpoint reference is inconsistent for experiment {experiment_id!r}.')
    for generation in evaluation_generations:
        _add_checkpoint(selection, run_path, experiment_id, generation, include_optimizer=False)
    if outcome is not None:
        _add_checkpoint(
            selection, run_path, experiment_id, outcome.latest_checkpoint_model_version, include_optimizer=True
        )

    resolved_tensorboard_root = _runtime_path(tensorboard_log_root, working_directory)
    tensorboard_path = resolved_tensorboard_root / experiment.run.tensorboard_run_directory
    _require_contained_path(tensorboard_path, resolved_tensorboard_root)
    selection.protect_root(tensorboard_path)
    selection.add_tree(
        tensorboard_path,
        f'{archive_prefix}/tensorboard',
        experiment_id,
        'tensorboard_event_logs',
    )
    return ExportedExperiment(
        experiment_id=experiment_id,
        run_name=experiment.run.run_name,
        queue_status=status.status,
        source_revision=run_manifest.source_revision,
        latest_generation=None if outcome is None else outcome.latest_checkpoint_model_version,
        evaluation_checkpoint_generations=evaluation_generations,
    )


def _validate_run_identity(
    experiment_id: str,
    authored: ExperimentConfiguration,
    resolved: ExperimentConfiguration,
    manifest: ExperimentRunManifest,
    outcome: RunOutcome | None,
) -> None:
    if authored != resolved or authored != manifest.experiment:
        raise ValueError(f'Authored, resolved, and manifested configurations disagree for {experiment_id!r}.')
    if manifest.approval.configuration_sha256 != experiment_configuration_sha256(authored):
        raise ValueError(f'Run approval configuration hash does not match experiment {experiment_id!r}.')
    if manifest.approval.source_revision != manifest.source_revision:
        raise ValueError(f'Run approval source revision does not match experiment {experiment_id!r}.')


def _add_checkpoint(
    selection: _ArchiveSelection,
    run_path: Path,
    experiment_id: str,
    generation: int,
    include_optimizer: bool,
) -> None:
    manifest = read_checkpoint_manifest(generation, run_path)
    manifest_path = run_path / f'checkpoint_{generation}.json'
    selection.add(
        manifest_path,
        f'experiments/{experiment_id}/run/{manifest_path.name}',
        experiment_id,
        'latest_checkpoint_manifest' if include_optimizer else 'elapsed_evaluation_checkpoint_manifest',
    )
    artifacts = (
        (manifest.model_path, manifest.model_sha256, 'training_model'),
        (manifest.inference_model_path, manifest.inference_model_sha256, 'inference_model'),
    )
    for relative_path, expected_sha256, artifact_kind in artifacts:
        path = _checkpoint_artifact_path(run_path, relative_path)
        _require_hash(path, expected_sha256)
        reason_prefix = 'latest_checkpoint' if include_optimizer else 'elapsed_evaluation_checkpoint'
        selection.add(
            path, f'experiments/{experiment_id}/run/{path.name}', experiment_id, f'{reason_prefix}_{artifact_kind}'
        )
    if include_optimizer:
        optimizer_path = _checkpoint_artifact_path(run_path, manifest.optimizer_path)
        _require_hash(optimizer_path, manifest.optimizer_sha256)
        selection.add(
            optimizer_path,
            f'experiments/{experiment_id}/run/{optimizer_path.name}',
            experiment_id,
            'latest_checkpoint_optimizer',
        )


def _checkpoint_artifact_path(run_path: Path, relative_path: str) -> Path:
    path = run_path / relative_path
    _require_contained_path(path, run_path)
    return _require_regular_file(path)


def _require_hash(path: Path, expected_sha256: str) -> None:
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise ValueError(f'Artifact hash does not match its checkpoint manifest: {path}')


def _add_optional_queue_log(selection: _ArchiveSelection, path: Path | None, stream: str) -> None:
    if path is None:
        selection.optional_missing(None, f'Queue supervisor {stream} log was not supplied.', None)
        return
    selection.add(path, f'queue/logs/{stream}.log', None, f'queue_supervisor_{stream}')


def _write_archive(
    output_path: Path,
    summary: QueueSummary,
    experiments: tuple[ExportedExperiment, ...],
    selection: _ArchiveSelection,
) -> ResultArchiveManifest:
    resolved_output = output_path.resolve()
    selection.validate_output_path(resolved_output)
    if resolved_output.exists():
        raise ValueError(f'Result archive already exists: {resolved_output}')
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = resolved_output.with_name(f'.{resolved_output.name}.partial')
    if temporary_path.exists():
        raise ValueError(f'Result archive temporary path already exists: {temporary_path}')
    exported_files: list[ExportedFile] = []
    try:
        with ZipFile(temporary_path, 'x', compression=ZIP_DEFLATED, compresslevel=6) as archive:
            for selected in selection.files:
                size, sha256 = _write_file(archive, selected.source_path, selected.archive_path)
                exported_files.append(
                    ExportedFile(
                        source_path=str(selected.source_path),
                        archive_path=selected.archive_path,
                        byte_size=size,
                        sha256=sha256,
                        experiment_ids=tuple(sorted(selected.experiment_ids)),
                        selection_reasons=tuple(sorted(selected.reasons)),
                    )
                )
            manifest = ResultArchiveManifest(
                queue_fingerprint=summary.queue_fingerprint,
                experiments=experiments,
                files=tuple(exported_files),
                missing_optional_artifacts=tuple(selection.missing_optional),
            )
            _write_bytes(archive, 'archive-manifest.json', (manifest.model_dump_json(indent=2) + '\n').encode('utf-8'))
        os.replace(temporary_path, resolved_output)
        return manifest
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _write_file(archive: ZipFile, source_path: Path, archive_path: str) -> tuple[int, str]:
    before = source_path.stat()
    digest = hashlib.sha256()
    size = 0
    with source_path.open('rb') as source, archive.open(_zip_info(archive_path), 'w') as destination:
        while chunk := source.read(1024 * 1024):
            destination.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    after = source_path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or size != before.st_size:
        raise ValueError(f'Source artifact changed while it was exported: {source_path}')
    return size, digest.hexdigest()


def _write_bytes(archive: ZipFile, archive_path: str, payload: bytes) -> None:
    with archive.open(_zip_info(archive_path), 'w') as destination:
        destination.write(payload)


def _zip_info(archive_path: str) -> ZipInfo:
    info = ZipInfo(_validate_archive_path(archive_path), date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def _runtime_path(path: str | Path, working_directory: Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (working_directory / candidate).resolve()


def _require_regular_file(path: Path) -> Path:
    if path.is_symlink():
        raise ValueError(f'Symbolic links are not allowed in result exports: {path}')
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f'Required result artifact does not exist: {resolved}')
    return resolved


def _require_directory(path: Path) -> Path:
    if path.is_symlink():
        raise ValueError(f'Symbolic links are not allowed in result exports: {path}')
    resolved = path.resolve()
    if not resolved.is_dir():
        raise ValueError(f'Required result directory does not exist: {resolved}')
    return resolved


def _require_contained_path(path: Path, root: Path) -> None:
    if path.is_symlink():
        raise ValueError(f'Symbolic links are not allowed in result exports: {path}')
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    if not resolved_path.is_relative_to(resolved_root):
        raise ValueError(f'Result artifact escapes its owned directory: {path}')


def _validate_archive_path(path: str) -> str:
    pure_path = PurePosixPath(path)
    if pure_path.is_absolute() or not pure_path.parts or any(part in {'', '.', '..'} for part in pure_path.parts):
        raise ValueError(f'Unsafe archive path: {path!r}')
    return pure_path.as_posix()
