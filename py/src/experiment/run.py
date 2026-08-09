"""Shared experiment approval, preparation, and run-manifest ownership."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import multiprocessing
import os
import subprocess
import sys
from pathlib import Path

import psutil
import torch

from src.experiment.base_configuration import (
    RandomInitializationResumeConfiguration,
    WeightsOnlyResumeConfiguration,
)
from src.experiment.configuration import ExperimentConfiguration, experiment_configuration_sha256
from src.evaluation.preparation import PreparedEvaluationArtifacts, prepare_evaluation_artifacts
from src.experiment.run_contract import ApprovalRecord, ResolvedHardware, load_approval_record
from src.games.chess.configuration import ChessExperimentConfiguration
from src.games.composition import create_game_implementation
from src.games.go.configuration import GoExperimentConfiguration
from src.training.targets import build_training_target_layout
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel
from src.training.checkpoint.paths import model_save_path
from src.training.checkpoint.persistence import (
    create_model,
    create_optimizer,
    load_model,
    save_model_and_optimizer,
)


SOURCE_ROOT = Path(__file__).resolve().parents[3]


class ExperimentRunManifest(FrozenModel):
    experiment: ExperimentConfiguration
    approval: ApprovalRecord
    resolved_hardware: ResolvedHardware
    source_revision: str
    source_worktree_clean: bool
    initial_model_sha256: str
    evaluation_dataset_sha256: str
    evaluation_dataset_manifest_sha256: str
    opening_suite_manifest_sha256: str
    evaluation_engine_artifact_sha256: tuple[str, ...]
    open_file_soft_limit: int
    torch_version: str
    cuda_version: str | None


@dataclass(frozen=True)
class _ValidatedRunEnvironment:
    hardware: ResolvedHardware
    approval: ApprovalRecord
    source_revision: str
    open_file_soft_limit: int


def _git_output(arguments: list[str]) -> str:
    completed = subprocess.run(
        ['git', *arguments],
        cwd=SOURCE_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_source_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else SOURCE_ROOT / candidate


def _resolved_hardware() -> ResolvedHardware:
    gpu_count = torch.cuda.device_count()
    return ResolvedHardware(
        visible_gpu_names=tuple(torch.cuda.get_device_name(device_id) for device_id in range(gpu_count)),
        visible_gpu_count=gpu_count,
        logical_cpu_count=multiprocessing.cpu_count(),
        total_ram_gib=psutil.virtual_memory().total / 2**30,
        free_disk_gib=psutil.disk_usage(SOURCE_ROOT).free / 2**30,
    )


def _open_file_soft_limit() -> int:
    if sys.platform != 'linux':
        raise ValueError('Training preparation must run inside the pinned Linux container.')
    completed = subprocess.run(['sh', '-c', 'ulimit -n'], check=True, capture_output=True, text=True)
    try:
        return int(completed.stdout.strip())
    except ValueError as error:
        raise ValueError(f'Could not parse the open-file soft limit: {completed.stdout!r}') from error


def _validate_hardware(experiment: ExperimentConfiguration, hardware: ResolvedHardware) -> None:
    requested = experiment.run.hardware
    topology = experiment.training.topology
    if hardware.visible_gpu_count != requested.gpu_count:
        raise ValueError(f'Expected {requested.gpu_count} visible GPUs, found {hardware.visible_gpu_count}.')
    if any(requested.gpu_model.casefold() not in gpu_name.casefold() for gpu_name in hardware.visible_gpu_names):
        raise ValueError(f'Expected every GPU to match {requested.gpu_model!r}.')
    if hardware.logical_cpu_count < requested.logical_cpu_count:
        raise ValueError('Host has fewer logical CPUs than requested.')
    if hardware.total_ram_gib < requested.minimum_ram_gib:
        raise ValueError('Host has less RAM than requested.')
    if hardware.free_disk_gib < requested.minimum_disk_gib:
        raise ValueError('Host has less free disk than requested.')
    device_ids = topology.trainer.ddp_device_ids + topology.evaluation.device_cycle + topology.self_play.device_ids
    if topology.trainer.device_type == 'cuda' and any(device_id >= requested.gpu_count for device_id in device_ids):
        raise ValueError('Configured CUDA device ID is outside the requested GPU range.')


def _validate_approval(
    experiment: ExperimentConfiguration,
    approval: ApprovalRecord,
    source_revision: str,
) -> None:
    run = experiment.run
    limits = experiment.training.limits
    if not run.requires_explicit_approval:
        raise ValueError('Training configurations must require explicit approval.')
    expected = (
        approval.run_name == run.run_name
        and approval.source_revision == source_revision
        and approval.configuration_sha256 == experiment_configuration_sha256(experiment)
        and approval.provider_name == run.hardware.provider_name
        and approval.offer_id == run.hardware.offer_id
        and approval.hourly_price == limits.hourly_price
        and approval.maximum_cost == limits.maximum_cost
        and approval.maximum_wall_time_minutes == int(limits.maximum_wall_time_seconds / 60)
    )
    if not expected:
        raise ValueError('Approval does not match the requested experiment.')


def _write_manifest(path: Path, manifest: ExperimentRunManifest) -> ExperimentRunManifest:
    serialized = manifest.model_dump_json(indent=2)
    if path.exists():
        existing = ExperimentRunManifest.model_validate_json(path.read_text(encoding='utf-8'))
        current = existing.model_copy(update={'resolved_hardware': manifest.resolved_hardware})
        if current == manifest:
            return existing
        history_hash = hashlib.sha256(existing.model_dump_json().encode('utf-8')).hexdigest()
        write_text_atomically(
            path.parent / 'run_manifests' / f'run_manifest-{history_hash}.json',
            existing.model_dump_json(indent=2) + '\n',
        )
    write_text_atomically(path, serialized + '\n')
    return manifest


def _validate_run_environment(
    experiment: ExperimentConfiguration,
    expected_source_revision: str,
    approval_path: Path,
) -> _ValidatedRunEnvironment:
    hardware = _resolved_hardware()
    _validate_hardware(experiment, hardware)
    source_revision = _git_output(['rev-parse', 'HEAD'])
    if source_revision != expected_source_revision:
        raise ValueError(f'Expected source revision {expected_source_revision}, found {source_revision}.')
    approval = load_approval_record(approval_path)
    _validate_approval(experiment, approval, source_revision)
    _validate_runtime_configuration(experiment)
    open_file_soft_limit = _validate_open_file_limit(experiment)
    if _git_output(['status', '--short']):
        raise ValueError('Refusing to start training from a dirty source working tree.')
    return _ValidatedRunEnvironment(hardware, approval, source_revision, open_file_soft_limit)


def _validate_runtime_configuration(experiment: ExperimentConfiguration) -> None:
    run = experiment.run
    if run.hardware.provider_name.casefold() == 'unconfirmed' or run.hardware.offer_id.casefold() == 'unconfirmed':
        raise ValueError('Hardware provider and offer ID must be confirmed before training.')
    dependency_lock_path = _resolve_source_path(run.environment.dependency_lock_path)
    if _sha256(dependency_lock_path) != run.environment.dependency_lock_sha256:
        raise ValueError('Dependency lock SHA-256 does not match the experiment.')
    actual_python_version = f'{sys.version_info.major}.{sys.version_info.minor}'
    if actual_python_version != run.environment.python_version:
        raise ValueError(f'Expected Python {run.environment.python_version}, found {actual_python_version}.')
    if torch.__version__ != run.environment.torch_version or torch.version.cuda != run.environment.cuda_version:
        raise ValueError('PyTorch or CUDA version does not match the experiment.')
    if os.environ.get('TRAINING_RUNTIME_IMAGE') != run.environment.runtime_image:
        raise ValueError('Training runtime image does not match the experiment.')


def _validate_open_file_limit(experiment: ExperimentConfiguration) -> int:
    open_file_soft_limit = _open_file_soft_limit()
    if open_file_soft_limit < experiment.run.environment.minimum_open_file_soft_limit:
        raise ValueError('Open-file soft limit is below the experiment requirement.')
    if experiment.training.limits.maximum_open_file_count >= open_file_soft_limit:
        raise ValueError('Open-file safety stop must be lower than the process soft limit.')
    return open_file_soft_limit


def _training_device(experiment: ExperimentConfiguration) -> torch.device:
    trainer_topology = experiment.training.topology.trainer
    if trainer_topology.device_type == 'cpu':
        return torch.device('cpu')
    return torch.device('cuda', trainer_topology.rank_zero_device_id)


def _auxiliary_action_sizes(experiment: ExperimentConfiguration) -> tuple[int, ...]:
    match experiment:
        case ChessExperimentConfiguration():
            objective = experiment.chess.objective
        case GoExperimentConfiguration():
            objective = experiment.go.objective
    layout = build_training_target_layout(experiment.network_dimensions.actions, objective.auxiliary_targets)
    return tuple(head.action_size for head in layout.auxiliary_heads)


def _prepare_initial_checkpoint(experiment: ExperimentConfiguration, output_path: Path, manifest_path: Path) -> Path:
    training = experiment.training
    checkpoint_path = model_save_path(0, output_path)
    device = _training_device(experiment)
    auxiliary_action_sizes = _auxiliary_action_sizes(experiment)
    match experiment.run.resume:
        case WeightsOnlyResumeConfiguration(model_path=model_path):
            initial_model_path = _resolve_source_path(model_path)
            if not initial_model_path.is_file():
                raise ValueError(f'Initial model does not exist: {initial_model_path}')
            if not checkpoint_path.exists():
                model = load_model(
                    initial_model_path,
                    training.network,
                    device,
                    experiment.network_dimensions,
                    auxiliary_action_sizes,
                )
                save_model_and_optimizer(model, create_optimizer(model, training.trainer.optimizer), 0, output_path)
        case RandomInitializationResumeConfiguration():
            if checkpoint_path.exists() and not manifest_path.exists():
                raise ValueError(f'Random checkpoint exists without a run manifest: {checkpoint_path}')
            if not checkpoint_path.exists():
                model = create_model(
                    training.network,
                    device,
                    experiment.network_dimensions,
                    auxiliary_action_sizes,
                )
                save_model_and_optimizer(model, create_optimizer(model, training.trainer.optimizer), 0, output_path)
    return checkpoint_path


def _run_manifest(
    experiment: ExperimentConfiguration,
    environment: _ValidatedRunEnvironment,
    artifacts: PreparedEvaluationArtifacts,
    initial_checkpoint_path: Path,
) -> ExperimentRunManifest:
    return ExperimentRunManifest(
        experiment=experiment,
        approval=environment.approval,
        resolved_hardware=environment.hardware,
        source_revision=environment.source_revision,
        source_worktree_clean=True,
        initial_model_sha256=_sha256(initial_checkpoint_path),
        evaluation_dataset_sha256=_sha256(artifacts.dataset_path),
        evaluation_dataset_manifest_sha256=_sha256(artifacts.dataset_manifest_path),
        opening_suite_manifest_sha256=_sha256(artifacts.opening_manifest_path),
        evaluation_engine_artifact_sha256=artifacts.dataset_manifest.engine_artifact_sha256,
        open_file_soft_limit=environment.open_file_soft_limit,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
    )


def prepare_experiment_training_run(
    experiment: ExperimentConfiguration,
    expected_source_revision: str,
    approval_path: Path,
) -> ExperimentRunManifest:
    environment = _validate_run_environment(experiment, expected_source_revision, approval_path)
    evaluation_artifacts = prepare_evaluation_artifacts(
        experiment,
        create_game_implementation(experiment),
        SOURCE_ROOT,
        environment.source_revision,
    )
    output_path = Path(experiment.training.save_path)
    manifest_path = output_path / 'run_manifest.json'
    initial_checkpoint_path = _prepare_initial_checkpoint(experiment, output_path, manifest_path)
    return _write_manifest(
        manifest_path,
        _run_manifest(experiment, environment, evaluation_artifacts, initial_checkpoint_path),
    )
