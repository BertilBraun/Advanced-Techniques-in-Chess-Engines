"""Shared experiment approval, preparation, and run-manifest ownership."""

from __future__ import annotations

import hashlib
import multiprocessing
import os
import subprocess
import sys
from pathlib import Path

import psutil
import torch

from src.experiment.configuration import (
    ChessExperimentConfiguration,
    ExperimentConfiguration,
    RandomInitializationResumeConfiguration,
    WeightsOnlyResumeConfiguration,
)
from src.experiment.evaluation_protocol import load_opening_suite
from src.experiment.run_contract import ApprovalRecord, ResolvedHardware, load_approval_record
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel
from src.util.save_paths import create_model, create_optimizer, load_model, model_save_path, save_model_and_optimizer


SOURCE_ROOT = Path(__file__).resolve().parents[3]


class ExperimentRunManifest(FrozenModel):
    experiment: ExperimentConfiguration
    approval: ApprovalRecord
    resolved_hardware: ResolvedHardware
    source_revision: str
    source_worktree_clean: bool
    initial_model_sha256: str
    evaluation_dataset_sha256: str | None
    stockfish_binary_sha256: str | None
    open_file_soft_limit: int
    torch_version: str
    cuda_version: str | None


def experiment_sha256(experiment: ExperimentConfiguration) -> str:
    return hashlib.sha256(experiment.model_dump_json().encode('utf-8')).hexdigest()


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


def _validate_hardware(experiment: ChessExperimentConfiguration, hardware: ResolvedHardware) -> None:
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
        and approval.configuration_sha256 == experiment_sha256(experiment)
        and approval.provider_name == run.hardware.provider_name
        and approval.offer_id == run.hardware.offer_id
        and approval.cost_currency == limits.cost_currency
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


def prepare_experiment_training_run(
    experiment: ExperimentConfiguration,
    expected_source_revision: str,
    approval_path: Path,
) -> ExperimentRunManifest:
    hardware = _resolved_hardware()
    _validate_hardware(experiment, hardware)
    source_revision = _git_output(['rev-parse', 'HEAD'])
    if source_revision != expected_source_revision:
        raise ValueError(f'Expected source revision {expected_source_revision}, found {source_revision}.')
    approval = load_approval_record(approval_path)
    _validate_approval(experiment, approval, source_revision)

    run = experiment.run
    training = experiment.training
    evaluation = experiment.chess.evaluation if isinstance(experiment, ChessExperimentConfiguration) else None
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
    open_file_soft_limit = _open_file_soft_limit()
    if open_file_soft_limit < run.environment.minimum_open_file_soft_limit:
        raise ValueError('Open-file soft limit is below the experiment requirement.')
    if training.limits.maximum_open_file_count >= open_file_soft_limit:
        raise ValueError('Open-file safety stop must be lower than the process soft limit.')

    opening_suite_path = (
        _resolve_source_path(evaluation.opening_suite_path)
        if evaluation is not None and evaluation.opening_suite_path
        else None
    )
    if (
        opening_suite_path is not None
        and evaluation is not None
        and evaluation.num_games != len(load_opening_suite(opening_suite_path)) * 2
    ):
        raise ValueError('Evaluation game count must cover every opening with both colors.')
    dataset_path = (
        _resolve_source_path(evaluation.dataset_path) if evaluation is not None and evaluation.dataset_path else None
    )
    if dataset_path is not None and not dataset_path.is_file():
        raise ValueError(f'Evaluation dataset does not exist: {dataset_path}')
    stockfish_path = (
        Path(evaluation.stockfish_binary_path) if evaluation is not None and evaluation.stockfish_binary_path else None
    )
    if stockfish_path is not None and not stockfish_path.is_file():
        raise ValueError(f'Stockfish binary does not exist: {stockfish_path}')

    source_worktree_clean = not bool(_git_output(['status', '--short']))
    if not source_worktree_clean:
        raise ValueError('Refusing to start training from a dirty source working tree.')
    output_path = Path(training.save_path)
    manifest_path = output_path / 'run_manifest.json'
    initial_checkpoint_path = model_save_path(0, output_path)
    optimizer_type = training.trainer.optimizer
    device = (
        torch.device('cpu')
        if training.topology.trainer.device_type == 'cpu'
        else torch.device('cuda', training.topology.trainer.rank_zero_device_id)
    )
    dimensions = experiment.network_dimensions
    match run.resume:
        case WeightsOnlyResumeConfiguration(model_path=model_path):
            initial_model_path = _resolve_source_path(model_path)
            if not initial_model_path.is_file():
                raise ValueError(f'Initial model does not exist: {initial_model_path}')
            if not initial_checkpoint_path.exists():
                model = load_model(initial_model_path, training.network, device, dimensions)
                save_model_and_optimizer(model, create_optimizer(model, optimizer_type), 0, output_path)
        case RandomInitializationResumeConfiguration():
            if initial_checkpoint_path.exists() and not manifest_path.exists():
                raise ValueError(f'Random checkpoint exists without a run manifest: {initial_checkpoint_path}')
            if not initial_checkpoint_path.exists():
                model = create_model(training.network, device, dimensions)
                save_model_and_optimizer(model, create_optimizer(model, optimizer_type), 0, output_path)

    manifest = ExperimentRunManifest(
        experiment=experiment,
        approval=approval,
        resolved_hardware=hardware,
        source_revision=source_revision,
        source_worktree_clean=source_worktree_clean,
        initial_model_sha256=_sha256(initial_checkpoint_path),
        evaluation_dataset_sha256=_sha256(dataset_path) if dataset_path else None,
        stockfish_binary_sha256=_sha256(stockfish_path) if stockfish_path else None,
        open_file_soft_limit=open_file_soft_limit,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
    )
    return _write_manifest(manifest_path, manifest)
