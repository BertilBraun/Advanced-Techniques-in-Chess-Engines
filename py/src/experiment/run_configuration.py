from __future__ import annotations

import hashlib
import multiprocessing
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

import psutil
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.experiment.cost_accounting import CostCurrency
from src.train.TrainingArgs import ArtifactRetention, ClusterParams, OptimizerType, RuntimeLimits, TrainingArgs
from src.experiment.evaluation_protocol import load_opening_suite
from src.util.save_paths import (
    create_optimizer,
    load_model,
    model_save_path,
    save_model_and_optimizer,
)


SOURCE_ROOT = Path(__file__).resolve().parents[3]


class TrainingStage(str, Enum):
    SYSTEMS_PILOT = 'systems_pilot'
    CONTINUATION = 'continuation'


class ResumeMode(str, Enum):
    WEIGHTS_ONLY = 'weights_only'


class ResumeConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    mode: ResumeMode
    model_path: str
    optimizer: OptimizerType


class BudgetConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    currency: CostCurrency
    hourly_price: float = Field(gt=0)
    maximum_cost: float | None
    maximum_wall_time_minutes: int = Field(gt=0)

    @model_validator(mode='after')
    def validate_projected_cost(self) -> BudgetConfiguration:
        projected_cost = self.hourly_price * self.maximum_wall_time_minutes / 60
        if self.maximum_cost is not None and projected_cost > self.maximum_cost:
            raise ValueError(
                f'Projected cost {self.currency.value} {projected_cost:.2f} exceeds the configured '
                f'{self.currency.value} {self.maximum_cost:.2f} maximum.'
            )
        return self


class HardwareConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    provider_name: str = Field(min_length=1)
    offer_id: str = Field(min_length=1)
    gpu_model: str = Field(min_length=1)
    gpu_count: int = Field(gt=0)
    logical_cpu_count: int = Field(gt=0)
    minimum_ram_gib: int = Field(gt=0)
    minimum_disk_gib: int = Field(gt=0)


class TopologyConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    trainer_device_id: int = Field(ge=0)
    evaluation_device_id: int = Field(ge=0)
    self_play_processes_per_device: tuple[int, ...]
    mcts_threads_per_process: int = Field(gt=0)
    parallel_games_per_process: int = Field(gt=0)
    trainer_cpu_threads: int = Field(gt=0)
    trainer_interop_threads: int = Field(gt=0)
    dataloader_workers: int = Field(ge=0)
    reserved_logical_cpus: int = Field(ge=1)
    max_concurrent_evaluations: int = Field(ge=1)
    max_concurrent_evaluation_tasks: int = Field(ge=1)

    @model_validator(mode='after')
    def validate_process_counts(self) -> TopologyConfiguration:
        if not self.self_play_processes_per_device:
            raise ValueError('At least one self-play device must be configured.')
        if any(process_count < 0 for process_count in self.self_play_processes_per_device):
            raise ValueError('Self-play process counts cannot be negative.')
        if sum(self.self_play_processes_per_device) == 0:
            raise ValueError('At least one self-play process must be configured.')
        return self


class WorkloadConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    iterations: int = Field(gt=0)
    games_per_iteration: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    random_seed: int = Field(ge=0)
    evaluation_games: int = Field(gt=0)
    evaluation_searches_per_turn: int = Field(gt=0)
    evaluation_every_iterations: int = Field(gt=0)


class SafetyConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    maximum_open_file_count: int = Field(gt=0)
    maximum_host_ram_percent: float = Field(gt=0, lt=100)
    minimum_free_disk_gib: float = Field(gt=0)
    telemetry_interval_seconds: float = Field(gt=0)


class RetentionConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    checkpoint_count: int = Field(gt=0)
    replay_window_iterations: int = Field(gt=0)


class EvaluationProtocolConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    opening_suite_path: str
    reference_model_path: str
    raw_results_subdirectory: str
    maximum_game_plies: int = Field(gt=0)
    bootstrap_seed: int = Field(ge=0)
    bootstrap_samples: int = Field(gt=0)
    mcts_threads: int = Field(gt=0)
    evaluate_dataset: bool
    evaluate_initial_checkpoint: bool
    stockfish_binary_path: str | None
    stockfish_nodes_per_move: int = Field(gt=0)
    stockfish_threads: int = Field(gt=0)
    stockfish_hash_mib: int = Field(gt=0)


class EnvironmentConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    runtime_image: str
    python_version: str
    torch_version: str
    cuda_version: str
    dependency_lock_path: str
    dependency_lock_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    stockfish_revision: str = Field(pattern=r'^[0-9a-f]{40}$')
    minimum_open_file_soft_limit: int = Field(gt=0)


class RunConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    run_name: str = Field(min_length=1)
    stage: TrainingStage
    requires_explicit_approval: bool
    output_path: str
    resume: ResumeConfiguration
    budget: BudgetConfiguration
    hardware: HardwareConfiguration
    topology: TopologyConfiguration
    workload: WorkloadConfiguration
    safety: SafetyConfiguration
    retention: RetentionConfiguration
    evaluation_protocol: EvaluationProtocolConfiguration
    environment: EnvironmentConfiguration


class ResolvedHardware(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    visible_gpu_names: tuple[str, ...]
    visible_gpu_count: int
    logical_cpu_count: int
    total_ram_gib: float
    free_disk_gib: float


class ApprovalRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    approved_by: str = Field(min_length=1)
    approved_at_utc: datetime
    run_name: str
    source_revision: str = Field(pattern=r'^[0-9a-f]{40}$')
    configuration_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    provider_name: str
    offer_id: str
    cost_currency: CostCurrency
    hourly_price: float
    maximum_cost: float | None
    maximum_wall_time_minutes: int

    @model_validator(mode='after')
    def validate_approval_timestamp(self) -> ApprovalRecord:
        if self.approved_at_utc.tzinfo is None or self.approved_at_utc.utcoffset() is None:
            raise ValueError('approved_at_utc must include a timezone.')
        return self


class RunManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    configuration: RunConfiguration
    approval: ApprovalRecord
    resolved_hardware: ResolvedHardware
    source_revision: str
    source_worktree_clean: bool
    initial_model_sha256: str
    stockfish_binary_sha256: str | None
    open_file_soft_limit: int
    torch_version: str
    cuda_version: str | None


@dataclass(frozen=True)
class ConstantLearningRate:
    value: float

    def __call__(self, _: int, __: OptimizerType) -> float:
        return self.value


def load_run_configuration(path: Path) -> RunConfiguration:
    return RunConfiguration.model_validate_json(path.read_text(encoding='utf-8'))


def load_approval_record(path: Path) -> ApprovalRecord:
    return ApprovalRecord.model_validate_json(path.read_text(encoding='utf-8'))


def configuration_sha256(configuration: RunConfiguration) -> str:
    serialized = configuration.model_dump_json()
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()


def validate_approval(
    configuration: RunConfiguration,
    approval: ApprovalRecord,
    source_revision: str,
) -> None:
    if not configuration.requires_explicit_approval:
        raise ValueError('Paid training configurations must require explicit approval.')
    if approval.run_name != configuration.run_name:
        raise ValueError('Approval run name does not match the requested configuration.')
    if approval.source_revision != source_revision:
        raise ValueError('Approval source revision does not match the requested revision.')
    if approval.configuration_sha256 != configuration_sha256(configuration):
        raise ValueError('Approval configuration hash does not match the requested configuration.')
    if approval.provider_name != configuration.hardware.provider_name:
        raise ValueError('Approval provider does not match the requested hardware provider.')
    if approval.offer_id != configuration.hardware.offer_id:
        raise ValueError('Approval offer ID does not match the requested hardware offer.')
    if approval.cost_currency != configuration.budget.currency:
        raise ValueError('Approval cost currency does not match the requested budget.')
    if approval.hourly_price != configuration.budget.hourly_price:
        raise ValueError('Approval hourly price does not match the requested budget.')
    if approval.maximum_cost != configuration.budget.maximum_cost:
        raise ValueError('Approval cost ceiling does not match the requested budget.')
    if approval.maximum_wall_time_minutes != configuration.budget.maximum_wall_time_minutes:
        raise ValueError('Approval duration does not match the requested budget.')


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
    parsed_path = Path(path)
    if parsed_path.is_absolute():
        return parsed_path
    return SOURCE_ROOT / parsed_path


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
        raise ValueError('Paid training preparation must run inside the pinned Linux container.')
    completed = subprocess.run(
        ['sh', '-c', 'ulimit -n'],
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        return int(completed.stdout.strip())
    except ValueError as error:
        raise ValueError(f'Could not parse the open-file soft limit: {completed.stdout!r}') from error


def validate_run_configuration(
    configuration: RunConfiguration,
    resolved_hardware: ResolvedHardware,
) -> None:
    hardware = configuration.hardware
    topology = configuration.topology

    if resolved_hardware.visible_gpu_count != hardware.gpu_count:
        raise ValueError(f'Expected {hardware.gpu_count} visible GPUs, found {resolved_hardware.visible_gpu_count}.')
    unexpected_gpu_names = [
        gpu_name
        for gpu_name in resolved_hardware.visible_gpu_names
        if hardware.gpu_model.casefold() not in gpu_name.casefold()
    ]
    if unexpected_gpu_names:
        raise ValueError(f'Expected every GPU to match {hardware.gpu_model!r}; found {unexpected_gpu_names}.')
    if len(topology.self_play_processes_per_device) != hardware.gpu_count:
        raise ValueError('self_play_processes_per_device must contain exactly one entry per visible GPU.')
    if topology.trainer_device_id >= hardware.gpu_count:
        raise ValueError(f'Trainer device {topology.trainer_device_id} is outside the configured GPU range.')
    if topology.evaluation_device_id >= hardware.gpu_count:
        raise ValueError(f'Evaluation device {topology.evaluation_device_id} is outside the configured GPU range.')
    if resolved_hardware.logical_cpu_count < hardware.logical_cpu_count:
        raise ValueError(
            f'Expected at least {hardware.logical_cpu_count} logical CPUs, found {resolved_hardware.logical_cpu_count}.'
        )
    if resolved_hardware.total_ram_gib < hardware.minimum_ram_gib:
        raise ValueError(
            f'Expected at least {hardware.minimum_ram_gib} GiB RAM, found {resolved_hardware.total_ram_gib:.1f} GiB.'
        )
    if resolved_hardware.free_disk_gib < hardware.minimum_disk_gib:
        raise ValueError(
            f'Expected at least {hardware.minimum_disk_gib} GiB free disk, '
            f'found {resolved_hardware.free_disk_gib:.1f} GiB.'
        )

    self_play_cpu_slots = sum(topology.self_play_processes_per_device) * topology.mcts_threads_per_process
    required_cpu_slots = (
        self_play_cpu_slots
        + topology.trainer_cpu_threads
        + topology.dataloader_workers
        + topology.reserved_logical_cpus
    )
    if required_cpu_slots > resolved_hardware.logical_cpu_count:
        raise ValueError(
            f'Topology reserves {required_cpu_slots} logical CPU slots but only '
            f'{resolved_hardware.logical_cpu_count} are visible.'
        )


def _self_play_device_ids(processes_per_device: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(
        device_id for device_id, process_count in enumerate(processes_per_device) for _ in range(process_count)
    )


def _constant_learning_rate(
    learning_rate: float,
) -> Callable[[int, OptimizerType], float]:
    return ConstantLearningRate(learning_rate)


def apply_run_configuration(
    training_args: TrainingArgs,
    configuration: RunConfiguration,
) -> None:
    topology = configuration.topology
    workload = configuration.workload
    maximum_sampling_window = max(
        training_args.training.sampling_window(iteration) for iteration in range(workload.iterations + 1)
    )
    if configuration.retention.replay_window_iterations < maximum_sampling_window:
        raise ValueError(
            f'Replay retention of {configuration.retention.replay_window_iterations} iterations is below '
            f'the maximum training sampling window of {maximum_sampling_window}.'
        )

    training_args.save_path = str(_resolve_source_path(configuration.output_path))
    training_args.num_iterations = workload.iterations
    training_args.num_games_per_iteration = workload.games_per_iteration
    training_args.random_seed = workload.random_seed
    training_args.self_play.num_parallel_games = topology.parallel_games_per_process
    training_args.self_play.mcts.num_threads = topology.mcts_threads_per_process
    training_args.training.num_workers = topology.dataloader_workers
    training_args.training.learning_rate = _constant_learning_rate(workload.learning_rate)
    training_args.cluster = ClusterParams(
        trainer_device_id=topology.trainer_device_id,
        evaluation_device_id=topology.evaluation_device_id,
        self_play_device_ids=_self_play_device_ids(topology.self_play_processes_per_device),
        trainer_cpu_threads=topology.trainer_cpu_threads,
        trainer_interop_threads=topology.trainer_interop_threads,
        max_concurrent_evaluations=topology.max_concurrent_evaluations,
    )
    training_args.run_limits = RuntimeLimits(
        cost_currency=configuration.budget.currency,
        hourly_price=configuration.budget.hourly_price,
        maximum_cost=configuration.budget.maximum_cost,
        maximum_wall_time_seconds=configuration.budget.maximum_wall_time_minutes * 60,
        maximum_open_file_count=configuration.safety.maximum_open_file_count,
        maximum_host_ram_percent=configuration.safety.maximum_host_ram_percent,
        minimum_free_disk_gib=configuration.safety.minimum_free_disk_gib,
    )
    training_args.artifact_retention = ArtifactRetention(
        checkpoint_count=configuration.retention.checkpoint_count,
        replay_window_iterations=configuration.retention.replay_window_iterations,
    )

    if training_args.evaluation is not None:
        evaluation_protocol = configuration.evaluation_protocol
        training_args.evaluation.num_games = workload.evaluation_games
        training_args.evaluation.num_searches_per_turn = workload.evaluation_searches_per_turn
        training_args.evaluation.every_n_iterations = workload.evaluation_every_iterations
        training_args.evaluation.max_concurrent_tasks = topology.max_concurrent_evaluation_tasks
        training_args.evaluation.evaluate_initial_checkpoint = evaluation_protocol.evaluate_initial_checkpoint
        training_args.evaluation.dataset_path = (
            training_args.evaluation.dataset_path if evaluation_protocol.evaluate_dataset else None
        )
        training_args.evaluation.reference_model_path = str(
            _resolve_source_path(evaluation_protocol.reference_model_path)
        )
        training_args.evaluation.opening_suite_path = str(_resolve_source_path(evaluation_protocol.opening_suite_path))
        training_args.evaluation.raw_results_path = str(
            _resolve_source_path(configuration.output_path) / Path(evaluation_protocol.raw_results_subdirectory)
        )
        training_args.evaluation.maximum_game_plies = evaluation_protocol.maximum_game_plies
        training_args.evaluation.bootstrap_seed = evaluation_protocol.bootstrap_seed
        training_args.evaluation.bootstrap_samples = evaluation_protocol.bootstrap_samples
        training_args.evaluation.mcts_threads = evaluation_protocol.mcts_threads
        training_args.evaluation.previous_model_offsets = ()
        training_args.evaluation.historical_model_iterations = ()
        training_args.evaluation.stockfish_skill_levels = ()
        training_args.evaluation.stockfish_binary_path = (
            evaluation_protocol.stockfish_binary_path if evaluation_protocol.stockfish_binary_path is not None else None
        )
        training_args.evaluation.stockfish_nodes_per_move = evaluation_protocol.stockfish_nodes_per_move
        training_args.evaluation.stockfish_threads = evaluation_protocol.stockfish_threads
        training_args.evaluation.stockfish_hash_mib = evaluation_protocol.stockfish_hash_mib
        training_args.evaluation.evaluate_random = False


def write_run_manifest(path: Path, manifest: RunManifest) -> RunManifest:
    serialized = manifest.model_dump_json(indent=2)
    if path.exists():
        existing = RunManifest.model_validate_json(path.read_text(encoding='utf-8'))
        existing_with_current_hardware = existing.model_copy(update={'resolved_hardware': manifest.resolved_hardware})
        if existing_with_current_hardware != manifest:
            raise ValueError(f'Existing run manifest does not match the requested run: {path}')
        return existing

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix('.json.tmp')
    temporary_path.write_text(serialized + '\n', encoding='utf-8')
    temporary_path.replace(path)
    return manifest


def prepare_training_run(
    training_args: TrainingArgs,
    configuration: RunConfiguration,
    expected_source_revision: str,
    approval_path: Path,
) -> RunManifest:
    resolved_hardware = _resolved_hardware()
    validate_run_configuration(configuration, resolved_hardware)

    source_revision = _git_output(['rev-parse', 'HEAD'])
    if source_revision != expected_source_revision:
        raise ValueError(f'Expected source revision {expected_source_revision}, found {source_revision}.')
    approval = load_approval_record(approval_path)
    validate_approval(configuration, approval, source_revision)
    if configuration.hardware.provider_name.casefold() == 'unconfirmed':
        raise ValueError('Hardware provider must be confirmed before training.')
    if configuration.hardware.offer_id.casefold() == 'unconfirmed':
        raise ValueError('Hardware offer ID must be confirmed before training.')

    environment = configuration.environment
    dependency_lock_path = _resolve_source_path(environment.dependency_lock_path)
    if _sha256(dependency_lock_path) != environment.dependency_lock_sha256:
        raise ValueError('Dependency lock SHA-256 does not match the run configuration.')
    actual_python_version = f'{sys.version_info.major}.{sys.version_info.minor}'
    if actual_python_version != environment.python_version:
        raise ValueError(f'Expected Python {environment.python_version}, found {actual_python_version}.')
    if torch.__version__ != environment.torch_version:
        raise ValueError(f'Expected PyTorch {environment.torch_version}, found {torch.__version__}.')
    if torch.version.cuda != environment.cuda_version:
        raise ValueError(f'Expected PyTorch CUDA {environment.cuda_version}, found {torch.version.cuda}.')
    if os.environ.get('TRAINING_RUNTIME_IMAGE') != environment.runtime_image:
        raise ValueError('Training runtime image does not match the approved configuration.')
    open_file_soft_limit = _open_file_soft_limit()
    if open_file_soft_limit < environment.minimum_open_file_soft_limit:
        raise ValueError(
            f'Open-file soft limit {open_file_soft_limit} is below the required '
            f'{environment.minimum_open_file_soft_limit}.'
        )
    if configuration.safety.maximum_open_file_count >= open_file_soft_limit:
        raise ValueError('The open-file safety stop must be lower than the process soft limit.')

    initial_model_path = _resolve_source_path(configuration.resume.model_path)
    if not initial_model_path.is_file():
        raise ValueError(f'Initial model does not exist: {initial_model_path}')

    opening_suite_path = _resolve_source_path(configuration.evaluation_protocol.opening_suite_path)
    openings = load_opening_suite(opening_suite_path)
    expected_evaluation_games = len(openings) * 2
    if configuration.workload.evaluation_games != expected_evaluation_games:
        raise ValueError(
            f'Configured evaluation requires {expected_evaluation_games} games for '
            f'{len(openings)} color-swapped openings, but '
            f'{configuration.workload.evaluation_games} games were requested.'
        )

    configured_stockfish_binary_path = configuration.evaluation_protocol.stockfish_binary_path
    stockfish_binary_path = (
        Path(configured_stockfish_binary_path) if configured_stockfish_binary_path is not None else None
    )
    if stockfish_binary_path is not None and not stockfish_binary_path.is_file():
        raise ValueError(f'Stockfish binary does not exist: {stockfish_binary_path}')

    source_worktree_clean = not bool(_git_output(['status', '--short']))
    if not source_worktree_clean:
        raise ValueError('Refusing to start training from a dirty source working tree.')
    manifest = RunManifest(
        configuration=configuration,
        approval=approval,
        resolved_hardware=resolved_hardware,
        source_revision=source_revision,
        source_worktree_clean=source_worktree_clean,
        initial_model_sha256=_sha256(initial_model_path),
        stockfish_binary_sha256=(_sha256(stockfish_binary_path) if stockfish_binary_path is not None else None),
        open_file_soft_limit=open_file_soft_limit,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
    )

    output_path = Path(training_args.save_path)
    manifest = write_run_manifest(output_path / 'run_manifest.json', manifest)

    initial_checkpoint_path = model_save_path(0, output_path)
    if not initial_checkpoint_path.exists():
        device = torch.device('cuda', configuration.topology.trainer_device_id)
        model = load_model(initial_model_path, training_args.network, device)
        optimizer = create_optimizer(model, configuration.resume.optimizer)
        save_model_and_optimizer(model, optimizer, 0, output_path)

    return manifest
