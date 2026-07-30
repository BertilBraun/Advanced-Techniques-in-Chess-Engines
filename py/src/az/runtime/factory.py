from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from src.az.config.model import FixedModelSchedule
from src.az.config.root import ResolvedRunConfiguration
from src.az.config.runtime import TelemetryMetric
from src.az.config.search import (
    DisabledTreeReuse,
    FullBudgetStopping,
    VisitMarginAdaptiveRule,
)
from src.az.config.seeds import (
    ModelInitializationSeedCoordinates,
    SeedPurpose,
    derive_seed,
)
from src.az.config.training import InitialStateOnly
from src.az.games.go.configuration import DisabledResignation
from src.az.runtime.topology import RuntimeTopology, WorkerAssignment
from src.az.self_play.configuration import (
    NativeSearchSpecification,
    GoWorkerSpecification,
)

STAGE7_DERIVABLE_METRICS = frozenset(
    {
        TelemetryMetric.GAMES,
        TelemetryMetric.POSITIONS,
        TelemetryMetric.ACTUAL_SIMULATIONS,
        TelemetryMetric.BUDGET_CLASS,
        TelemetryMetric.POLICY_ELIGIBILITY,
        TelemetryMetric.OPTIMIZER_STEPS,
        TelemetryMetric.REPLAY_REUSE,
        TelemetryMetric.STOP_REASON,
    }
)


@dataclass(frozen=True)
class RuntimeBuildEnvironment:
    run_id: UUID
    resolved_configuration_sha256: str
    output_directory: Path
    checkpoint_directory: Path
    startup_timeout_seconds: float
    shutdown_grace_seconds: float
    visible_cuda_models: tuple[str, ...]
    logical_cpu_count: int
    ram_gib: float
    free_disk_gib: float
    allow_cpu_smoke: bool

    def __post_init__(self) -> None:
        if not self.output_directory.is_absolute() or not self.checkpoint_directory.is_absolute():
            raise ValueError('Runtime output and checkpoint directories must be absolute.')
        if self.startup_timeout_seconds <= 0 or self.shutdown_grace_seconds <= 0:
            raise ValueError('Runtime startup timeout and shutdown grace must be positive.')
        if self.logical_cpu_count <= 0 or self.ram_gib <= 0 or self.free_disk_gib <= 0:
            raise ValueError('Runtime hardware observations must be positive.')


@dataclass(frozen=True)
class RuntimePlan:
    topology: RuntimeTopology
    worker_specifications: tuple[GoWorkerSpecification, ...]
    replay_directory: Path
    telemetry_path: Path
    duration_seconds: int
    startup_timeout_seconds: float
    shutdown_grace_seconds: float
    games_per_shard: int
    telemetry_write_every_seconds: int
    resource_sample_every_seconds: int
    required_metrics: tuple[TelemetryMetric, ...]
    search_trace_sample_probability: float


def build_runtime_plan(
    configuration: ResolvedRunConfiguration,
    environment: RuntimeBuildEnvironment,
) -> RuntimePlan:
    match configuration.model.schedule:
        case FixedModelSchedule(architecture=architecture):
            pass
        case _:
            raise ValueError('Stage 7 runtime supports only a fixed model schedule.')
    match configuration.search.tree_reuse:
        case DisabledTreeReuse():
            pass
        case _:
            raise ValueError('Stage 7 runtime requires disabled tree reuse.')
    match configuration.self_play.start_states:
        case InitialStateOnly():
            pass
    match configuration.game.resignation:
        case DisabledResignation():
            pass
        case _:
            raise ValueError('Stage 7 runtime does not implement resignation.')
    if configuration.topology.inference_workers_per_device != 1:
        raise ValueError('Stage 7 uses exactly one shared inference owner per self-play device.')
    if configuration.search.inference.maximum_batch_size != configuration.topology.inference_batch_size:
        raise ValueError('Search and topology inference batch sizes must match.')
    maximum_shard_positions = configuration.self_play.games_per_shard * configuration.game.safety_ply_cap
    if maximum_shard_positions > configuration.replay.maximum_positions_per_shard:
        raise ValueError('Replay shard capacity cannot hold the configured games per shard.')
    if configuration.replay.compression != 'none':
        raise ValueError('Stage 7 replay publication supports only explicit uncompressed shards.')
    if configuration.telemetry.search_trace_sample_probability != 0:
        raise ValueError('Search trace sampling is reserved for Stage 9.')
    match configuration.search.stopping:
        case VisitMarginAdaptiveRule(calibration_id=calibration_id) if calibration_id.startswith('placeholder'):
            raise ValueError('Adaptive search requires a non-placeholder calibration identifier.')
        case VisitMarginAdaptiveRule() | FullBudgetStopping():
            pass
    unsupported_metrics = set(configuration.telemetry.required_metrics) - STAGE7_DERIVABLE_METRICS
    if unsupported_metrics:
        unsupported = ', '.join(sorted(metric.value for metric in unsupported_metrics))
        raise ValueError(f'Required telemetry metrics are not derivable in Stage 7: {unsupported}.')
    if environment.logical_cpu_count < configuration.hardware.minimum_logical_cpu_count:
        raise ValueError('Observed logical CPU count is below the resolved hardware profile.')
    if environment.ram_gib < configuration.hardware.minimum_ram_gib:
        raise ValueError('Observed RAM is below the resolved hardware profile.')
    if environment.free_disk_gib < configuration.hardware.minimum_free_disk_gib:
        raise ValueError('Observed free disk is below the resolved hardware profile.')
    if not environment.allow_cpu_smoke:
        if len(environment.visible_cuda_models) < configuration.hardware.expected_gpu_count:
            raise ValueError('Visible CUDA device count is below the resolved hardware profile.')
        expected = configuration.hardware.expected_gpu_model
        if any(
            model != expected for model in environment.visible_cuda_models[: configuration.hardware.expected_gpu_count]
        ):
            raise ValueError('Visible CUDA model does not match the resolved hardware profile.')
    model_seed = derive_seed(
        configuration.experiment.root_seed,
        ModelInitializationSeedCoordinates(
            purpose=SeedPurpose.MODEL_INITIALIZATION,
            model_stage=0,
        ),
    )
    assignments: list[WorkerAssignment] = []
    specifications: list[GoWorkerSpecification] = []
    for worker_index, device_index in enumerate(configuration.topology.self_play.device_ids):
        assignments.append(
            WorkerAssignment(
                worker_index=worker_index,
                device_index=None if environment.allow_cpu_smoke else device_index,
                maximum_active_searches=(
                    configuration.topology.self_play_workers_per_device
                    * configuration.topology.maximum_active_searches_per_worker
                ),
            )
        )
        specifications.append(
            GoWorkerSpecification(
                worker_index=worker_index,
                process_index=worker_index,
                run_id=environment.run_id,
                root_seed=configuration.experiment.root_seed,
                game_configuration=configuration.game,
                model_configuration=architecture,
                model_initialization_seed=model_seed,
                search=NativeSearchSpecification(
                    budget=configuration.search.budget,
                    stopping=configuration.search.stopping,
                    fpu=configuration.search.fpu,
                    exploration_constant=configuration.search.algorithm.exploration_constant,
                    backup_discount=configuration.search.backup_discount,
                    temperature=configuration.search.temperature,
                    root_exploration=configuration.search.root_exploration,
                ),
                logical_worker_start_index=(worker_index * configuration.topology.self_play_workers_per_device),
                logical_worker_count=configuration.topology.self_play_workers_per_device,
                maximum_active_searches_per_worker=configuration.topology.maximum_active_searches_per_worker,
                maximum_batch_size=configuration.search.inference.maximum_batch_size,
                maximum_wait_microseconds=configuration.search.inference.maximum_wait_microseconds,
                maximum_pending_batches=configuration.topology.maximum_pending_inference_batches,
                inference_cache_capacity=configuration.search.inference.cache_capacity,
                value_target_weight=configuration.self_play.value_target_weight,
                device='cpu' if environment.allow_cpu_smoke else f'cuda:{device_index}',
                checkpoint_directory=str(environment.checkpoint_directory),
                resolved_configuration_sha256=environment.resolved_configuration_sha256,
                telemetry_write_every_seconds=configuration.telemetry.write_every_seconds,
                resource_sample_every_seconds=configuration.telemetry.resource_sample_every_seconds,
            )
        )
    topology = RuntimeTopology(
        workers=tuple(assignments),
        trainer_device_indices=(() if environment.allow_cpu_smoke else configuration.topology.trainer.device_ids),
        evaluation_device_indices=(() if environment.allow_cpu_smoke else configuration.topology.evaluation.device_ids),
    )
    try:
        replay_relative = configuration.replay.shard_directory.relative_to(configuration.experiment.output_directory)
    except ValueError as error:
        raise ValueError('Replay shard directory must be below the experiment output directory.') from error
    return RuntimePlan(
        topology=topology,
        worker_specifications=tuple(specifications),
        replay_directory=environment.output_directory.joinpath(*replay_relative.parts),
        telemetry_path=environment.output_directory / 'runtime-telemetry.azt',
        duration_seconds=configuration.experiment.duration_seconds,
        startup_timeout_seconds=environment.startup_timeout_seconds,
        shutdown_grace_seconds=environment.shutdown_grace_seconds,
        games_per_shard=configuration.self_play.games_per_shard,
        telemetry_write_every_seconds=configuration.telemetry.write_every_seconds,
        resource_sample_every_seconds=configuration.telemetry.resource_sample_every_seconds,
        required_metrics=configuration.telemetry.required_metrics,
        search_trace_sample_probability=configuration.telemetry.search_trace_sample_probability,
    )
