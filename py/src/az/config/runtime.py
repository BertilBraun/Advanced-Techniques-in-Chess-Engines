from __future__ import annotations

from enum import Enum

from pydantic import Field, PositiveInt, model_validator

from src.az.config.base import FrozenModel


class DeviceAssignment(FrozenModel):
    device_ids: tuple[int, ...]

    @model_validator(mode="after")
    def validate_device_ids(self) -> DeviceAssignment:
        if not self.device_ids:
            raise ValueError("At least one device ID is required.")
        if len(set(self.device_ids)) != len(self.device_ids):
            raise ValueError("Device IDs must be unique.")
        if min(self.device_ids) < 0:
            raise ValueError("Device IDs cannot be negative.")
        return self


class TopologyConfiguration(FrozenModel):
    trainer: DeviceAssignment
    self_play: DeviceAssignment
    evaluation: DeviceAssignment
    self_play_workers_per_device: PositiveInt
    search_threads_per_worker: PositiveInt
    optimizer_active_self_play_worker_ids: tuple[int, ...]
    maximum_active_searches_per_worker: PositiveInt
    inference_workers_per_device: PositiveInt
    inference_batch_size: PositiveInt
    maximum_pending_inference_batches: PositiveInt
    data_loader_workers_per_rank: int = Field(ge=0)
    evaluation_concurrency: PositiveInt

    @model_validator(mode="after")
    def validate_worker_phase_allocation(self) -> TopologyConfiguration:
        total_workers = self.self_play_worker_count
        active = self.optimizer_active_self_play_worker_ids
        if tuple(sorted(set(active))) != active:
            raise ValueError(
                "Optimizer-active self-play worker IDs must be unique and increasing."
            )
        if not active or active[-1] >= total_workers:
            raise ValueError(
                "Optimizer-active self-play worker IDs must identify configured workers."
            )
        for device_position in range(len(self.self_play.device_ids)):
            device_workers = set(self.worker_ids_for_device_position(device_position))
            if len(device_workers.intersection(active)) != 1:
                raise ValueError(
                    "Exactly one self-play worker per device must remain active during optimizer quanta."
                )
        return self

    @property
    def self_play_worker_count(self) -> int:
        return len(self.self_play.device_ids) * self.self_play_workers_per_device

    @property
    def optimizer_paused_self_play_worker_ids(self) -> tuple[int, ...]:
        active = set(self.optimizer_active_self_play_worker_ids)
        return tuple(
            worker_id
            for worker_id in range(self.self_play_worker_count)
            if worker_id not in active
        )

    def worker_ids_for_device_position(self, device_position: int) -> tuple[int, ...]:
        if not 0 <= device_position < len(self.self_play.device_ids):
            raise ValueError("Self-play device position is outside the topology.")
        first = device_position * self.self_play_workers_per_device
        return tuple(range(first, first + self.self_play_workers_per_device))


class TelemetryMetric(str, Enum):
    GAMES = "games"
    POSITIONS = "positions"
    ACTUAL_SIMULATIONS = "actual_simulations"
    BUDGET_CLASS = "budget_class"
    POLICY_ELIGIBILITY = "policy_eligibility"
    GPU_UTILIZATION = "gpu_utilization"
    OPTIMIZER_STEPS = "optimizer_steps"
    REPLAY_REUSE = "replay_reuse"
    STOP_REASON = "stop_reason"
    PREFIX_FULL_DISAGREEMENT = "prefix_full_disagreement"


class TelemetryConfiguration(FrozenModel):
    write_every_seconds: PositiveInt
    resource_sample_every_seconds: PositiveInt
    required_metrics: tuple[TelemetryMetric, ...] = Field(min_length=1)
    search_trace_sample_probability: float = Field(ge=0, le=1)
    search_trace_checkpoints: tuple[PositiveInt, ...]

    @model_validator(mode="after")
    def validate_metrics(self) -> TelemetryConfiguration:
        if len(set(self.required_metrics)) != len(self.required_metrics):
            raise ValueError("Required telemetry metrics must be unique.")
        if (
            tuple(sorted(set(self.search_trace_checkpoints)))
            != self.search_trace_checkpoints
        ):
            raise ValueError("Search trace checkpoints must increase strictly.")
        if (self.search_trace_sample_probability > 0) != bool(
            self.search_trace_checkpoints
        ):
            raise ValueError(
                "Search trace checkpoints are required exactly when trace sampling is enabled."
            )
        return self


class RetentionConfiguration(FrozenModel):
    recent_checkpoint_count: PositiveInt
    milestone_every_optimizer_steps: PositiveInt
    retain_replay_shards: bool
    retain_search_traces: bool
    retain_raw_evaluation_games: bool
