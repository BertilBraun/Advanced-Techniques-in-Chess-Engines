from __future__ import annotations

from enum import Enum

from pydantic import Field, PositiveInt, model_validator

from src.az.config.base import FrozenModel


class DeviceAssignment(FrozenModel):
    device_ids: tuple[int, ...]

    @model_validator(mode='after')
    def validate_device_ids(self) -> DeviceAssignment:
        if not self.device_ids:
            raise ValueError('At least one device ID is required.')
        if len(set(self.device_ids)) != len(self.device_ids):
            raise ValueError('Device IDs must be unique.')
        if min(self.device_ids) < 0:
            raise ValueError('Device IDs cannot be negative.')
        return self


class TopologyConfiguration(FrozenModel):
    trainer: DeviceAssignment
    self_play: DeviceAssignment
    evaluation: DeviceAssignment
    self_play_workers_per_device: PositiveInt
    maximum_active_searches_per_worker: PositiveInt
    inference_workers_per_device: PositiveInt
    inference_batch_size: PositiveInt
    maximum_pending_inference_batches: PositiveInt
    data_loader_workers_per_rank: int = Field(ge=0)
    evaluation_concurrency: PositiveInt


class TelemetryMetric(str, Enum):
    GAMES = 'games'
    POSITIONS = 'positions'
    ACTUAL_SIMULATIONS = 'actual_simulations'
    BUDGET_CLASS = 'budget_class'
    POLICY_ELIGIBILITY = 'policy_eligibility'
    GPU_UTILIZATION = 'gpu_utilization'
    OPTIMIZER_STEPS = 'optimizer_steps'
    REPLAY_REUSE = 'replay_reuse'
    STOP_REASON = 'stop_reason'
    PREFIX_FULL_DISAGREEMENT = 'prefix_full_disagreement'


class TelemetryConfiguration(FrozenModel):
    write_every_seconds: PositiveInt
    resource_sample_every_seconds: PositiveInt
    required_metrics: tuple[TelemetryMetric, ...] = Field(min_length=1)
    search_trace_sample_probability: float = Field(ge=0, le=1)

    @model_validator(mode='after')
    def validate_metrics(self) -> TelemetryConfiguration:
        if len(set(self.required_metrics)) != len(self.required_metrics):
            raise ValueError('Required telemetry metrics must be unique.')
        return self


class RetentionConfiguration(FrozenModel):
    recent_checkpoint_count: PositiveInt
    milestone_every_optimizer_steps: PositiveInt
    retain_replay_shards: bool
    retain_search_traces: bool
    retain_raw_evaluation_games: bool
