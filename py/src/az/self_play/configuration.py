from __future__ import annotations

from uuid import UUID

from pydantic import Field, model_validator

from src.az.config.base import FrozenModel, Sha256
from src.az.config.search import (
    FpuConfiguration,
    RootExplorationConfiguration,
    SearchBudgetConfiguration,
    SearchStoppingConfiguration,
    TemperatureConfiguration,
)
from src.az.games.go.configuration import (
    DisabledResignation,
    GoGameConfiguration,
    ResidualGoModelConfiguration,
)


class NativeSearchSpecification(FrozenModel):
    budget: SearchBudgetConfiguration
    stopping: SearchStoppingConfiguration
    fpu: FpuConfiguration
    exploration_constant: float = Field(gt=0)
    backup_discount: float = Field(gt=0, le=1)
    temperature: TemperatureConfiguration
    root_exploration: RootExplorationConfiguration


class GoWorkerSpecification(FrozenModel):
    worker_index: int = Field(ge=0)
    process_index: int = Field(ge=0)
    run_id: UUID
    root_seed: int = Field(ge=0, le=2**63 - 1)
    game_configuration: GoGameConfiguration
    model_configuration: ResidualGoModelConfiguration
    model_initialization_seed: int = Field(ge=0, le=2**63 - 1)
    search: NativeSearchSpecification
    logical_worker_start_index: int = Field(ge=0)
    logical_worker_count: int = Field(gt=0)
    maximum_active_searches_per_worker: int = Field(gt=0)
    maximum_batch_size: int = Field(gt=0)
    maximum_wait_microseconds: int = Field(ge=0)
    maximum_pending_batches: int = Field(gt=0)
    inference_cache_capacity: int = Field(ge=0)
    value_target_weight: float = Field(gt=0)
    device: str = Field(pattern=r'^(cpu|cuda:[0-9]+)$')
    checkpoint_directory: str = Field(min_length=1)
    resolved_configuration_sha256: Sha256
    telemetry_write_every_seconds: int = Field(gt=0)
    resource_sample_every_seconds: int = Field(gt=0)

    @model_validator(mode='after')
    def validate_supported_runtime_features(self) -> GoWorkerSpecification:
        match self.game_configuration.resignation:
            case DisabledResignation():
                return self
            case _:
                raise ValueError('Stage 7 native Go self-play does not implement resignation.')
