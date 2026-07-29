from __future__ import annotations

from pathlib import PurePosixPath
from typing import Literal

from pydantic import Field, PositiveFloat, PositiveInt, model_validator

from src.az.config.base import DeterminismMode, FrozenModel


class ManifestPolicy(FrozenModel):
    require_clean_source: bool
    record_dependency_versions: bool
    determinism_mode: DeterminismMode


class ExperimentConfiguration(FrozenModel):
    name: str = Field(min_length=1)
    arm_id: str = Field(min_length=1)
    hypothesis: str = Field(min_length=1)
    root_seed: int = Field(ge=0, le=2**63 - 1)
    duration_seconds: PositiveInt
    checkpoint_elapsed_seconds: tuple[PositiveInt, ...]
    output_directory: PurePosixPath
    manifest_policy: ManifestPolicy

    @model_validator(mode='after')
    def validate_checkpoints(self) -> ExperimentConfiguration:
        if tuple(sorted(set(self.checkpoint_elapsed_seconds))) != self.checkpoint_elapsed_seconds:
            raise ValueError('Checkpoint elapsed times must be unique and strictly increasing.')
        if self.checkpoint_elapsed_seconds and self.checkpoint_elapsed_seconds[-1] > self.duration_seconds:
            raise ValueError('Checkpoint elapsed times cannot exceed the experiment duration.')
        return self


class HardwareConfiguration(FrozenModel):
    profile_name: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    offer_id: str = Field(min_length=1)
    expected_gpu_model: str = Field(min_length=1)
    expected_gpu_count: PositiveInt
    minimum_logical_cpu_count: PositiveInt
    minimum_ram_gib: PositiveFloat
    minimum_free_disk_gib: PositiveFloat
    hourly_cost: float | None = Field(ge=0)
    currency: Literal['EUR', 'USD']
