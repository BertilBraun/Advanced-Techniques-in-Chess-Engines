from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path

from pydantic import Field, model_validator
from src.util.frozen_model import FrozenModel


class TrainingStage(str, Enum):
    SYSTEMS_PILOT = 'systems_pilot'
    CONTINUATION = 'continuation'
    CLEAN_RETRAIN = 'clean_retrain'


class HardwareConfiguration(FrozenModel):
    provider_name: str = Field(min_length=1)
    offer_id: str = Field(min_length=1)
    gpu_model: str = Field(min_length=1)
    gpu_count: int = Field(gt=0)
    logical_cpu_count: int = Field(gt=0)
    minimum_ram_gib: int = Field(gt=0)
    minimum_disk_gib: int = Field(gt=0)


class EnvironmentConfiguration(FrozenModel):
    runtime_image: str
    python_version: str
    torch_version: str
    cuda_version: str
    dependency_lock_path: str
    dependency_lock_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    minimum_open_file_soft_limit: int = Field(gt=0)


class ResolvedHardware(FrozenModel):
    visible_gpu_names: tuple[str, ...]
    visible_gpu_count: int
    logical_cpu_count: int
    total_ram_gib: float
    free_disk_gib: float


class ApprovalRecord(FrozenModel):
    approved_by: str = Field(min_length=1)
    approved_at_utc: datetime
    source_revision: str = Field(pattern=r'^[0-9a-f]{40}$')
    configuration_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    maximum_cost: float | None

    @model_validator(mode='after')
    def validate_approval_timestamp(self) -> ApprovalRecord:
        if self.approved_at_utc.tzinfo is None or self.approved_at_utc.utcoffset() is None:
            raise ValueError('approved_at_utc must include a timezone.')
        return self


def load_approval_record(path: Path) -> ApprovalRecord:
    return ApprovalRecord.model_validate_json(path.read_text(encoding='utf-8'))
