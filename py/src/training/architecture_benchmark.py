from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from src.util.frozen_model import FrozenModel


class ComparisonProtocol(str, Enum):
    EQUAL_SAMPLES = 'equal_samples'
    EQUAL_WALL_TIME = 'equal_wall_time'


class AttentionBackend(str, Enum):
    AUTOMATIC = 'automatic'
    FLASH = 'flash'
    MEMORY_EFFICIENT = 'memory_efficient'
    CUDNN = 'cudnn'
    MATH = 'math'


class ProductionTopology(FrozenModel):
    trainer_device_ids: tuple[int, ...] = Field(min_length=1)
    global_training_batch_size: int = Field(gt=0)
    local_training_batch_size: int = Field(gt=0)
    self_play_processes_per_device: int = Field(gt=0)
    inference_workers_per_process: int = Field(gt=0)
    outstanding_batches_per_worker: int = Field(gt=0)
    production_inference_batch_size: int = Field(gt=0)

    @model_validator(mode='after')
    def validate_training_batch_partition(self) -> ProductionTopology:
        if any(device_id < 0 for device_id in self.trainer_device_ids):
            raise ValueError('Benchmark trainer device IDs must be nonnegative.')
        if len(set(self.trainer_device_ids)) != len(self.trainer_device_ids):
            raise ValueError('Benchmark trainer device IDs must be unique.')
        if self.global_training_batch_size != self.local_training_batch_size * len(self.trainer_device_ids):
            raise ValueError('Global benchmark batch must equal local batch times trainer ranks.')
        return self


class TrainingBenchmarkConfiguration(FrozenModel):
    warmup_optimizer_steps: int = Field(ge=0)
    equal_sample_optimizer_steps: int = Field(gt=0)
    equal_wall_time_seconds: float = Field(gt=0.0)
    precision: Literal['bfloat16']


class InferenceBenchmarkConfiguration(FrozenModel):
    batch_sizes: tuple[int, ...] = Field(min_length=1)
    warmup_batches: int = Field(ge=0)
    measured_batches: int = Field(gt=0)

    @model_validator(mode='after')
    def validate_batch_sizes(self) -> InferenceBenchmarkConfiguration:
        if any(batch_size <= 0 for batch_size in self.batch_sizes):
            raise ValueError('Inference benchmark batch sizes must be positive.')
        if len(set(self.batch_sizes)) != len(self.batch_sizes):
            raise ValueError('Inference benchmark batch sizes must be unique.')
        return self


class ArchitectureBenchmarkPlan(FrozenModel):
    schema_version: Literal[1]
    catalog_path: Path
    topology: ProductionTopology
    training: TrainingBenchmarkConfiguration
    inference: InferenceBenchmarkConfiguration

    @model_validator(mode='after')
    def validate_production_inference_batch(self) -> ArchitectureBenchmarkPlan:
        if self.topology.production_inference_batch_size not in self.inference.batch_sizes:
            raise ValueError('Inference benchmark sizes must include the production batch size.')
        return self


def load_architecture_benchmark_plan(path: Path) -> ArchitectureBenchmarkPlan:
    plan = ArchitectureBenchmarkPlan.model_validate(yaml.safe_load(path.read_text(encoding='utf-8')))
    return plan.validated_copy(update={'catalog_path': str((path.parent / plan.catalog_path).resolve())})
