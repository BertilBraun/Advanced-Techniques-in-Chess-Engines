from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
import torch.distributed as distributed
from pydantic import Field, model_validator

from src.az.config.base import FrozenModel
from src.az.config.base import DeterminismMode as RunDeterminismMode


class DistributedBackend(str, Enum):
    GLOO = 'gloo'
    NCCL = 'nccl'


class TrainingDeterminism(str, Enum):
    STRICT = 'strict'
    SEEDED_CONCURRENT = 'seeded_concurrent'
    BEST_EFFORT_CUDA = 'best_effort_cuda'


class ProcessGroupLifecycle(FrozenModel):
    backend: DistributedBackend
    rank: int = Field(ge=0)
    world_size: int = Field(gt=0)
    initialized: bool

    @model_validator(mode='after')
    def validate_rank(self) -> ProcessGroupLifecycle:
        if self.rank >= self.world_size:
            raise ValueError('Process-group rank must be within the distributed world.')
        if self.world_size > 1 and not self.initialized:
            raise ValueError('A multi-rank checkpoint requires an initialized process group.')
        return self


@dataclass(frozen=True)
class TrainingRank:
    rank: int
    world_size: int
    device: torch.device
    backend: DistributedBackend = DistributedBackend.GLOO

    def __post_init__(self) -> None:
        if self.rank < 0 or self.world_size <= 0 or self.rank >= self.world_size:
            raise ValueError('Training rank must be within the distributed world.')
        if self.device.type not in ('cpu', 'cuda'):
            raise ValueError('Training devices must be CPU or CUDA.')
        if self.backend is DistributedBackend.NCCL and self.device.type != 'cuda':
            raise ValueError('NCCL requires a CUDA training device.')

    def training_determinism(
        self,
        run_mode: RunDeterminismMode,
    ) -> TrainingDeterminism:
        if run_mode is RunDeterminismMode.STRICT_SINGLE_THREAD:
            if self.world_size != 1 or self.device.type != 'cpu':
                raise ValueError('Strict-single-thread manifest mode requires one CPU trainer rank.')
            return TrainingDeterminism.STRICT
        if self.device.type == 'cuda':
            return TrainingDeterminism.BEST_EFFORT_CUDA
        return TrainingDeterminism.SEEDED_CONCURRENT

    def lifecycle(self) -> ProcessGroupLifecycle:
        initialized = distributed.is_available() and distributed.is_initialized()
        if self.world_size > 1:
            if not initialized:
                raise ValueError('Multi-rank training requires an initialized process group.')
            if distributed.get_rank() != self.rank or distributed.get_world_size() != self.world_size:
                raise ValueError('Initialized process group does not match the configured training rank.')
            if distributed.get_backend() != self.backend.value:
                raise ValueError('Initialized process-group backend does not match configuration.')
        return ProcessGroupLifecycle(
            backend=self.backend,
            rank=self.rank,
            world_size=self.world_size,
            initialized=initialized,
        )
