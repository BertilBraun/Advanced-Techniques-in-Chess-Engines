from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

from src.self_play.configuration import SdpaBackend


class InferenceTarget(str, Enum):
    AUTO = 'auto'
    CPU = 'cpu'
    CUDA = 'cuda'


DEFAULT_EXPLORATION_CONSTANT = 1.0


@dataclass(frozen=True)
class InteractiveEngineConfiguration:
    model_path: str
    device_id: int = 0
    parallel_searches: int = 64
    exploration_constant: float = DEFAULT_EXPLORATION_CONSTANT
    inference_workers: int = 2
    outstanding_batches_per_worker: int = 2
    maximum_batch_size: int | None = None
    inference_target: InferenceTarget = InferenceTarget.AUTO
    sdpa_backend: SdpaBackend = SdpaBackend.AUTOMATIC

    def __post_init__(self) -> None:
        if not self.model_path:
            raise ValueError('model_path must not be empty.')
        if self.parallel_searches < 1:
            raise ValueError('parallel_searches must be positive.')
        if not isfinite(self.exploration_constant) or self.exploration_constant <= 0:
            raise ValueError('exploration_constant must be finite and positive.')
        if self.inference_workers < 1:
            raise ValueError('inference_workers must be positive.')
        if not 1 <= self.outstanding_batches_per_worker <= 2:
            raise ValueError('outstanding_batches_per_worker must be in [1, 2].')
        if self.maximum_batch_size is not None and self.maximum_batch_size < 1:
            raise ValueError('maximum_batch_size must be positive when supplied.')

    @property
    def resolved_batch_size(self) -> int:
        return self.parallel_searches if self.maximum_batch_size is None else self.maximum_batch_size
