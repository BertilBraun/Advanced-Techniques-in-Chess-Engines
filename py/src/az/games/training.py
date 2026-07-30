from __future__ import annotations

from typing import Protocol, TypeVar

from torch import Tensor, nn


BatchType = TypeVar('BatchType')
RecordType = TypeVar('RecordType')


class TrainingLoss(Protocol):
    @property
    def total(self) -> Tensor: ...


class GameTrainingModule(Protocol[BatchType, RecordType]):
    @property
    def model(self) -> nn.Module: ...

    def create_training_batch(
        self,
        records: tuple[RecordType, ...],
        augmentation_seeds: tuple[int, ...],
    ) -> BatchType: ...

    def move_batch(self, batch: BatchType) -> BatchType: ...

    def calculate_loss(self, batch: BatchType) -> TrainingLoss: ...

    def serialize_model(self) -> bytes: ...

    def restore_model(self, artifact: bytes) -> None: ...
