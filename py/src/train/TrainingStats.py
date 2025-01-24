from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainingStats:
    _policy_loss: float = 0.0
    _value_loss: float = 0.0
    _total_loss: float = 0.0
    _value_mean: float = 0.0
    _value_std: float = 0.0
    _num_batches: int = 0

    @property
    def policy_loss(self) -> float:
        return self._policy_loss / self._num_batches

    @property
    def value_loss(self) -> float:
        return self._value_loss / self._num_batches

    @property
    def total_loss(self) -> float:
        return self._total_loss / self._num_batches

    @property
    def value_mean(self) -> float:
        return self._value_mean / self._num_batches

    @property
    def value_std(self) -> float:
        return self._value_std / self._num_batches

    def update(
        self,
        policy_loss: float,
        value_loss: float,
        total_loss: float,
        value_mean: float,
        value_std: float,
        num_batches: int = 1,
    ) -> None:
        self._policy_loss += policy_loss
        self._value_loss += value_loss
        self._total_loss += total_loss
        self._value_mean += value_mean
        self._value_std += value_std
        self._num_batches += num_batches

    def __add__(self, other: TrainingStats) -> TrainingStats:
        return TrainingStats(
            self._policy_loss + other._policy_loss,
            self._value_loss + other._value_loss,
            self._total_loss + other._total_loss,
            self._value_mean + other._value_mean,
            self._value_std + other._value_std,
            self._num_batches + other._num_batches,
        )

    def __repr__(self) -> str:
        return f'Policy Loss: {self.policy_loss:.4f}, Value Loss: {self.value_loss:.4f}, Total Loss: {self.total_loss:.4f}, Value Mean: {self.value_mean:.4f}, Value Std: {self.value_std:.4f}'
