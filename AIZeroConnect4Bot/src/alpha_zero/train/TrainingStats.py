from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrainingStats:
    _policy_loss: list[float] = field(default_factory=list)
    _value_loss: list[float] = field(default_factory=list)
    _total_loss: list[float] = field(default_factory=list)

    @property
    def policy_loss(self) -> float:
        return sum(self._policy_loss) / len(self._policy_loss)

    @property
    def value_loss(self) -> float:
        return sum(self._value_loss) / len(self._value_loss)

    @property
    def total_loss(self) -> float:
        return sum(self._total_loss) / len(self._total_loss)

    def update(self, policy_loss: float, value_loss: float, total_loss: float, num_batches: int = 1) -> None:
        self._policy_loss.append(policy_loss)
        self._value_loss.append(value_loss)
        self._total_loss.append(total_loss)

    def __add__(self, other: TrainingStats) -> TrainingStats:
        return TrainingStats(
            self._policy_loss + other._policy_loss,
            self._value_loss + other._value_loss,
            self._total_loss + other._total_loss,
        )

    def __repr__(self) -> str:
        return (
            f'Policy Loss: {self.policy_loss:.4f}, Value Loss: {self.value_loss:.4f}, Total Loss: {self.total_loss:.4f}'
        )
