from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainingStats:
    policy_loss: float = 0.0
    value_loss: float = 0.0
    total_loss: float = 0.0
    num_batches: int = 0

    def update(self, policy_loss: float, value_loss: float, total_loss: float) -> None:
        self.policy_loss += policy_loss
        self.value_loss += value_loss
        self.total_loss += total_loss
        self.num_batches += 1

    def __add__(self, other: TrainingStats) -> TrainingStats:
        return TrainingStats(
            self.policy_loss + other.policy_loss,
            self.value_loss + other.value_loss,
            self.total_loss + other.total_loss,
            self.num_batches + other.num_batches,
        )

    def __repr__(self) -> str:
        return f'Policy Loss: {self.policy_loss / self.num_batches:.4f}, Value Loss: {self.value_loss / self.num_batches:.4f}, Total Loss: {self.total_loss / self.num_batches:.4f}'
