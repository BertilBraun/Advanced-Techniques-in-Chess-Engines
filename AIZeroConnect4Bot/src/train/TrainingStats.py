from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainingStats:
    policy_loss: float = 0.0
    value_loss: float = 0.0
    total_loss: float = 0.0
    num_batches: int = 0
    batch_size: int = 0

    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size

    def update(self, policy_loss: float, value_loss: float, total_loss: float) -> None:
        self.policy_loss += policy_loss
        self.value_loss += value_loss
        self.total_loss += total_loss
        self.num_batches += 1

    def __add__(self, other: TrainingStats) -> TrainingStats:
        ts = TrainingStats(self.batch_size)
        ts.policy_loss = self.policy_loss + other.policy_loss
        ts.value_loss = self.value_loss + other.value_loss
        ts.total_loss = self.total_loss + other.total_loss
        ts.num_batches = self.num_batches + other.num_batches
        return ts

    def __repr__(self) -> str:
        samples = self.num_batches * self.batch_size
        return f'Policy Loss: {self.policy_loss / samples:.4f}, Value Loss: {self.value_loss / samples:.4f}, Total Loss: {self.total_loss / samples:.4f}'
