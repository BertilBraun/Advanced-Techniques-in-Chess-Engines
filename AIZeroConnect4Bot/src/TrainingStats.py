from __future__ import annotations

from dataclasses import dataclass, field


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


@dataclass
class LearningStats:
    total_num_games: int = 0
    total_iterations: int = 0
    training_stats: list[TrainingStats] = field(default_factory=lambda: [])

    def update(self, num_games: int, training_stats: TrainingStats) -> None:
        self.total_num_games += num_games
        self.training_stats.append(training_stats)
        self.total_iterations += 1

    def __repr__(self) -> str:
        res = f'Total Games: {self.total_num_games}\n'
        res += f'Total Iterations: {self.total_iterations}\n'
        for i, ts in enumerate(self.training_stats):
            res += f'Iteration {i + 1}: {ts}\n'
        return res
