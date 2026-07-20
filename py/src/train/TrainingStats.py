from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from src.util.tensorboard import log_scalar


@dataclass(frozen=True)
class TrainingStats:
    policy_loss_sum: float
    value_loss_sum: float
    total_loss_sum: float
    sample_count: int
    value_sum: float
    value_square_sum: float
    gradient_norm_sum: float
    gradient_norm_count: int
    num_batches: int

    @property
    def policy_loss(self) -> float:
        return self.policy_loss_sum / self.sample_count if self.sample_count else 0.0

    @property
    def value_loss(self) -> float:
        return self.value_loss_sum / self.sample_count if self.sample_count else 0.0

    @property
    def total_loss(self) -> float:
        return self.total_loss_sum / self.sample_count if self.sample_count else 0.0

    @property
    def value_mean(self) -> float:
        return self.value_sum / self.sample_count if self.sample_count else 0.0

    @property
    def value_std(self) -> float:
        if self.sample_count < 2:
            return 0.0
        centered_square_sum = self.value_square_sum - self.value_sum**2 / self.sample_count
        return sqrt(max(0.0, centered_square_sum / (self.sample_count - 1)))

    @property
    def gradient_norm(self) -> float:
        return self.gradient_norm_sum / self.gradient_norm_count if self.gradient_norm_count else 0.0

    def log_to_tensorboard(self, iteration: int, prefix: str) -> None:
        log_scalar(f'{prefix}/policy_loss', self.policy_loss, iteration)
        log_scalar(f'{prefix}/value_loss', self.value_loss, iteration)
        log_scalar(f'{prefix}/total_loss', self.total_loss, iteration)
        log_scalar(f'{prefix}/value_mean', self.value_mean, iteration)
        log_scalar(f'{prefix}/value_std', self.value_std, iteration)
        if self.gradient_norm > 0:
            log_scalar(f'{prefix}/gradient_norm', self.gradient_norm, iteration)

    def __repr__(self) -> str:
        return (
            f'Policy Loss: {self.policy_loss:.4f}, Value Loss: {self.value_loss:.4f}, '
            f'Total Loss: {self.total_loss:.4f}, Value Mean: {self.value_mean:.4f}, '
            f'Value Std: {self.value_std:.4f}, Gradient Norm: {self.gradient_norm:.4f}, '
            f'Num Batches: {self.num_batches}, Samples: {self.sample_count}'
        )

    @staticmethod
    def combine(stats_list: list[TrainingStats]) -> TrainingStats:
        if not stats_list:
            raise ValueError('At least one training-statistics value is required.')
        return TrainingStats(
            policy_loss_sum=sum(stats.policy_loss_sum for stats in stats_list),
            value_loss_sum=sum(stats.value_loss_sum for stats in stats_list),
            total_loss_sum=sum(stats.total_loss_sum for stats in stats_list),
            sample_count=sum(stats.sample_count for stats in stats_list),
            value_sum=sum(stats.value_sum for stats in stats_list),
            value_square_sum=sum(stats.value_square_sum for stats in stats_list),
            gradient_norm_sum=sum(stats.gradient_norm_sum for stats in stats_list),
            gradient_norm_count=sum(stats.gradient_norm_count for stats in stats_list),
            num_batches=sum(stats.num_batches for stats in stats_list),
        )
