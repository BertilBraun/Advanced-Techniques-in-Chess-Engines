from __future__ import annotations

from dataclasses import dataclass

from src.util.tensorboard import log_scalar


@dataclass
class TrainingStats:
    _policy_loss: float = 0.0
    _value_loss: float = 0.0
    _total_loss: float = 0.0
    _value_mean: float = 0.0
    _value_std: float = 0.0
    _total_gradient_norm: float = 0.0
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

    @property
    def gradient_norm(self) -> float:
        return self._total_gradient_norm / self._num_batches

    def log_to_tensorboard(self, iteration: int, prefix: str) -> None:
        log_scalar(f'{prefix}/policy_loss', self.policy_loss, iteration)
        log_scalar(f'{prefix}/value_loss', self.value_loss, iteration)
        log_scalar(f'{prefix}/total_loss', self.total_loss, iteration)
        log_scalar(f'{prefix}/value_mean', self.value_mean, iteration)
        log_scalar(f'{prefix}/value_std', self.value_std, iteration)
        if self.gradient_norm > 0:
            # Only log gradient norm if it's greater than 0 to avoid cluttering the logs
            log_scalar(f'{prefix}/gradient_norm', self.gradient_norm, iteration)

    def __init__(
        self,
        policy_loss: float,
        value_loss: float,
        total_loss: float,
        value_mean: float,
        value_std: float,
        grad_norm: float,
        num_batches: int = 1,
    ) -> None:
        self._policy_loss += policy_loss
        self._value_loss += value_loss
        self._total_loss += total_loss
        self._value_mean += value_mean
        self._value_std += value_std
        self._total_gradient_norm += grad_norm
        self._num_batches += num_batches

    @staticmethod
    def combine(stats_list: list[TrainingStats]) -> TrainingStats:
        policy_loss = sum(stats._policy_loss for stats in stats_list)
        value_loss = sum(stats._value_loss for stats in stats_list)
        total_loss = sum(stats._total_loss for stats in stats_list)
        value_mean = sum(stats._value_mean for stats in stats_list)
        value_std = sum(stats._value_std for stats in stats_list)
        gradient_norm = sum(stats._total_gradient_norm for stats in stats_list)
        num_batches = sum(stats._num_batches for stats in stats_list)
        return TrainingStats(
            policy_loss=policy_loss,
            value_loss=value_loss,
            total_loss=total_loss,
            value_mean=value_mean,
            value_std=value_std,
            grad_norm=gradient_norm,
            num_batches=num_batches,
        )

    def __repr__(self) -> str:
        return f'Policy Loss: {self.policy_loss:.4f}, Value Loss: {self.value_loss:.4f}, Total Loss: {self.total_loss:.4f}, Value Mean: {self.value_mean:.4f}, Value Std: {self.value_std:.4f}, Gradient Norm: {self.gradient_norm:.4f}, Num Batches: {self._num_batches}'
