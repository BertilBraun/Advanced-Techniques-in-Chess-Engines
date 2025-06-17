from __future__ import annotations
from typing import Optional

import torch

from src.util.tensorboard import log_scalar


class TrainingStats:
    """Training statistics collector and calculator"""

    def __init__(self, device: torch.device) -> None:
        self.device: torch.device = device
        self.reset()

    def reset(self) -> None:
        """Reset all accumulated statistics"""
        self.policy_losses: list[torch.Tensor] = []
        self.value_losses: list[torch.Tensor] = []
        self.total_losses: list[torch.Tensor] = []
        self.value_outputs: list[torch.Tensor] = []
        self.gradient_norms: list[float] = []

    def add_batch(
        self,
        policy_loss: torch.Tensor,
        value_loss: torch.Tensor,
        total_loss: torch.Tensor,
        value_output: torch.Tensor,
        gradient_norm: Optional[float] = None,
    ) -> None:
        """Add statistics from a single batch"""
        self.policy_losses.append(policy_loss.detach())
        self.value_losses.append(value_loss.detach())
        self.total_losses.append(total_loss.detach())
        self.value_outputs.append(value_output.detach())
        if gradient_norm is not None and gradient_norm > 0:
            self.gradient_norms.append(gradient_norm)

    @property
    def policy_loss(self) -> float:
        """Average policy loss across all batches"""
        if not self.policy_losses:
            return 0.0
        return torch.stack(self.policy_losses).mean().item()

    @property
    def value_loss(self) -> float:
        """Average value loss across all batches"""
        if not self.value_losses:
            return 0.0
        return torch.stack(self.value_losses).mean().item()

    @property
    def total_loss(self) -> float:
        """Average total loss across all batches"""
        if not self.total_losses:
            return 0.0
        return torch.stack(self.total_losses).mean().item()

    @property
    def value_mean(self) -> float:
        """Mean of all value outputs"""
        if not self.value_outputs:
            return 0.0
        return torch.cat(self.value_outputs, dim=0).mean().item()

    @property
    def value_std(self) -> float:
        """Standard deviation of all value outputs"""
        if not self.value_outputs:
            return 0.0
        return torch.cat(self.value_outputs, dim=0).std().item()

    @property
    def gradient_norm(self) -> float:
        """Average gradient norm across all batches"""
        if not self.gradient_norms:
            return 0.0
        return sum(self.gradient_norms) / len(self.gradient_norms)

    @property
    def num_batches(self) -> int:
        """Number of batches processed"""
        return len(self.policy_losses)

    def log_to_tensorboard(self, iteration: int, prefix: str) -> None:
        """Log statistics to tensorboard"""
        log_scalar(f'{prefix}/policy_loss', self.policy_loss, iteration)
        log_scalar(f'{prefix}/value_loss', self.value_loss, iteration)
        log_scalar(f'{prefix}/total_loss', self.total_loss, iteration)
        log_scalar(f'{prefix}/value_mean', self.value_mean, iteration)
        log_scalar(f'{prefix}/value_std', self.value_std, iteration)
        if self.gradient_norm > 0:
            # Only log gradient norm if it's greater than 0 to avoid cluttering the logs
            log_scalar(f'{prefix}/gradient_norm', self.gradient_norm, iteration)

    def __repr__(self) -> str:
        return f'Policy Loss: {self.policy_loss:.4f}, Value Loss: {self.value_loss:.4f}, Total Loss: {self.total_loss:.4f}, Value Mean: {self.value_mean:.4f}, Value Std: {self.value_std:.4f}, Gradient Norm: {self.gradient_norm:.4f}, Num Batches: {self.num_batches}'

    @staticmethod
    def combine(stats_list: list[TrainingStats]) -> TrainingStats:
        """Combine multiple TrainingStats objects into one"""
        combined = TrainingStats(stats_list[0].device)
        for stats in stats_list:
            combined.policy_losses.extend(stats.policy_losses)
            combined.value_losses.extend(stats.value_losses)
            combined.total_losses.extend(stats.total_losses)
            combined.value_outputs.extend(stats.value_outputs)
            combined.gradient_norms.extend(stats.gradient_norms)
        return combined
