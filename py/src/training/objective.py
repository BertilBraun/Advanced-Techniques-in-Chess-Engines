from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from src.training.batch import TrainingBatch, TrainingModelOutput
from src.util.frozen_model import FrozenModel


def _scalar_to_wdl(scores: torch.Tensor) -> torch.Tensor:
    """Represent scores in [-1, 1] with residual probability shared across WDL."""
    wins = torch.clamp(scores, min=0.0)
    losses = torch.clamp(-scores, min=0.0)
    remainders = 1.0 - torch.abs(scores)
    return torch.stack((wins + remainders / 3, remainders / 3, losses + remainders / 3), dim=-1)


@dataclass(frozen=True)
class ObjectiveLoss:
    policy: torch.Tensor
    wdl: torch.Tensor
    auxiliary: tuple[torch.Tensor, ...]
    total: torch.Tensor


class ResolvedTrainingObjective(FrozenModel):
    policy_loss_weight: float
    value_loss_weight: float
    root_value_blend: float
    auxiliary_loss_weights: tuple[float, ...]

    def calculate_loss(self, output: TrainingModelOutput, batch: TrainingBatch) -> ObjectiveLoss:
        auxiliary_count = len(self.auxiliary_loss_weights)
        if not (
            len(output.auxiliary_logits)
            == len(batch.auxiliary_targets)
            == len(batch.auxiliary_eligibility)
            == auxiliary_count
        ):
            raise ValueError('Model outputs, batch targets, eligibility masks, and objective layout must agree.')
        sample_weights = batch.sample_weights / batch.sample_weights.mean()
        policy_rows = functional.cross_entropy(output.policy_logits, batch.policy_targets, reduction='none')
        policy_loss = (policy_rows * sample_weights).mean()
        blended_wdl = torch.lerp(
            batch.wdl_targets,
            _scalar_to_wdl(batch.root_values),
            self.root_value_blend,
        )
        wdl_rows = functional.cross_entropy(output.wdl_logits, blended_wdl, reduction='none')
        wdl_loss = (wdl_rows * sample_weights).mean()
        auxiliary_losses = tuple(
            self._masked_auxiliary_loss(logits, target, eligibility, sample_weights)
            for logits, target, eligibility in zip(
                output.auxiliary_logits,
                batch.auxiliary_targets,
                batch.auxiliary_eligibility,
            )
        )
        total = self.policy_loss_weight * policy_loss + self.value_loss_weight * wdl_loss
        for weight, loss in zip(self.auxiliary_loss_weights, auxiliary_losses):
            total = total + weight * loss
        return ObjectiveLoss(policy=policy_loss, wdl=wdl_loss, auxiliary=auxiliary_losses, total=total)

    @staticmethod
    def _masked_auxiliary_loss(
        logits: torch.Tensor,
        target: torch.Tensor,
        eligibility: torch.Tensor,
        sample_weights: torch.Tensor,
    ) -> torch.Tensor:
        rows = functional.cross_entropy(logits, target, reduction='none')
        eligible_weights = eligibility.to(dtype=rows.dtype) * sample_weights
        denominator = eligible_weights.sum().clamp_min(1.0)
        return (rows * eligible_weights).sum() / denominator
