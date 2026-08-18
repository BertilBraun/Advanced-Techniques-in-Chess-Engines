from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal, TypeAlias

import torch
import torch.nn.functional as functional
from pydantic import Field

from src.training.batch import TrainingBatch, TrainingModelOutput
from src.training.targets import (
    AuxiliaryTargetConfiguration,
    FutureSearchValueTargetConfiguration,
    IrreversibleProgressTargetConfiguration,
    LegalMovesTargetConfiguration,
    NextPolicyTargetConfiguration,
    RemainingGameLengthTargetConfiguration,
    SearchCorrectionTargetConfiguration,
)
from src.util.frozen_model import FrozenModel


def _scalar_to_wdl(scores: torch.Tensor) -> torch.Tensor:
    """Represent scores in [-1, 1] with residual probability shared across WDL."""
    wins = torch.clamp(scores, min=0.0)
    losses = torch.clamp(-scores, min=0.0)
    remainders = 1.0 - torch.abs(scores)
    return torch.stack((wins + remainders / 3, remainders / 3, losses + remainders / 3), dim=-1)


def blended_wdl_targets(
    terminal_targets: torch.Tensor,
    root_values: torch.Tensor,
    root_value_blend: float,
) -> torch.Tensor:
    return torch.lerp(terminal_targets, _scalar_to_wdl(root_values), root_value_blend)


def mask_policy_logits(logits: torch.Tensor, legal_action_ids: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or legal_action_ids.ndim != 2 or logits.shape[0] != legal_action_ids.shape[0]:
        raise ValueError('Policy logits and legal action IDs must be aligned rank-two tensors.')
    if bool(torch.any(legal_action_ids < -1)) or bool(torch.any(legal_action_ids >= logits.shape[1])):
        raise ValueError('Legal action IDs must be valid policy indices or -1 padding.')
    legal_entries = legal_action_ids >= 0
    if not bool(torch.all(legal_entries.any(dim=1))):
        raise ValueError('Every policy row requires at least one legal action.')
    legal_counts = torch.zeros_like(logits, dtype=torch.int32)
    legal_counts.scatter_add_(1, legal_action_ids.clamp_min(0), legal_entries.to(dtype=torch.int32))
    legal_mask = legal_counts > 0
    return logits.masked_fill(~legal_mask, torch.finfo(logits.dtype).min)


@dataclass(frozen=True)
class ObjectiveLoss:
    policy: torch.Tensor
    wdl: torch.Tensor
    auxiliary: tuple[torch.Tensor, ...]
    total: torch.Tensor


class ResolvedNextPolicyLoss(FrozenModel):
    kind: Literal['next_policy'] = 'next_policy'
    weight: float = Field(ge=0.0)


class ResolvedRemainingGameLengthLoss(FrozenModel):
    kind: Literal['remaining_game_length'] = 'remaining_game_length'
    weight: float = Field(ge=0.0)


class ResolvedFutureSearchValueLoss(FrozenModel):
    kind: Literal['future_search_value'] = 'future_search_value'
    weight: float = Field(ge=0.0)
    smooth_l1_beta: float = Field(gt=0.0)


class ResolvedIrreversibleProgressLoss(FrozenModel):
    kind: Literal['irreversible_progress'] = 'irreversible_progress'
    weight: float = Field(ge=0.0)


class ResolvedLegalMovesLoss(FrozenModel):
    kind: Literal['legal_moves'] = 'legal_moves'
    weight: float = Field(ge=0.0)


class ResolvedSearchCorrectionLoss(FrozenModel):
    kind: Literal['search_correction'] = 'search_correction'
    weight: float = Field(ge=0.0)


ResolvedAuxiliaryLoss: TypeAlias = Annotated[
    ResolvedNextPolicyLoss
    | ResolvedRemainingGameLengthLoss
    | ResolvedFutureSearchValueLoss
    | ResolvedIrreversibleProgressLoss
    | ResolvedLegalMovesLoss
    | ResolvedSearchCorrectionLoss,
    Field(discriminator='kind'),
]


def resolve_auxiliary_losses(
    targets: tuple[AuxiliaryTargetConfiguration, ...],
    model_generation: int,
) -> tuple[ResolvedAuxiliaryLoss, ...]:
    losses: list[ResolvedAuxiliaryLoss] = []
    for target in targets:
        match target:
            case NextPolicyTargetConfiguration(loss_weight=loss_weight):
                losses.append(ResolvedNextPolicyLoss(weight=loss_weight.value_at(model_generation)))
            case RemainingGameLengthTargetConfiguration(loss_weight=loss_weight):
                losses.append(ResolvedRemainingGameLengthLoss(weight=loss_weight.value_at(model_generation)))
            case FutureSearchValueTargetConfiguration(
                loss_weight=loss_weight,
                smooth_l1_beta=smooth_l1_beta,
            ):
                losses.append(
                    ResolvedFutureSearchValueLoss(
                        weight=loss_weight.value_at(model_generation),
                        smooth_l1_beta=smooth_l1_beta,
                    )
                )
            case IrreversibleProgressTargetConfiguration(loss_weight=loss_weight):
                losses.append(ResolvedIrreversibleProgressLoss(weight=loss_weight.value_at(model_generation)))
            case LegalMovesTargetConfiguration(loss_weight=loss_weight):
                losses.append(ResolvedLegalMovesLoss(weight=loss_weight.value_at(model_generation)))
            case SearchCorrectionTargetConfiguration(loss_weight=loss_weight):
                losses.append(ResolvedSearchCorrectionLoss(weight=loss_weight.value_at(model_generation)))
    return tuple(losses)


class ResolvedTrainingObjective(FrozenModel):
    policy_loss_weight: float
    value_loss_weight: float
    root_value_blend: float
    auxiliary_losses: tuple[ResolvedAuxiliaryLoss, ...]

    def calculate_loss(self, output: TrainingModelOutput, batch: TrainingBatch) -> ObjectiveLoss:
        auxiliary_count = len(self.auxiliary_losses)
        if not (
            len(output.auxiliary_logits)
            == len(batch.auxiliary_targets)
            == len(batch.auxiliary_eligibility)
            == auxiliary_count
        ):
            raise ValueError('Model outputs, batch targets, eligibility masks, and objective layout must agree.')
        sample_weights = batch.sample_weights / batch.sample_weights.mean()
        policy_rows = functional.cross_entropy(
            mask_policy_logits(output.policy_logits, batch.policy_legal_action_ids),
            batch.policy_targets,
            reduction='none',
        )
        policy_loss = (policy_rows * sample_weights).mean()
        blended_wdl = blended_wdl_targets(batch.wdl_targets, batch.root_values, self.root_value_blend)
        wdl_rows = functional.cross_entropy(output.wdl_logits, blended_wdl, reduction='none')
        wdl_loss = (wdl_rows * sample_weights).mean()
        auxiliary_losses = tuple(
            self._masked_auxiliary_loss(configuration, prediction, target, legal_actions, eligibility, sample_weights)
            for configuration, prediction, target, legal_actions, eligibility in zip(
                self.auxiliary_losses,
                output.auxiliary_logits,
                batch.auxiliary_targets,
                batch.auxiliary_legal_action_ids,
                batch.auxiliary_eligibility,
            )
        )
        total = self.policy_loss_weight * policy_loss + self.value_loss_weight * wdl_loss
        for configuration, loss in zip(self.auxiliary_losses, auxiliary_losses):
            total = total + configuration.weight * loss
        return ObjectiveLoss(policy=policy_loss, wdl=wdl_loss, auxiliary=auxiliary_losses, total=total)

    @staticmethod
    def _masked_auxiliary_loss(
        configuration: ResolvedAuxiliaryLoss,
        prediction: torch.Tensor,
        target: torch.Tensor,
        legal_action_ids: torch.Tensor,
        eligibility: torch.Tensor,
        sample_weights: torch.Tensor,
    ) -> torch.Tensor:
        match configuration:
            case ResolvedNextPolicyLoss():
                eligible_rows = eligibility.to(dtype=torch.bool)
                rows = torch.zeros(prediction.shape[0], dtype=prediction.dtype, device=prediction.device)
                rows[eligible_rows] = functional.cross_entropy(
                    mask_policy_logits(prediction[eligible_rows], legal_action_ids[eligible_rows]),
                    target[eligible_rows],
                    reduction='none',
                )
            case ResolvedRemainingGameLengthLoss():
                rows = functional.smooth_l1_loss(prediction, target, reduction='none').squeeze(1)
            case ResolvedFutureSearchValueLoss(smooth_l1_beta=beta):
                rows = functional.smooth_l1_loss(prediction, target, reduction='none', beta=beta).squeeze(1)
            case ResolvedIrreversibleProgressLoss():
                rows = functional.smooth_l1_loss(
                    torch.sigmoid(prediction),
                    target,
                    reduction='none',
                ).squeeze(1)
            case ResolvedSearchCorrectionLoss():
                rows = functional.smooth_l1_loss(
                    torch.sigmoid(prediction),
                    target,
                    reduction='none',
                ).squeeze(1)
            case ResolvedLegalMovesLoss():
                element_loss = functional.binary_cross_entropy_with_logits(
                    prediction,
                    target,
                    reduction='none',
                )
                legal = target > 0.5
                illegal = ~legal
                positive = (element_loss * legal).sum(dim=1) / legal.sum(dim=1).clamp_min(1)
                negative = (element_loss * illegal).sum(dim=1) / illegal.sum(dim=1).clamp_min(1)
                rows = 0.5 * (positive + negative)
        eligible_weights = eligibility.to(dtype=rows.dtype) * sample_weights
        denominator = eligible_weights.sum().clamp_min(1.0)
        return (rows * eligible_weights).sum() / denominator
