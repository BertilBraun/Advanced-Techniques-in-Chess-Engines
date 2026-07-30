from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from src.az.games.go.configuration import GoObjectiveConfiguration
from src.az.games.go.model import GoModelOutput
from src.az.games.go.samples import GoBatch


@dataclass(frozen=True)
class WeightedLossStatistics:
    weighted_sum: Tensor
    mean: Tensor
    eligible_count: int
    weight_sum: float


@dataclass(frozen=True)
class GoLossResult:
    total: Tensor
    policy: WeightedLossStatistics
    value: WeightedLossStatistics
    l2_regularization: Tensor


def _weighted_statistics(losses: Tensor, weights: Tensor) -> WeightedLossStatistics:
    weighted_sum = torch.sum(losses * weights)
    weight_sum_tensor = torch.sum(weights)
    mean = torch.where(weight_sum_tensor > 0, weighted_sum / weight_sum_tensor.clamp_min(1), weighted_sum * 0)
    return WeightedLossStatistics(
        weighted_sum=weighted_sum,
        mean=mean,
        eligible_count=int(torch.count_nonzero(weights).item()),
        weight_sum=float(weight_sum_tensor.detach().item()),
    )


def _validate_batch(batch: GoBatch) -> None:
    if batch.policy_targets.ndim != 2:
        raise ValueError('Policy targets must have shape B x actions.')
    batch_size, action_count = batch.policy_targets.shape
    if batch.inputs.ndim != 4 or batch.inputs.shape[0] != batch_size:
        raise ValueError('Go batch inputs must have shape B x planes x N x N.')
    if batch.legal_action_masks.shape != (batch_size, action_count):
        raise ValueError('Policy targets and legal-action masks must have shape B x actions.')
    if batch.legal_action_masks.dtype != torch.bool or torch.any(~torch.any(batch.legal_action_masks, dim=1)):
        raise ValueError('Every sample must have a boolean mask with at least one legal action.')
    vectors = (batch.policy_weights, batch.value_weights, batch.value_targets)
    if any(vector.shape != (batch_size,) for vector in vectors):
        raise ValueError('Value targets and loss weights must have shape B.')
    if (
        not torch.all(torch.isfinite(batch.policy_targets))
        or torch.any(batch.policy_targets < 0)
        or not torch.all(torch.isfinite(batch.value_targets))
        or not torch.all(torch.isfinite(batch.policy_weights))
        or not torch.all(torch.isfinite(batch.value_weights))
        or torch.any(batch.policy_weights < 0)
        or torch.any(batch.value_weights < 0)
        or torch.any(batch.value_targets < -1)
        or torch.any(batch.value_targets > 1)
    ):
        raise ValueError('Go targets and weights must be finite and nonnegative where applicable.')
    target_mass = torch.sum(batch.policy_targets, dim=1)
    if torch.any((target_mass > 0) & ~torch.isclose(target_mass, torch.ones_like(target_mass))):
        raise ValueError('Nonempty policy targets must be normalized.')
    if torch.any((batch.policy_weights > 0) & (target_mass == 0)):
        raise ValueError('Positive policy weight requires a nonempty policy target.')


def calculate_go_loss(
    output: GoModelOutput,
    batch: GoBatch,
    model: nn.Module,
    configuration: GoObjectiveConfiguration,
) -> GoLossResult:
    """Calculate weighted losses; explicit L2 covers every trainable model parameter."""
    _validate_batch(batch)
    if output.policy_logits.shape != batch.policy_targets.shape:
        raise ValueError('Policy output and target shapes must match.')
    if output.value.shape != batch.value_targets.shape:
        raise ValueError('Value output and target shapes must match.')
    if (
        not torch.all(torch.isfinite(output.policy_logits))
        or not torch.all(torch.isfinite(output.value))
        or torch.any(output.value < -1)
        or torch.any(output.value > 1)
    ):
        raise ValueError('Go model outputs must be finite and values must be in [-1, 1].')
    if torch.any(batch.policy_targets[~batch.legal_action_masks] != 0):
        raise ValueError('Policy targets cannot assign mass to illegal actions.')
    masked_logits = output.policy_logits.masked_fill(~batch.legal_action_masks, -torch.inf)
    policy_log_probabilities = torch.log_softmax(masked_logits, dim=1)
    policy_terms = torch.where(
        batch.policy_targets > 0,
        batch.policy_targets * policy_log_probabilities,
        torch.zeros_like(batch.policy_targets),
    )
    policy_losses = -torch.sum(policy_terms, dim=1)
    value_losses = torch.square(output.value - batch.value_targets)
    policy = _weighted_statistics(policy_losses, batch.policy_weights)
    value = _weighted_statistics(value_losses, batch.value_weights)
    if configuration.l2_regularization_weight > 0:
        parameter_squares = (torch.sum(torch.square(parameter)) for parameter in model.parameters())
        l2_regularization = configuration.l2_regularization_weight * sum(parameter_squares)
    else:
        l2_regularization = output.policy_logits.sum() * 0
    total = (
        configuration.policy_loss_weight * policy.mean
        + configuration.value_loss_weight * value.mean
        + l2_regularization
    )
    return GoLossResult(
        total=total,
        policy=policy,
        value=value,
        l2_regularization=l2_regularization,
    )
