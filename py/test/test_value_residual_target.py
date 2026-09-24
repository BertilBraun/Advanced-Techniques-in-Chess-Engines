from __future__ import annotations

import pytest
import torch
from src.training.batch import TrainingBatch, TrainingModelOutput
from src.training.objective import (
    ResolvedTrainingObjective,
    ResolvedValueResidualLoss,
    resolve_auxiliary_losses,
)
from src.training.targets import (
    ValueResidualHeadLayout,
    ValueResidualTargetConfiguration,
    auxiliary_head_output_size,
    build_training_target_layout,
)

ACTION_SIZE = 6


def _configuration(weight: float = 0.05, beta: float = 0.05) -> ValueResidualTargetConfiguration:
    return ValueResidualTargetConfiguration.model_validate({'loss_weight': weight, 'smooth_l1_beta': beta})


def _batch(residual_targets: torch.Tensor) -> TrainingBatch:
    rows = residual_targets.shape[0]
    return TrainingBatch(
        states=torch.zeros(rows, 1),
        policy_targets=torch.full((rows, ACTION_SIZE), 1.0 / ACTION_SIZE),
        policy_legal_action_ids=torch.arange(ACTION_SIZE).repeat(rows, 1),
        wdl_targets=torch.full((rows, 3), 1.0 / 3.0),
        root_values=torch.zeros(rows),
        auxiliary_targets=(residual_targets,),
        auxiliary_legal_action_ids=(torch.zeros(rows, 1, dtype=torch.long),),
        auxiliary_eligibility=(torch.ones(rows, dtype=torch.bool),),
        sample_weights=torch.ones(rows),
        source_model_generations=torch.zeros(rows, dtype=torch.long),
        source_created_at_seconds=torch.zeros(rows),
    )


def _objective(weight: float) -> ResolvedTrainingObjective:
    return ResolvedTrainingObjective(
        policy_loss_weight=1.0,
        value_loss_weight=1.0,
        root_value_blend=0.0,
        auxiliary_losses=resolve_auxiliary_losses((_configuration(weight=weight),), model_generation=0),
    )


def _output(predictions: torch.Tensor) -> TrainingModelOutput:
    rows = predictions.shape[0]
    return TrainingModelOutput(
        policy_logits=torch.zeros(rows, ACTION_SIZE),
        wdl_logits=torch.zeros(rows, 3),
        auxiliary_logits=(predictions,),
        features=torch.zeros(rows, 1),
    )


def test_value_residual_head_is_a_single_scalar() -> None:
    layout = build_training_target_layout(ACTION_SIZE, (_configuration(),))

    assert layout.auxiliary_heads == (ValueResidualHeadLayout(kind='value_residual', smooth_l1_beta=0.05),)
    assert auxiliary_head_output_size(layout.auxiliary_heads[0]) == 1


def test_value_residual_resolves_its_weight_and_beta() -> None:
    (resolved,) = resolve_auxiliary_losses((_configuration(weight=0.04, beta=0.02),), model_generation=7)

    assert resolved == ResolvedValueResidualLoss(weight=0.04, smooth_l1_beta=0.02)


def test_value_residual_loss_vanishes_when_the_head_matches_the_target() -> None:
    targets = torch.tensor([[0.5], [0.5]])
    # The head is squashed, so the logit that reproduces a target of 0.5 is zero.
    loss = _objective(0.05).calculate_loss(_output(torch.zeros(2, 1)), _batch(targets))

    assert loss.auxiliary[0] == pytest.approx(0.0, abs=1e-6)


def test_value_residual_loss_grows_with_the_error() -> None:
    targets = torch.tensor([[0.9], [0.9]])
    small = _objective(0.05).calculate_loss(_output(torch.full((2, 1), 1.0)), _batch(targets))
    large = _objective(0.05).calculate_loss(_output(torch.full((2, 1), -3.0)), _batch(targets))

    assert large.auxiliary[0] > small.auxiliary[0] > 0.0


def test_value_residual_weight_scales_its_contribution_to_the_total() -> None:
    targets = torch.tensor([[0.9], [0.9]])
    predictions = torch.full((2, 1), -3.0)
    light = _objective(0.0).calculate_loss(_output(predictions), _batch(targets))
    heavy = _objective(0.5).calculate_loss(_output(predictions), _batch(targets))

    assert heavy.total > light.total
    assert heavy.total - light.total == pytest.approx(0.5 * heavy.auxiliary[0], abs=1e-6)
