from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from src.az.games.go.losses import calculate_go_loss
from src.az.games.go.model import GoModelOutput, ResidualGoModel
from src.az.games.go.samples import GoBatch, create_batch
from test.unit.go_stage5_helpers import (
    game_configuration,
    model_configuration,
    objective_configuration,
    sample,
)


@pytest.mark.parametrize('board_size', [7, 9])
def test_model_forward_shapes_and_bounded_value(board_size: int) -> None:
    game = game_configuration(board_size)
    model = ResidualGoModel(game, model_configuration())
    batch = create_batch((sample(board_size), sample(board_size)), game)

    output = model(batch.inputs)

    assert output.policy_logits.shape == (2, game.action_count)
    assert output.value.shape == (2,)
    assert torch.all(output.value >= -1)
    assert torch.all(output.value <= 1)


def test_model_initialization_is_deterministic_under_explicit_seed() -> None:
    game = game_configuration()
    torch.manual_seed(918)
    first = ResidualGoModel(game, model_configuration())
    torch.manual_seed(918)
    second = ResidualGoModel(game, model_configuration())

    assert all(torch.equal(left, right) for left, right in zip(first.parameters(), second.parameters(), strict=True))


def test_model_rejects_wrong_input_shape() -> None:
    model = ResidualGoModel(game_configuration(), model_configuration())

    with pytest.raises(ValueError, match='B x planes x N x N'):
        model(torch.zeros((1, 2, 7, 7)))


def fixture_batch() -> GoBatch:
    return GoBatch(
        inputs=torch.zeros((2, 1, 1, 1)),
        legal_action_masks=torch.tensor([[True, True, False], [True, True, True]]),
        policy_targets=torch.tensor([[0.25, 0.75, 0.0], [0.0, 1.0, 0.0]]),
        value_targets=torch.tensor([1.0, -1.0]),
        policy_weights=torch.tensor([2.0, 0.0]),
        value_weights=torch.tensor([1.0, 3.0]),
    )


def test_loss_matches_exact_weighted_fixture_math_and_gradients() -> None:
    policy_logits = torch.tensor([[0.0, math.log(3), 99.0], [0.0, 0.0, 0.0]], requires_grad=True)
    values = torch.tensor([0.5, 0.0], requires_grad=True)
    output = GoModelOutput(policy_logits=policy_logits, value=values)
    model = nn.Linear(1, 1)

    result = calculate_go_loss(output, fixture_batch(), model, objective_configuration())
    result.total.backward()

    expected_policy = -(0.25 * math.log(0.25) + 0.75 * math.log(0.75))
    expected_value = (0.25 + 3.0) / 4.0
    assert result.policy.mean.item() == pytest.approx(expected_policy)
    assert result.value.mean.item() == pytest.approx(expected_value)
    assert result.total.item() == pytest.approx(expected_policy + expected_value)
    assert result.policy.eligible_count == 1
    assert result.value.eligible_count == 2
    assert policy_logits.grad is not None
    assert values.grad is not None
    assert policy_logits.grad[0, 2].item() == 0


def test_zero_eligible_weights_produce_differentiable_finite_zero() -> None:
    batch = fixture_batch()
    zero_batch = GoBatch(
        inputs=batch.inputs,
        legal_action_masks=batch.legal_action_masks,
        policy_targets=torch.zeros_like(batch.policy_targets),
        value_targets=batch.value_targets,
        policy_weights=torch.zeros_like(batch.policy_weights),
        value_weights=torch.zeros_like(batch.value_weights),
    )
    logits = torch.zeros((2, 3), requires_grad=True)
    values = torch.zeros(2, requires_grad=True)

    result = calculate_go_loss(
        GoModelOutput(logits, values),
        zero_batch,
        nn.Linear(1, 1),
        objective_configuration(),
    )
    result.total.backward()

    assert result.total.item() == 0
    assert torch.all(torch.isfinite(logits.grad))
    assert torch.all(torch.isfinite(values.grad))


def test_explicit_l2_is_reported_and_part_of_total() -> None:
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(2)

    result = calculate_go_loss(
        GoModelOutput(torch.zeros((2, 3)), torch.zeros(2)),
        fixture_batch(),
        model,
        objective_configuration(l2_weight=0.5),
    )

    assert result.l2_regularization.item() == pytest.approx(2.0)
    assert result.total.item() == pytest.approx(result.policy.mean.item() + result.value.mean.item() + 2)


@pytest.mark.parametrize(
    ('policy_weights', 'policy_targets', 'message'),
    [
        (torch.tensor([-1.0, 0.0]), fixture_batch().policy_targets, 'finite and nonnegative'),
        (torch.tensor([1.0, 0.0]), torch.zeros((2, 3)), 'nonempty policy target'),
        (
            torch.tensor([1.0, 0.0]),
            torch.tensor([[0.2, 0.2, 0.0], [0.0, 0.0, 0.0]]),
            'must be normalized',
        ),
    ],
)
def test_loss_rejects_malformed_weights_and_targets(
    policy_weights: torch.Tensor,
    policy_targets: torch.Tensor,
    message: str,
) -> None:
    batch = fixture_batch()
    malformed = GoBatch(
        inputs=batch.inputs,
        legal_action_masks=batch.legal_action_masks,
        policy_targets=policy_targets,
        value_targets=batch.value_targets,
        policy_weights=policy_weights,
        value_weights=batch.value_weights,
    )

    with pytest.raises(ValueError, match=message):
        calculate_go_loss(
            GoModelOutput(torch.zeros((2, 3)), torch.zeros(2)),
            malformed,
            nn.Linear(1, 1),
            objective_configuration(),
        )


@pytest.mark.parametrize(
    ('policy_logits', 'values'),
    [
        (torch.tensor([[float('nan'), 0.0, 0.0], [0.0, 0.0, 0.0]]), torch.zeros(2)),
        (torch.zeros((2, 3)), torch.tensor([float('inf'), 0.0])),
        (torch.zeros((2, 3)), torch.tensor([1.01, 0.0])),
    ],
)
def test_loss_rejects_nonfinite_or_unbounded_model_outputs(
    policy_logits: torch.Tensor,
    values: torch.Tensor,
) -> None:
    with pytest.raises(ValueError, match='outputs must be finite'):
        calculate_go_loss(
            GoModelOutput(policy_logits, values),
            fixture_batch(),
            nn.Linear(1, 1),
            objective_configuration(),
        )


def test_loss_rejects_misaligned_batch_leading_dimension() -> None:
    batch = fixture_batch()
    malformed = GoBatch(
        inputs=torch.zeros((1, 1, 1, 1)),
        legal_action_masks=batch.legal_action_masks,
        policy_targets=batch.policy_targets,
        value_targets=batch.value_targets,
        policy_weights=batch.policy_weights,
        value_weights=batch.value_weights,
    )

    with pytest.raises(ValueError, match='B x planes'):
        calculate_go_loss(
            GoModelOutput(torch.zeros((2, 3)), torch.zeros(2)),
            malformed,
            nn.Linear(1, 1),
            objective_configuration(),
        )
