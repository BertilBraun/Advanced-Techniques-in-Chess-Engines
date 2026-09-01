from __future__ import annotations

import pytest
import torch
from src.training.objective import ObjectiveLoss, ResolvedNextPolicyLoss, ResolvedTrainingObjective
from src.training.trainer.rank import _term_trunk_gradients


def _objective(policy_weight: float, value_weight: float, auxiliary_weight: float) -> ResolvedTrainingObjective:
    return ResolvedTrainingObjective(
        policy_loss_weight=policy_weight,
        value_loss_weight=value_weight,
        root_value_blend=0.0,
        auxiliary_losses=(ResolvedNextPolicyLoss(weight=auxiliary_weight),),
    )


def _loss(features: torch.Tensor) -> ObjectiveLoss:
    policy = (features * 2.0).sum()
    wdl = (features * 1.0).sum()
    auxiliary = (features * 4.0).sum()
    return ObjectiveLoss(policy=policy, wdl=wdl, auxiliary=(auxiliary,), total=policy + wdl + auxiliary)


def test_trunk_gradients_scale_with_the_loss_weight() -> None:
    features = torch.ones((4, 8), requires_grad=True)
    gradients = _term_trunk_gradients(_objective(1.0, 0.5, 0.25), _loss(features), features)

    root = torch.tensor(32.0).sqrt()
    assert gradients.tolist() == pytest.approx(
        [float(2.0 * root), float(0.5 * root), float(0.25 * 4.0 * root)], rel=1e-5
    )


def test_a_zero_weight_term_contributes_no_trunk_gradient() -> None:
    features = torch.ones((4, 8), requires_grad=True)
    gradients = _term_trunk_gradients(_objective(1.0, 1.0, 0.0), _loss(features), features)

    assert gradients[2].item() == pytest.approx(0.0)


def test_trunk_gradients_report_one_entry_per_loss_term() -> None:
    features = torch.ones((2, 3), requires_grad=True)
    gradients = _term_trunk_gradients(_objective(1.0, 1.0, 0.1), _loss(features), features)

    assert gradients.shape == (3,)


def test_the_probe_leaves_the_graph_usable_for_the_real_backward() -> None:
    features = torch.ones((4, 8), requires_grad=True)
    loss = _loss(features)

    _term_trunk_gradients(_objective(1.0, 1.0, 1.0), loss, features)
    loss.total.backward()

    assert features.grad is not None
    assert features.grad.abs().sum().item() > 0.0
