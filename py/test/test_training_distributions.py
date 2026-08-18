import pytest
import torch

from src.training.batch import TrainingBatch, TrainingModelOutput
from src.training.distributions import RemainingGameLengthTrainingDistribution, capture_training_distributions
from src.training.objective import mask_policy_logits, ResolvedRemainingGameLengthLoss, ResolvedTrainingObjective


def test_training_distributions_capture_targets_predictions_losses_and_replay_age() -> None:
    batch = TrainingBatch(
        states=torch.zeros((2, 1)),
        policy_targets=torch.tensor(((0.6, 0.3, 0.1), (0.5, 0.5, 0.0))),
        policy_legal_action_ids=torch.tensor(((0, 1, 2), (0, 1, -1))),
        wdl_targets=torch.tensor(((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))),
        root_values=torch.tensor((0.25, -0.5)),
        auxiliary_targets=(torch.tensor(((0.75,), (0.25,))),),
        auxiliary_legal_action_ids=(torch.empty((2, 0), dtype=torch.int64),),
        auxiliary_eligibility=(torch.tensor((True, False)),),
        sample_weights=torch.tensor((1.0, 2.0)),
        source_model_generations=torch.tensor((7, 9)),
        source_created_at_seconds=torch.tensor((90.0, 95.0), dtype=torch.float64),
    )
    output = TrainingModelOutput(
        policy_logits=torch.log(torch.tensor(((0.5, 0.3, 0.2), (0.4, 0.4, 0.2)))),
        wdl_logits=torch.log(torch.tensor(((0.8, 0.1, 0.1), (0.1, 0.2, 0.7)))),
        auxiliary_logits=(torch.tensor(((0.5,), (100.0,))),),
    )
    objective = ResolvedTrainingObjective(
        policy_loss_weight=1.0,
        value_loss_weight=1.0,
        root_value_blend=0.0,
        auxiliary_losses=(ResolvedRemainingGameLengthLoss(weight=0.15),),
    )

    distributions = capture_training_distributions(
        output,
        batch,
        objective,
        source_generation=10,
        captured_at_seconds=100.0,
    )

    assert distributions.policy.target_top1_mass == pytest.approx((0.6, 0.5))
    assert distributions.policy.target_top2_mass == pytest.approx((0.9, 1.0))
    assert distributions.policy.target_top3_mass == pytest.approx((1.0, 1.0))
    assert distributions.terminal_value == pytest.approx((1.0, -1.0))
    assert distributions.predicted_value == pytest.approx((0.7, -0.6))
    assert distributions.replay_generation_age == pytest.approx((3.0, 1.0))
    assert distributions.replay_age_seconds == pytest.approx((10.0, 5.0))
    auxiliary = distributions.auxiliary[0]
    assert isinstance(auxiliary, RemainingGameLengthTrainingDistribution)
    assert auxiliary.target == pytest.approx((0.75,))
    assert auxiliary.prediction == pytest.approx((0.5,))
    assert auxiliary.absolute_error == pytest.approx((0.25,))


def test_training_distribution_capture_is_bounded() -> None:
    batch = TrainingBatch(
        states=torch.zeros((3, 1)),
        policy_targets=torch.tensor(((1.0, 0.0),) * 3),
        policy_legal_action_ids=torch.tensor(((0, 1),) * 3),
        wdl_targets=torch.tensor(((0.0, 1.0, 0.0),) * 3),
        root_values=torch.zeros(3),
        auxiliary_targets=(),
        auxiliary_legal_action_ids=(),
        auxiliary_eligibility=(),
        sample_weights=torch.ones(3),
        source_model_generations=torch.zeros(3, dtype=torch.int64),
        source_created_at_seconds=torch.zeros(3, dtype=torch.float64),
    )
    output = TrainingModelOutput(
        policy_logits=torch.zeros((3, 2)),
        wdl_logits=torch.zeros((3, 3)),
        auxiliary_logits=(),
    )
    objective = ResolvedTrainingObjective(
        policy_loss_weight=1.0,
        value_loss_weight=1.0,
        root_value_blend=0.0,
        auxiliary_losses=(),
    )

    distributions = capture_training_distributions(output, batch, objective, 0, 0.0, maximum_rows=2)

    assert len(distributions.policy.loss) == 2
    assert len(distributions.wdl_loss) == 2


def test_training_distribution_capture_promotes_mixed_precision_predictions() -> None:
    batch = TrainingBatch(
        states=torch.zeros((1, 1)),
        policy_targets=torch.tensor(((1.0, 0.0),)),
        policy_legal_action_ids=torch.tensor(((0, 1),)),
        wdl_targets=torch.tensor(((1.0, 0.0, 0.0),)),
        root_values=torch.zeros(1),
        auxiliary_targets=(),
        auxiliary_legal_action_ids=(),
        auxiliary_eligibility=(),
        sample_weights=torch.ones(1),
        source_model_generations=torch.zeros(1, dtype=torch.int64),
        source_created_at_seconds=torch.zeros(1, dtype=torch.float64),
    )
    output = TrainingModelOutput(
        policy_logits=torch.zeros((1, 2), dtype=torch.bfloat16),
        wdl_logits=torch.zeros((1, 3), dtype=torch.bfloat16),
        auxiliary_logits=(),
    )
    objective = ResolvedTrainingObjective(
        policy_loss_weight=1.0,
        value_loss_weight=1.0,
        root_value_blend=0.0,
        auxiliary_losses=(),
    )

    distributions = capture_training_distributions(output, batch, objective, 0, 0.0)

    assert distributions.predicted_value == pytest.approx((0.0,))
    assert distributions.policy.prediction_entropy == pytest.approx((0.693147,), rel=1e-5)


def test_masked_dense_policy_loss_remains_finite() -> None:
    logits = torch.tensor(((1.0, 2.0, 100.0, 100.0),), requires_grad=True)
    targets = torch.tensor(((0.5, 0.5, 0.0, 0.0),))
    masked = mask_policy_logits(logits, torch.tensor(((0, 1, -1, -1),)))

    loss = torch.nn.functional.cross_entropy(masked, targets)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(masked[0, 0])
    assert logits.grad[0, 2] == 0.0
    assert logits.grad[0, 3] == 0.0


@pytest.mark.parametrize('legal_action_ids', (((-2,),), ((4,),)))
def test_policy_mask_rejects_invalid_legal_action_ids(legal_action_ids: tuple[tuple[int, ...], ...]) -> None:
    with pytest.raises(ValueError, match='valid policy indices'):
        mask_policy_logits(torch.zeros((1, 4)), torch.tensor(legal_action_ids))
