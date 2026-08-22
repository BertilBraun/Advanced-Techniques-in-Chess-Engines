from __future__ import annotations

import torch
from src.games.representation import NetworkDimensions
from src.training.batch import TrainingBatch
from src.training.objective import ResolvedLegalMovesLoss, ResolvedNextPolicyLoss, ResolvedTrainingObjective
from tools.benchmark_training_overfit import (
    BenchmarkModel,
    ObjectiveProfile,
    achievable_loss_floor,
    benchmark_network,
    objective_for_profile,
)


def _batch() -> TrainingBatch:
    return TrainingBatch(
        states=torch.zeros((2, 1, 1, 1)),
        policy_targets=torch.tensor(((0.75, 0.25), (0.5, 0.5))),
        policy_legal_action_ids=torch.tensor(((0, 1), (0, 1))),
        wdl_targets=torch.tensor(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
        root_values=torch.zeros(2),
        auxiliary_targets=(torch.tensor(((0.25, 0.75), (0.5, 0.5))),),
        auxiliary_legal_action_ids=(torch.tensor(((0, 1), (0, 1))),),
        auxiliary_eligibility=(torch.ones(2, dtype=torch.bool),),
        sample_weights=torch.ones(2),
        source_model_generations=torch.zeros(2, dtype=torch.int64),
        source_created_at_seconds=torch.zeros(2, dtype=torch.float64),
    )


def _objective() -> ResolvedTrainingObjective:
    return ResolvedTrainingObjective(
        policy_loss_weight=1.0,
        value_loss_weight=1.0,
        root_value_blend=0.0,
        auxiliary_losses=(ResolvedNextPolicyLoss(weight=0.2),),
    )


def test_achievable_loss_floor_includes_soft_target_entropies() -> None:
    floor = achievable_loss_floor(_batch(), _objective())

    assert floor.policy > 0.0
    assert floor.wdl == 0.0
    assert floor.auxiliary[0] > 0.0
    assert floor.total == floor.policy + 0.2 * floor.auxiliary[0]


def test_objective_profiles_scale_only_auxiliary_weights() -> None:
    objective = _objective()

    half = objective_for_profile(objective, ObjectiveProfile.HALF_AUXILIARY)
    primary = objective_for_profile(objective, ObjectiveProfile.PRIMARY_ONLY)
    objective_with_legal_moves = objective.validated_copy(
        update={
            'auxiliary_losses': [
                ResolvedNextPolicyLoss(weight=0.2).model_dump(mode='json'),
                ResolvedLegalMovesLoss(weight=0.05).model_dump(mode='json'),
            ]
        }
    )
    without_legal_moves = objective_for_profile(
        objective_with_legal_moves,
        ObjectiveProfile.WITHOUT_LEGAL_MOVES,
    )

    assert half.auxiliary_losses[0].weight == 0.1
    assert primary.auxiliary_losses[0].weight == 0.0
    assert half.policy_loss_weight == primary.policy_loss_weight == 1.0
    assert half.value_loss_weight == primary.value_loss_weight == 1.0
    assert tuple(loss.weight for loss in without_legal_moves.auxiliary_losses) == (0.2, 0.0)


def test_benchmark_models_use_distinct_architectures() -> None:
    dimensions = NetworkDimensions(channels=29, rows=8, columns=8, actions=4864)

    attention = benchmark_network(BenchmarkModel.ATTENTION_1M)
    convolutional = benchmark_network(BenchmarkModel.CNN_1M)

    assert attention.kind == 'attention'
    assert convolutional.kind == 'convolutional'
    assert dimensions.actions == 4864
