from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.Network import Network
from src.self_play.SelfPlayDataset import TrainingBatch
from src.self_play.value_target import (
    FinalOutcome,
    ReplayValueTarget,
    TerminationReason,
    outcome_from_sample_perspective,
)
from src.settings import TRAINING_ARGS
from src.train.Trainer import Trainer
from src.train.TrainingArgs import ModelVersionLearningRate, ModelVersionLearningRateStage, TrainingParams
from src.value import scalar_to_wdl


class FixedValueNetwork(Network):
    def __init__(self, value_logits: tuple[float, float, float]) -> None:
        nn.Module.__init__(self)
        self.device = torch.device('cpu')
        self.policy_logits = nn.Parameter(torch.zeros(2))
        self.value_logits = nn.Parameter(torch.tensor(value_logits))

    def logit_forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = states.shape[0]
        return (
            self.policy_logits.unsqueeze(0).expand(batch_size, -1),
            self.value_logits.unsqueeze(0).expand(batch_size, -1),
        )


class FixedBatchLoader:
    def __init__(self, batch: TrainingBatch, repetitions: int = 1) -> None:
        self.batch = batch
        self.repetitions = repetitions

    def __iter__(self) -> Iterator[TrainingBatch]:
        for _ in range(self.repetitions):
            yield self.batch

    def __len__(self) -> int:
        return self.repetitions


def training_parameters(
    mcts_value_target_warmup_optimizer_steps: int = 0,
    duplicate_multiplicity_weight_cap: float | None = 4.0,
) -> TrainingParams:
    return TrainingParams(
        global_batch_size=4,
        local_batch_size=4,
        optimizer='adamw',
        learning_rate=ModelVersionLearningRate(
            stages=(ModelVersionLearningRateStage(0, 0.001),),
            optimizer_steps_per_model_version=1,
        ),
        credit_training=TRAINING_ARGS.training.credit_training,
        outcome_value_loss_weight=0.85,
        mcts_value_loss_weight=0.15,
        mcts_value_target_warmup_optimizer_steps=mcts_value_target_warmup_optimizer_steps,
        duplicate_multiplicity_weight_cap=duplicate_multiplicity_weight_cap,
    )


def training_batch(
    outcomes: tuple[FinalOutcome, ...],
    mcts_values: tuple[float, ...],
    eligibility: tuple[bool, ...],
    reasons: tuple[TerminationReason, ...],
    material_scores: tuple[float, ...] | None = None,
    material_eligibility: tuple[bool, ...] | None = None,
) -> TrainingBatch:
    sample_count = len(outcomes)
    resolved_material_scores = material_scores or (0.0,) * sample_count
    resolved_material_eligibility = material_eligibility or (False,) * sample_count
    return TrainingBatch(
        states=torch.zeros((sample_count, 1)),
        policy_targets=torch.full((sample_count, 2), 0.5),
        final_outcomes=torch.tensor(tuple(int(outcome) for outcome in outcomes)),
        mcts_root_values=torch.tensor(mcts_values),
        outcome_target_eligible=torch.tensor(eligibility),
        material_result_scores=torch.tensor(resolved_material_scores),
        material_target_eligible=torch.tensor(resolved_material_eligibility),
        termination_reasons=torch.tensor(tuple(int(reason) for reason in reasons)),
        plies=torch.arange(sample_count, dtype=torch.int32),
        current_player_piece_counts=torch.full((sample_count,), 8, dtype=torch.int8),
        opponent_piece_counts=torch.full((sample_count,), 8, dtype=torch.int8),
        occurrence_counts=torch.ones(sample_count, dtype=torch.int32),
        sample_weights=torch.ones(sample_count, dtype=torch.float32),
    )


def trainer(value_logits: tuple[float, float, float]) -> Trainer:
    model = FixedValueNetwork(value_logits)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    return Trainer(model, optimizer, training_parameters())


def test_wdl_order_is_win_draw_loss_in_python_and_torchscript() -> None:
    assert tuple(int(outcome) for outcome in FinalOutcome) == (0, 1, 2)
    probabilities = torch.tensor(((0.7, 0.2, 0.1),))
    scripted_expected_score = torch.jit.trace(
        lambda values: values[:, int(FinalOutcome.WIN)] - values[:, int(FinalOutcome.LOSS)],
        probabilities,
    )

    torch.testing.assert_close(scripted_expected_score(probabilities), torch.tensor((0.6,)))


def test_final_outcome_and_mcts_target_are_independent() -> None:
    win_target = ReplayValueTarget.from_scores(1.0, 0.25, TerminationReason.NATURAL)
    loss_target = ReplayValueTarget.from_scores(-1.0, 0.25, TerminationReason.NATURAL)

    assert win_target.mcts_root_value == loss_target.mcts_root_value
    assert win_target.final_outcome is FinalOutcome.WIN
    assert loss_target.final_outcome is FinalOutcome.LOSS


def test_real_resignation_is_an_eligible_hard_loss() -> None:
    target = ReplayValueTarget.from_scores(-1.0, -0.93, TerminationReason.RESIGNATION)

    assert target.final_outcome is FinalOutcome.LOSS
    assert target.outcome_target_eligible


def test_material_adjudication_uses_continuous_score_without_hard_wdl() -> None:
    target = ReplayValueTarget.from_scores(0.2, -0.1, TerminationReason.MATERIAL_ADJUDICATION)

    assert target.final_outcome is FinalOutcome.DRAW
    assert not target.outcome_target_eligible
    assert target.material_result_score == pytest.approx(0.2)
    assert target.material_target_eligible


def test_outcome_perspective_alternates_without_distance_discount() -> None:
    assert outcome_from_sample_perspective(1.0, final_current_player=1, sample_current_player=1) == 1.0
    assert outcome_from_sample_perspective(1.0, final_current_player=1, sample_current_player=-1) == -1.0
    assert outcome_from_sample_perspective(-1.0, final_current_player=-1, sample_current_player=-1) == -1.0


def test_equal_expected_scalar_can_represent_different_wdl_distributions() -> None:
    first = torch.tensor((0.6, 0.2, 0.2))
    second = torch.tensor((0.4, 0.6, 0.0))

    assert first[0] - first[2] == pytest.approx(float(second[0] - second[2]))
    assert not torch.equal(first, second)


def test_scalar_to_wdl_places_positive_and_negative_mass_on_decisive_result() -> None:
    targets = scalar_to_wdl(torch.tensor((-1.0, -0.25, 0.0, 0.4, 1.0)))

    torch.testing.assert_close(
        targets,
        torch.tensor(
            (
                (0.0, 0.0, 1.0),
                (0.0, 0.75, 0.25),
                (0.0, 1.0, 0.0),
                (0.4, 0.6, 0.0),
                (1.0, 0.0, 0.0),
            )
        ),
    )


def test_unlabelled_ply_cap_has_no_value_objective_but_keeps_mcts_diagnostic() -> None:
    batch = training_batch(
        (FinalOutcome.WIN,),
        (1.0,),
        (False,),
        (TerminationReason.PLY_CAP,),
    )

    training_trainer = trainer((0.0, 0.0, 0.0))
    result = training_trainer._calculate_loss_for_batch(batch)
    stats = training_trainer._train_epoch(FixedBatchLoader(batch))

    assert torch.isfinite(result.total_loss)
    assert result.value_loss.item() == pytest.approx(0.0)
    assert stats.value_metrics.outcome_target_count == 0
    assert stats.value_metrics.mcts_huber > 0.0


def test_value_objective_blends_base_scalar_with_mcts_before_soft_wdl_conversion() -> None:
    batch = training_batch(
        (FinalOutcome.WIN, FinalOutcome.DRAW),
        (0.2, -0.4),
        (True, True),
        (TerminationReason.NATURAL, TerminationReason.RESIGNATION),
    )

    value_logits = (0.3, -0.2, 0.1)
    result = trainer(value_logits)._calculate_loss_for_batch(batch, mcts_value_target_weight=0.15)

    expected_scores = torch.tensor((0.88, -0.06))
    expected_targets = scalar_to_wdl(expected_scores)
    expected_loss = F.cross_entropy(
        torch.tensor((value_logits, value_logits)),
        expected_targets,
    )
    torch.testing.assert_close(result.target_expected_scores, expected_scores)
    assert result.value_loss.item() == pytest.approx(expected_loss.item())


def test_material_adjudication_uses_material_score_as_soft_wdl_target() -> None:
    batch = training_batch(
        (FinalOutcome.DRAW,),
        (0.1,),
        (False,),
        (TerminationReason.MATERIAL_ADJUDICATION,),
        material_scores=(0.4,),
        material_eligibility=(True,),
    )

    training_trainer = trainer((0.0, 0.0, 0.0))
    result = training_trainer._calculate_loss_for_batch(batch)
    stats = training_trainer._train_epoch(FixedBatchLoader(batch))

    torch.testing.assert_close(result.target_expected_scores, torch.tensor((0.4,)))
    assert result.value_loss.item() == pytest.approx(torch.log(torch.tensor(3.0)).item())
    assert stats.value_metrics.outcome_target_count == 0
    assert stats.value_metrics.material_huber == pytest.approx(0.5 * 0.4**2)


def test_mcts_huber_diagnostic_does_not_change_backward_gradient() -> None:
    first_batch = training_batch(
        (FinalOutcome.WIN,),
        (-1.0,),
        (True,),
        (TerminationReason.NATURAL,),
    )
    second_batch = replace(first_batch, mcts_root_values=torch.tensor((1.0,)))

    gradients: list[torch.Tensor] = []
    for batch in (first_batch, second_batch):
        training_trainer = trainer((0.3, -0.2, 0.1))
        result = training_trainer._calculate_loss_for_batch(batch, mcts_value_target_weight=0.0)
        result.total_loss.backward()
        gradients.append(training_trainer.model.value_logits.grad.detach().clone())

    torch.testing.assert_close(gradients[0], gradients[1])


def test_mcts_target_weight_warms_up_over_optimizer_steps() -> None:
    batch = training_batch(
        (FinalOutcome.WIN, FinalOutcome.DRAW),
        (0.2, -0.4),
        (True, True),
        (TerminationReason.NATURAL, TerminationReason.RESIGNATION),
    )
    model = FixedValueNetwork((0.0, 0.0, 0.0))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    warmup_trainer = Trainer(
        model,
        optimizer,
        training_parameters(mcts_value_target_warmup_optimizer_steps=100),
    )

    stats = warmup_trainer.train(FixedBatchLoader(batch), optimizer_step=50)

    assert stats.mcts_value_target_weight == pytest.approx(0.075)


def test_occurrence_count_weight_uses_uncapped_square_root() -> None:
    batch = training_batch(
        (FinalOutcome.WIN, FinalOutcome.LOSS),
        (0.0, 0.0),
        (True, True),
        (TerminationReason.NATURAL, TerminationReason.NATURAL),
    )
    batch = replace(
        batch,
        occurrence_counts=torch.tensor((1, 100), dtype=torch.int32),
        sample_weights=torch.tensor((1.0, 10.0)),
    )
    value_logits = (2.0, 0.0, -1.0)
    model = FixedValueNetwork(value_logits)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    uncapped_trainer = Trainer(
        model,
        optimizer,
        training_parameters(duplicate_multiplicity_weight_cap=None),
    )

    result = uncapped_trainer._calculate_loss_for_batch(batch)

    per_sample_losses = F.cross_entropy(
        torch.tensor((value_logits, value_logits)),
        torch.tensor((int(FinalOutcome.WIN), int(FinalOutcome.LOSS))),
        reduction='none',
    )
    normalized_weights = torch.tensor((1.0, 10.0)) / 5.5
    expected_loss = (per_sample_losses * normalized_weights).mean()
    assert result.value_loss.item() == pytest.approx(expected_loss.item())


def test_training_metrics_use_outcome_and_mcts_denominators_independently() -> None:
    batch = training_batch(
        (FinalOutcome.WIN, FinalOutcome.DRAW, FinalOutcome.LOSS),
        (0.2, 0.0, -0.2),
        (True, False, True),
        (
            TerminationReason.NATURAL,
            TerminationReason.PLY_CAP,
            TerminationReason.RESIGNATION,
        ),
    )
    training_trainer = trainer((0.0, 0.0, 0.0))

    stats = training_trainer._train_epoch(FixedBatchLoader(batch))

    assert stats.value_metrics.outcome_target_count == 2
    assert stats.value_metrics.mcts_target_count == 3
    assert stats.excluded_outcome_target_count == 1
    assert stats.termination_value_metrics[int(TerminationReason.PLY_CAP)].outcome_target_count == 0
    assert stats.termination_value_metrics[int(TerminationReason.PLY_CAP)].mcts_target_count == 1
    assert stats.value_metrics.outcome_cross_entropy == pytest.approx(torch.log(torch.tensor(3.0)).item())
    assert stats.value_metrics.brier_score == pytest.approx(2.0 / 3.0)
    assert sum(stats.value_metrics.expected_score_bin_counts) == 2


def test_diagnostic_rows_are_excluded_from_both_value_objectives() -> None:
    batch = training_batch(
        (FinalOutcome.WIN,),
        (0.0,),
        (False,),
        (TerminationReason.DIAGNOSTIC,),
    )
    training_trainer = trainer((0.0, 0.0, 0.0))

    loss = training_trainer._calculate_loss_for_batch(batch)
    stats = training_trainer._train_epoch(FixedBatchLoader(batch))

    assert loss.value_loss.item() == 0.0
    assert stats.value_metrics.outcome_target_count == 0
    assert stats.value_metrics.mcts_target_count == 0


def test_expected_score_calibration_compares_binned_predictions_to_outcomes() -> None:
    batch = training_batch(
        (FinalOutcome.WIN,),
        (0.0,),
        (True,),
        (TerminationReason.NATURAL,),
    )

    stats = trainer((0.0, 0.0, 0.0))._train_epoch(FixedBatchLoader(batch))

    assert stats.value_metrics.expected_score_calibration_error == pytest.approx(1.0)


def test_value_metrics_are_sliced_by_ply_and_total_material() -> None:
    batch = training_batch(
        (FinalOutcome.WIN,) * 5,
        (0.0,) * 5,
        (True,) * 5,
        (TerminationReason.NATURAL,) * 5,
    )
    batch = replace(
        batch,
        plies=torch.tensor((0, 50, 100, 150, 200), dtype=torch.int32),
        current_player_piece_counts=torch.tensor((1, 6, 10, 14, 16), dtype=torch.int8),
        opponent_piece_counts=torch.tensor((1, 6, 10, 14, 16), dtype=torch.int8),
    )

    stats = trainer((0.0, 0.0, 0.0))._train_epoch(FixedBatchLoader(batch))

    assert tuple(metrics.outcome_target_count for metrics in stats.ply_value_metrics) == (1, 1, 1, 1, 1)
    assert tuple(metrics.outcome_target_count for metrics in stats.material_value_metrics) == (1, 1, 1, 2)


def test_sliced_value_metrics_sample_every_tenth_batch() -> None:
    batch = training_batch(
        (FinalOutcome.WIN,),
        (0.0,),
        (True,),
        (TerminationReason.NATURAL,),
    )

    stats = trainer((0.0, 0.0, 0.0))._train_epoch(FixedBatchLoader(batch, repetitions=11))

    assert stats.value_metrics.outcome_target_count == 11
    assert stats.ply_value_metrics[0].outcome_target_count == 2
    assert stats.material_value_metrics[1].outcome_target_count == 2
