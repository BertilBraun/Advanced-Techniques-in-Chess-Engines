from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.Network import Network
from src.self_play.SelfPlayDataset import SelfPlayDataset, TrainingBatch
from src.self_play.SelfPlayCpp import SelfPlayCpp, SelfPlayGame, SelfPlayGameMemory
from src.self_play.value_target import (
    FinalOutcome,
    ReplayValueTarget,
    TerminationReason,
    outcome_from_sample_perspective,
)
from src.train.Trainer import Trainer
from src.train.TrainingArgs import TrainingParams


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
    def __init__(self, batch: TrainingBatch) -> None:
        self.batch = batch

    def __iter__(self) -> Iterator[TrainingBatch]:
        yield self.batch

    def __len__(self) -> int:
        return 1


def training_parameters() -> TrainingParams:
    return TrainingParams(
        num_epochs=1,
        global_batch_size=4,
        local_batch_size=4,
        optimizer='adamw',
        sampling_window=lambda _: 1,
        learning_rate=lambda _iteration, _optimizer: 0.0,
        learning_rate_scheduler=lambda _progress, learning_rate: learning_rate,
        outcome_value_loss_weight=0.85,
        mcts_value_loss_weight=0.15,
    )


def training_batch(
    outcomes: tuple[FinalOutcome, ...],
    mcts_values: tuple[float, ...],
    eligibility: tuple[bool, ...],
    reasons: tuple[TerminationReason, ...],
) -> TrainingBatch:
    sample_count = len(outcomes)
    return TrainingBatch(
        states=torch.zeros((sample_count, 1)),
        policy_targets=torch.full((sample_count, 2), 0.5),
        final_outcomes=torch.tensor(tuple(int(outcome) for outcome in outcomes)),
        mcts_root_values=torch.tensor(mcts_values),
        outcome_target_eligible=torch.tensor(eligibility),
        termination_reasons=torch.tensor(tuple(int(reason) for reason in reasons)),
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


def test_outcome_perspective_alternates_without_distance_discount() -> None:
    assert outcome_from_sample_perspective(1.0, final_current_player=1, sample_current_player=1) == 1.0
    assert outcome_from_sample_perspective(1.0, final_current_player=1, sample_current_player=-1) == -1.0
    assert outcome_from_sample_perspective(-1.0, final_current_player=-1, sample_current_player=-1) == -1.0


def test_self_play_stores_hard_outcome_and_independent_root_values_without_discount() -> None:
    self_play = object.__new__(SelfPlayCpp)
    self_play.args = SimpleNamespace(mcts=SimpleNamespace(min_visit_count=0))
    self_play.dataset = SelfPlayDataset()
    self_play.iteration = 0
    game = SelfPlayGame()
    game.acknowledge_model_version(0)
    game.memory = [
        SelfPlayGameMemory(game.board.copy(), [(0, 1)], 0.2),
        SelfPlayGameMemory(game.board.copy(), [(0, 1)], -0.4),
    ]

    self_play._add_training_data(game, 1.0, TerminationReason.NATURAL)

    assert {target.final_outcome for target in self_play.dataset.value_targets} == {FinalOutcome.WIN}
    assert {target.mcts_root_value for target in self_play.dataset.value_targets} == {0.2, -0.4}


def test_equal_expected_scalar_can_represent_different_wdl_distributions() -> None:
    first = torch.tensor((0.6, 0.2, 0.2))
    second = torch.tensor((0.4, 0.6, 0.0))

    assert first[0] - first[2] == pytest.approx(float(second[0] - second[2]))
    assert not torch.equal(first, second)


def test_all_ineligible_outcomes_have_finite_zero_ce_and_valid_mcts_loss() -> None:
    batch = training_batch(
        (FinalOutcome.WIN, FinalOutcome.LOSS),
        (1.0, -1.0),
        (False, False),
        (TerminationReason.PLY_CAP, TerminationReason.MATERIAL_ADJUDICATION),
    )

    result = trainer((0.0, 0.0, 0.0))._calculate_loss_for_batch(batch)

    assert torch.isfinite(result.total_loss)
    assert result.outcome_loss.item() == pytest.approx(0.0)
    assert result.mcts_auxiliary_loss.item() > 0.0
    assert result.combined_value_loss.item() == pytest.approx(0.15 * result.mcts_auxiliary_loss.item())


def test_value_objective_uses_configured_component_weights() -> None:
    batch = training_batch(
        (FinalOutcome.WIN, FinalOutcome.DRAW),
        (0.2, -0.4),
        (True, True),
        (TerminationReason.NATURAL, TerminationReason.RESIGNATION),
    )

    result = trainer((0.3, -0.2, 0.1))._calculate_loss_for_batch(batch)

    assert result.combined_value_loss.item() == pytest.approx(
        0.85 * result.outcome_loss.item() + 0.15 * result.mcts_auxiliary_loss.item()
    )


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

    assert loss.outcome_loss.item() == 0.0
    assert loss.mcts_auxiliary_loss.item() == 0.0
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
