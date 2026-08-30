from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest
import src.training.reporting as reporting_module

pytest.importorskip('AlphaZeroCpp')
from AlphaZeroCpp import GameSearchVisit
from src.games.representation import PackedPlaneLayout
from src.replay.description import ReplayDescription
from src.replay.layout import ReplayLayout
from src.replay.manager import IngestedCompletedGame
from src.self_play.completed_game import (
    SearchObservation,
    SearchStopReason,
    SearchVisitCounts,
    TerminationReason,
)
from src.training.configuration import CreditTrainingParams
from src.training.credit_ledger import CreditLedgerState
from src.training.reporting import TrainingReporter
from src.training.targets import TrainingTargetLayout
from src.training.telemetry import (
    completed_game_length_telemetry,
    search_budget_telemetry,
    training_lifecycle_telemetry,
)
from test_helpers.checkpoints import checkpoint_reference


def test_training_lifecycle_telemetry_reports_credit_backlog_and_observed_ratio() -> None:
    checkpoint = checkpoint_reference(generation=3)
    state = CreditLedgerState(
        completed_optimizer_steps=1500,
        earned_credits=Decimal(600),
        consumed_credits=Decimal(480),
        active_checkpoint=checkpoint,
    )
    credit = CreditTrainingParams(
        replay_ratio=Decimal(4),
        optimizer_steps_per_quantum=500,
        maximum_optimizer_steps=1_000_000,
        retained_checkpoint_interval_generations=1,
        self_play_backpressure_quanta=5,
    )
    replay = ReplayDescription(
        path=Path('replay.bin'),
        head=0,
        size=100,
        logical_capacity=250,
        maximum_capacity=1000,
        layout=ReplayLayout(
            packed_planes=PackedPlaneLayout(board_size=7, binary_plane_count=1, scalar_count=1),
            targets=TrainingTargetLayout(action_size=50, wdl_size=3, auxiliary_heads=()),
            maximum_policy_entries=50,
            maximum_legal_actions=50,
        ),
    )

    telemetry = training_lifecycle_telemetry(state, credit, replay, global_batch_size=2)

    assert telemetry.configured_replay_ratio == 4.0
    assert telemetry.observed_replay_ratio == 3.2
    assert telemetry.materialized_samples == 150
    assert telemetry.consumed_presentations == 480.0
    assert telemetry.available_presentations == 120.0
    assert telemetry.required_presentations_per_quantum == 1000
    assert telemetry.available_quantum_fraction == 0.12
    assert telemetry.live_replay_rows == 100
    assert telemetry.logical_replay_capacity == 250
    assert telemetry.replay_fill_fraction == 0.4


def test_completed_game_length_telemetry_reports_distribution_and_terminations() -> None:
    games = (
        IngestedCompletedGame(length_plies=10, termination_reason=TerminationReason.NATURAL),
        IngestedCompletedGame(length_plies=20, termination_reason=TerminationReason.RESIGNATION),
        IngestedCompletedGame(length_plies=30, termination_reason=TerminationReason.RESIGNATION),
        IngestedCompletedGame(length_plies=40, termination_reason=TerminationReason.MAXIMUM_PLIES),
    )

    telemetry = completed_game_length_telemetry(games)

    assert telemetry is not None
    assert telemetry.lengths_plies == (10, 20, 30, 40)
    assert telemetry.mean_plies == pytest.approx(25.0)
    assert telemetry.median_plies == pytest.approx(25.0)
    assert telemetry.p90_plies == pytest.approx(37.0)
    assert telemetry.p99_plies == pytest.approx(39.7)
    assert telemetry.maximum_plies == 40
    by_reason = {entry.reason: entry for entry in telemetry.terminations}
    assert by_reason[TerminationReason.NATURAL].completed_games == 1
    assert by_reason[TerminationReason.NATURAL].mean_plies == pytest.approx(10.0)
    assert by_reason[TerminationReason.RESIGNATION].fraction == pytest.approx(0.5)
    assert by_reason[TerminationReason.RESIGNATION].mean_plies == pytest.approx(25.0)
    assert by_reason[TerminationReason.ADJUDICATION].completed_games == 0
    assert by_reason[TerminationReason.ADJUDICATION].mean_plies is None


def test_completed_game_length_telemetry_omits_empty_windows() -> None:
    assert completed_game_length_telemetry(()) is None


def test_search_budget_telemetry_omits_windows_without_observations() -> None:
    games = (IngestedCompletedGame(length_plies=10, termination_reason=TerminationReason.NATURAL),)

    assert search_budget_telemetry(games) is None


def test_search_budget_telemetry_reports_prediction_allocation_and_residual() -> None:
    observation = SearchObservation(
        ply=0,
        model_generation=50,
        policy_target_visits=SearchVisitCounts.from_native((GameSearchVisit(action_id=2, visit_count=500),)),
        root_value=0.22,
        highest_visited_child_action_id=2,
        highest_visited_child_visit_count=500,
        highest_visited_child_q=0.2,
        selected_action_id=2,
        sample_weight=1.0,
        baseline_visits=800,
        network_root_value=0.1,
        policy_correction=0.3,
        value_correction=0.06,
        predicted_baseline_log_kl=0.85,
        selected_budget_index=5,
        assigned_additional_visits=700,
        parallel_searches=4,
        spend_residual=-2,
        starting_visits=100,
        final_visits=800,
        stop_reason=SearchStopReason.PREDICTED_BUDGET,
    )
    games = (
        IngestedCompletedGame(
            length_plies=1,
            termination_reason=TerminationReason.NATURAL,
            observations=(observation,),
        ),
    )

    telemetry = search_budget_telemetry(games)

    assert telemetry is not None
    assert telemetry.baseline_visits == (800,)
    assert telemetry.assigned_additional_visits == (700,)
    assert telemetry.predicted_baseline_log_kls == (0.85,)
    assert telemetry.selected_budget_indices == (5,)
    assert telemetry.parallel_searches == (4,)
    assert telemetry.spend_residuals == (-2,)
    assert telemetry.starting_visits == (100,)
    assert telemetry.final_visits == (800,)
    assert dict(telemetry.stop_reasons)[SearchStopReason.PREDICTED_BUDGET] == 1


def test_training_reporter_logs_completed_game_length_window(monkeypatch: pytest.MonkeyPatch) -> None:
    completed_games = (
        IngestedCompletedGame(length_plies=20, termination_reason=TerminationReason.NATURAL),
        IngestedCompletedGame(length_plies=40, termination_reason=TerminationReason.RESIGNATION),
    )
    scalars: dict[str, tuple[float, int | None]] = {}
    histograms: dict[str, tuple[np.ndarray, int | None]] = {}

    def record_scalar(name: str, value: float, step: int | None = None) -> None:
        scalars[name] = (value, step)

    def record_histogram(name: str, values: np.ndarray, step: int | None = None) -> None:
        histograms[name] = (values, step)

    monkeypatch.setattr(reporting_module, 'log_scalar', record_scalar)
    monkeypatch.setattr(reporting_module, 'log_histogram', record_histogram)

    TrainingReporter._record_completed_game_lengths(completed_games, generation=7)

    assert scalars['self_play/completed_games'] == (2, 7)
    assert scalars['self_play/game_length_plies_mean'] == (30.0, 7)
    assert scalars['self_play/game_length_plies_median'] == (30.0, 7)
    assert scalars['self_play/termination/natural/fraction'] == (0.5, 7)
    assert scalars['self_play/termination/resignation/game_length_plies_mean'] == (40.0, 7)
    assert np.array_equal(histograms['self_play/game_length_plies'][0], np.asarray((20, 40), dtype=np.int32))
    assert histograms['self_play/game_length_plies'][1] == 7


@pytest.mark.parametrize(
    ('available_credits', 'expected'),
    ((5_120_000, False), (5_120_001, True)),
)
def test_self_play_backpressure_starts_above_configured_quantum_surplus(
    available_credits: int,
    expected: bool,
) -> None:
    credit = CreditTrainingParams(
        replay_ratio=Decimal(8),
        optimizer_steps_per_quantum=500,
        maximum_optimizer_steps=1_000_000,
        retained_checkpoint_interval_generations=1,
        self_play_backpressure_quanta=5,
    )

    assert credit.requires_self_play_backpressure(Decimal(available_credits), global_batch_size=2048) is expected
