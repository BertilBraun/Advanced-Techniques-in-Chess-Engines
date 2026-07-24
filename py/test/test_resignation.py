from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

from src.self_play.SelfPlayCpp import SelfPlayCpp, SelfPlayGame
from src.self_play.resignation import (
    CompletedResignationAudit,
    ExternalResignationSafetyApproval,
    ResignationManager,
    ResignationObservation,
    ResignationParams,
    ResignationTerminationReason,
    best_child_value_from_root_perspective,
    one_sided_binomial_upper_bound,
)


def observation(
    root_value: float = -0.95,
    child_value_from_root_perspective: float = -0.95,
    side_to_move: int = 1,
) -> ResignationObservation:
    return ResignationObservation(
        model_version=40,
        ply=80,
        side_to_move=side_to_move,
        root_value=root_value,
        best_child_value=child_value_from_root_perspective,
    )


def completed_audit(
    game_id: str,
    game_outcome: float = -1.0,
    final_current_player: int = 1,
    termination_reason: ResignationTerminationReason = ResignationTerminationReason.NATURAL,
    audit_cutoff_threshold: float | None = None,
) -> CompletedResignationAudit:
    return CompletedResignationAudit(
        game_id=game_id,
        observations=(observation(),),
        audit_cutoff_threshold=audit_cutoff_threshold,
        audit_cutoff_ply=None,
        final_current_player=final_current_player,
        game_outcome=game_outcome,
        termination_reason=termination_reason,
    )


def calibration_parameters(**changes: object) -> ResignationParams:
    parameters = ResignationParams(
        audit_enabled=True,
        audit_game_probability=1.0,
        minimum_completed_audit_triggers=100,
        calibration_window_size=200,
        calibration_interval=100,
        threshold_candidate_minimum=-0.99,
        threshold_candidate_maximum=-0.90,
        threshold_candidate_resolution=0.09,
        maximum_threshold_change_per_calibration=0.20,
    )
    return replace(parameters, **changes)


def test_default_configuration_preserves_existing_self_play_behavior(tmp_path: Path) -> None:
    manager = ResignationManager(str(tmp_path), ResignationParams())

    assert manager.assignment(model_version=100).is_audit_game is False
    assert manager.assignment(model_version=100).production_resignation_enabled is False
    assert not (tmp_path / 'resignation').exists()


def test_native_child_edge_value_is_converted_to_root_perspective() -> None:
    children = [
        SimpleNamespace(visits=10, result_sum=5.0),
        SimpleNamespace(visits=5, result_sum=-4.0),
    ]

    assert best_child_value_from_root_perspective(children) == pytest.approx(-0.5)


def test_root_and_best_child_must_both_cross_the_governing_threshold() -> None:
    client = object.__new__(SelfPlayCpp)
    client.args = SimpleNamespace(resignation=ResignationParams(minimum_eligible_ply=0))
    client.iteration = 40
    game = SelfPlayGame(is_resignation_audit=True, resignation_threshold=-0.90)
    result = SimpleNamespace(
        result=-0.95,
        root=SimpleNamespace(children=[SimpleNamespace(visits=10, result_sum=-9.5)]),
    )

    assert client._record_resignation_observation(game, result) is False
    assert game.resignation_trigger_ply is None

    result.root.children[0].result_sum = 9.5
    assert client._record_resignation_observation(game, result) is True
    assert game.resignation_trigger_ply == 0


def test_audit_without_cutoff_records_full_trajectory_without_stopping_replay() -> None:
    client = object.__new__(SelfPlayCpp)
    client.args = SimpleNamespace(resignation=ResignationParams(minimum_eligible_ply=0))
    client.iteration = 40
    game = SelfPlayGame(is_resignation_audit=True, resignation_threshold=None)
    result = SimpleNamespace(
        result=-0.99,
        root=SimpleNamespace(children=[SimpleNamespace(visits=10, result_sum=9.9)]),
    )

    assert client._record_resignation_observation(game, result) is False
    assert len(game.resignation_observations) == 1
    assert game.resignation_trigger_ply is None


def test_post_cutoff_audit_positions_cannot_enter_ordinary_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = object.__new__(SelfPlayCpp)
    client.args = SimpleNamespace(mcts=SimpleNamespace(playout_cap_randomization=1.0))
    game = SelfPlayGame(is_resignation_audit=True, resignation_threshold=-0.97)
    game.resignation_trigger_ply = 80
    monkeypatch.setattr('src.self_play.SelfPlayCpp.random.random', lambda: 0.0)

    assert client._should_run_full_search(game, force_fast_endgame_playout=False) is False


def test_audit_assignment_survives_game_copy() -> None:
    game = SelfPlayGame(
        is_resignation_audit=True,
        production_resignation_enabled=False,
        resignation_threshold=-0.97,
    )
    copied = game.copy()

    assert copied.game_id == game.game_id
    assert copied.is_resignation_audit is True
    assert copied.production_resignation_enabled is False
    assert copied.resignation_threshold == pytest.approx(-0.97)


def test_audit_assignment_can_never_enable_production_resignation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameters = calibration_parameters(
        audit_enabled=True,
        production_enabled=True,
        audit_cutoff_threshold=-0.97,
    )
    monkeypatch.setattr('src.self_play.resignation.random.random', lambda: 0.0)

    assignment = ResignationManager(str(tmp_path), parameters).assignment(model_version=100)

    assert assignment.is_audit_game is True
    assert assignment.production_resignation_enabled is False
    assert assignment.governing_threshold == pytest.approx(-0.97)


def test_audit_cutoff_increases_until_calibration_selects_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameters = calibration_parameters(
        audit_cutoff_threshold=-0.95,
        audit_cutoff_increase_per_model=0.01,
        minimum_published_model_version=30,
        threshold_candidate_maximum=-0.80,
        threshold_candidate_resolution=0.01,
    )
    monkeypatch.setattr('src.self_play.resignation.random.random', lambda: 0.0)
    manager = ResignationManager(str(tmp_path), parameters)

    assert manager.assignment(model_version=29).is_audit_game is False
    assert manager.assignment(model_version=30).governing_threshold == pytest.approx(-0.95)
    assert manager.assignment(model_version=31).governing_threshold == pytest.approx(-0.94)
    assert manager.assignment(model_version=45).governing_threshold == pytest.approx(-0.80)
    assert manager.assignment(model_version=100).governing_threshold == pytest.approx(-0.80)

    for index in range(100):
        manager.record_completed_audit(completed_audit(str(index)))

    assert manager.assignment(model_version=100).governing_threshold == pytest.approx(-0.80)


def test_production_resignation_uses_bootstrap_cutoff_until_calibration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameters = calibration_parameters(
        audit_game_probability=0.0,
        production_enabled=True,
        audit_cutoff_threshold=-0.95,
        audit_cutoff_increase_per_model=0.01,
        minimum_published_model_version=30,
        threshold_candidate_maximum=-0.80,
        threshold_candidate_resolution=0.01,
        production_resignation_fade_versions=0,
        require_external_safety_approval=False,
    )
    monkeypatch.setattr('src.self_play.resignation.random.random', lambda: 0.0)
    manager = ResignationManager(str(tmp_path), parameters)

    assert manager.assignment(model_version=29).production_resignation_enabled is False
    assignment = manager.assignment(model_version=30)
    assert assignment.production_resignation_enabled is True
    assert assignment.governing_threshold == pytest.approx(-0.95)
    assert manager.assignment(model_version=45).governing_threshold == pytest.approx(-0.80)

    for index in range(99):
        manager.record_completed_audit(completed_audit(str(index), audit_cutoff_threshold=-0.80))
    unsafe_state = manager.record_completed_audit(
        completed_audit('recovered-draw', game_outcome=0.0, audit_cutoff_threshold=-0.80),
    )

    assert unsafe_state.threshold_statistics
    assert unsafe_state.bootstrap_aggressiveness_complete is True
    assert unsafe_state.selected_threshold_is_safe is False
    assert manager.assignment(model_version=46).production_resignation_enabled is False


def test_safe_bootstrap_evidence_keeps_increasing_aggressiveness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameters = calibration_parameters(
        audit_game_probability=0.0,
        production_enabled=True,
        audit_cutoff_threshold=-0.95,
        audit_cutoff_increase_per_model=0.01,
        minimum_published_model_version=30,
        threshold_candidate_maximum=-0.80,
        threshold_candidate_resolution=0.01,
        production_resignation_fade_versions=0,
        require_external_safety_approval=False,
    )
    monkeypatch.setattr('src.self_play.resignation.random.random', lambda: 0.0)
    manager = ResignationManager(str(tmp_path), parameters)

    for index in range(100):
        state = manager.record_completed_audit(
            completed_audit(str(index), audit_cutoff_threshold=-0.95),
        )

    assert state.bootstrap_aggressiveness_complete is False
    assignment = manager.assignment(model_version=31)
    assert assignment.production_resignation_enabled is True
    assert assignment.governing_threshold == pytest.approx(-0.94)


def test_real_resignation_is_a_hard_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    client = object.__new__(SelfPlayCpp)
    game = SelfPlayGame(production_resignation_enabled=True, resignation_threshold=-0.95)
    replacement = SelfPlayGame()
    calls: list[tuple[float, ResignationTerminationReason]] = []

    def handle(
        handled_game: SelfPlayGame,
        outcome: float,
        reason: ResignationTerminationReason,
    ) -> SelfPlayGame:
        assert handled_game is game
        calls.append((outcome, reason))
        return replacement

    monkeypatch.setattr(client, '_handle_end_of_game', handle)

    assert client._handle_resignation(game) is replacement
    assert calls == [(-1.0, ResignationTerminationReason.RESIGNATION)]


def test_exact_upper_bound_matches_minigo_three_percent_gate() -> None:
    assert one_sided_binomial_upper_bound(0, 100, 0.95) == pytest.approx(0.029513, abs=1e-6)
    assert one_sided_binomial_upper_bound(1, 100, 0.95) > 0.03
    assert one_sided_binomial_upper_bound(0, 98, 0.95) > 0.03


def test_calibration_counts_recovered_draw_as_false_non_loss(tmp_path: Path) -> None:
    parameters = calibration_parameters()
    manager = ResignationManager(str(tmp_path), parameters)
    for index in range(99):
        manager.record_completed_audit(completed_audit(str(index)))
    state = manager.record_completed_audit(
        completed_audit('draw', game_outcome=0.0),
    )

    aggressive = state.threshold_statistics[-1]
    assert aggressive.completed_triggers == 100
    assert aggressive.recovered_wins == 0
    assert aggressive.recovered_draws == 1
    assert aggressive.false_non_loss_upper_confidence > 0.03
    assert state.selected_threshold is None
    assert state.selected_threshold_is_safe is False

    production_manager = ResignationManager(
        str(tmp_path),
        replace(
            parameters,
            audit_enabled=False,
            production_enabled=True,
            require_external_safety_approval=False,
        ),
    )
    assert production_manager.assignment(model_version=100).production_resignation_enabled is False


def test_calibration_selects_safe_threshold_and_persists_across_restart(tmp_path: Path) -> None:
    manager = ResignationManager(str(tmp_path), calibration_parameters())
    for index in range(100):
        state = manager.record_completed_audit(completed_audit(str(index)))

    assert state.selected_threshold == pytest.approx(-0.90)
    assert state.selected_threshold_is_safe is True

    restarted = ResignationManager(str(tmp_path), calibration_parameters())
    assert restarted.state.selected_threshold == pytest.approx(-0.90)
    assert restarted.state.total_completed_trigger_count == 100
    assert restarted.state.completed_trigger_count_at_last_calibration == 100


def test_capped_audits_are_excluded_from_primary_calibration(tmp_path: Path) -> None:
    manager = ResignationManager(str(tmp_path), calibration_parameters())
    for index in range(100):
        state = manager.record_completed_audit(
            completed_audit(
                str(index),
                termination_reason=ResignationTerminationReason.PLY_CAP,
            )
        )

    assert state.completed_trigger_count_at_last_calibration == 0
    assert state.completed_audit_count_at_last_calibration == 100
    assert all(statistics.completed_triggers == 0 for statistics in state.threshold_statistics)


def test_enabling_production_preserves_compatible_audit_calibration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit_parameters = calibration_parameters()
    manager = ResignationManager(str(tmp_path), audit_parameters)
    for index in range(100):
        manager.record_completed_audit(completed_audit(str(index)))

    production_parameters = replace(
        audit_parameters,
        audit_enabled=False,
        production_enabled=True,
        production_resignation_enable_fraction=1.0,
        production_resignation_fade_versions=0,
        require_external_safety_approval=False,
    )
    monkeypatch.setattr('src.self_play.resignation.random.random', lambda: 0.0)
    restarted = ResignationManager(str(tmp_path), production_parameters)
    assignment = restarted.assignment(model_version=40)

    assert assignment.is_audit_game is False
    assert assignment.production_resignation_enabled is True
    assert assignment.governing_threshold == pytest.approx(-0.90)


def test_external_safety_approval_can_disable_production_without_self_play_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit_parameters = calibration_parameters()
    manager = ResignationManager(str(tmp_path), audit_parameters)
    for index in range(100):
        manager.record_completed_audit(completed_audit(str(index)))

    production_parameters = replace(
        audit_parameters,
        audit_enabled=False,
        production_enabled=True,
        production_resignation_fade_versions=0,
        require_external_safety_approval=True,
    )
    monkeypatch.setattr('src.self_play.resignation.random.random', lambda: 0.0)
    production_manager = ResignationManager(str(tmp_path), production_parameters)
    assert production_manager.assignment(model_version=40).production_resignation_enabled is False

    approval = ExternalResignationSafetyApproval(
        approved=True,
        approved_through_model_version=50,
        reason='Trainer-held-out safety checks passed.',
    )
    approval_path = tmp_path / 'resignation' / 'external-safety-approval.json'
    approval_path.write_text(approval.model_dump_json(), encoding='utf-8')
    assert production_manager.assignment(model_version=40).production_resignation_enabled is True
    assert production_manager.assignment(model_version=51).production_resignation_enabled is False
    approval_path.write_text('{invalid', encoding='utf-8')
    assert production_manager.assignment(model_version=40).production_resignation_enabled is False


def test_threshold_movement_is_rate_limited(tmp_path: Path) -> None:
    parameters = calibration_parameters(
        threshold_candidate_minimum=-0.99,
        threshold_candidate_maximum=-0.90,
        threshold_candidate_resolution=0.01,
        maximum_threshold_change_per_calibration=0.01,
    )
    manager = ResignationManager(str(tmp_path), parameters)
    for index in range(100):
        state = manager.record_completed_audit(completed_audit(str(index)))

    assert state.selected_threshold == pytest.approx(-0.98)
    assert state.selected_threshold_is_safe is False


def test_calibration_cadence_remains_monotonic_after_rolling_eviction(tmp_path: Path) -> None:
    parameters = calibration_parameters(calibration_window_size=100)
    manager = ResignationManager(str(tmp_path), parameters)
    for index in range(200):
        state = manager.record_completed_audit(completed_audit(str(index)))

    assert state.total_completed_trigger_count == 200
    assert state.completed_trigger_count_at_last_calibration == 200


def test_insufficient_recent_triggers_automatically_disable_production(tmp_path: Path) -> None:
    parameters = calibration_parameters(calibration_window_size=100)
    manager = ResignationManager(str(tmp_path), parameters)
    for index in range(100):
        state = manager.record_completed_audit(completed_audit(f'trigger-{index}'))
    assert state.selected_threshold_is_safe is True

    for index in range(100):
        no_trigger = completed_audit(f'quiet-{index}').model_copy(
            update={'observations': (observation(root_value=0.0, child_value_from_root_perspective=0.0),)}
        )
        state = manager.record_completed_audit(no_trigger)

    assert state.selected_threshold is None
    assert state.selected_threshold_is_safe is False
    assert state.total_completed_trigger_count == 100
    assert state.completed_audit_count_at_last_calibration == 200


def test_duplicate_audit_submission_is_idempotent_across_restart(tmp_path: Path) -> None:
    parameters = calibration_parameters()
    audit = completed_audit('stable-game-id')
    first_manager = ResignationManager(str(tmp_path), parameters)
    first_state = first_manager.record_completed_audit(audit)

    restarted_manager = ResignationManager(str(tmp_path), parameters)
    duplicate_state = restarted_manager.record_completed_audit(audit)

    audit_database_path = tmp_path / 'resignation' / 'completed-audits.sqlite3'
    with sqlite3.connect(audit_database_path) as audit_database:
        (audit_count,) = audit_database.execute('SELECT COUNT(*) FROM completed_audits').fetchone()
    assert audit_count == 1
    assert first_state.total_completed_trigger_count == 1
    assert duplicate_state.total_completed_trigger_count == 1


def test_interrupted_audit_append_is_recovered_before_next_commit(tmp_path: Path) -> None:
    parameters = calibration_parameters()
    manager = ResignationManager(str(tmp_path), parameters)
    manager.record_completed_audit(completed_audit('first'))
    audit_database_path = tmp_path / 'resignation' / 'completed-audits.sqlite3'
    interrupted = completed_audit('interrupted')
    with sqlite3.connect(audit_database_path) as audit_database:
        audit_database.execute(
            """
            INSERT INTO completed_audits (
                game_id,
                payload,
                is_natural_maximum_threshold_trigger
            ) VALUES (?, ?, ?)
            """,
            (interrupted.game_id, interrupted.model_dump_json(), 1),
        )
        audit_database.execute(
            """
            UPDATE audit_metadata
            SET
                record_count = record_count + 1,
                natural_trigger_count = natural_trigger_count + 1
            WHERE singleton = 1
            """
        )

    restarted = ResignationManager(str(tmp_path), parameters)
    assert restarted.state.total_completed_trigger_count == 2
    duplicate_state = restarted.record_completed_audit(interrupted)
    persisted_state = ResignationManager(str(tmp_path), parameters).state
    assert duplicate_state.total_completed_trigger_count == 2
    assert persisted_state.total_completed_trigger_count == 2

    state = restarted.record_completed_audit(completed_audit('next'))

    assert state.total_completed_trigger_count == 3
    with sqlite3.connect(audit_database_path) as audit_database:
        (audit_count,) = audit_database.execute('SELECT COUNT(*) FROM completed_audits').fetchone()
    assert audit_count == 3


def test_restart_recovers_safety_state_before_production_assignment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameters = calibration_parameters(
        minimum_completed_audit_triggers=2,
        calibration_window_size=10,
        calibration_interval=2,
        false_non_loss_upper_bound=0.9,
    )
    manager = ResignationManager(str(tmp_path), parameters)
    manager.record_completed_audit(completed_audit('first'))
    second = completed_audit('second')
    audit_database_path = tmp_path / 'resignation' / 'completed-audits.sqlite3'
    with sqlite3.connect(audit_database_path) as audit_database:
        audit_database.execute(
            """
            INSERT INTO completed_audits (
                game_id,
                payload,
                is_natural_maximum_threshold_trigger
            ) VALUES (?, ?, ?)
            """,
            (second.game_id, second.model_dump_json(), 1),
        )
        audit_database.execute(
            """
            UPDATE audit_metadata
            SET
                record_count = record_count + 1,
                natural_trigger_count = natural_trigger_count + 1
            WHERE singleton = 1
            """
        )

    production_parameters = replace(
        parameters,
        audit_enabled=False,
        production_enabled=True,
        production_resignation_fade_versions=0,
        require_external_safety_approval=False,
    )
    monkeypatch.setattr('src.self_play.resignation.random.random', lambda: 0.0)
    restarted = ResignationManager(str(tmp_path), production_parameters)

    assert restarted.state.total_completed_trigger_count == 2
    assert restarted.state.selected_threshold_is_safe is True
    assert restarted.assignment(model_version=40).production_resignation_enabled is True
