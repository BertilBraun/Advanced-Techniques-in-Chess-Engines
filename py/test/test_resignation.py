import sqlite3
from pathlib import Path
from unittest.mock import MagicMock
from uuid import UUID

import pytest
import src.self_play.resignation as resignation_module
from AlphaZeroCpp import GameSearchVisit
from pydantic import ValidationError
from src.games.contracts import WdlTarget
from src.self_play.completed_game import (
    CompletedSelfPlayGame,
    GameIdentity,
    SearchObservation,
    SearchStopReason,
    TerminationReason,
)
from src.self_play.resignation import (
    CalibratedResignationConfiguration,
    ResignationCalibrator,
    one_sided_binomial_upper_bound,
)


def configuration(**updates: float) -> CalibratedResignationConfiguration:
    values: dict[str, float | int | str] = {
        'kind': 'calibrated',
        'first_production_generation': 50,
        'false_nonloss_rate_ceiling': 0.03,
        'continuation_game_probability': 0.10,
        'triggered_game_window': 2000,
        'candidate_threshold_minimum': -0.99,
        'candidate_threshold_maximum': -0.70,
        'candidate_threshold_step': 0.01,
        'minimum_evidence_trigger_count': 100,
        'confidence_level': 0.95,
        'maximum_relaxation_per_generation': 0.01,
    }
    values.update(updates)
    return CalibratedResignationConfiguration.model_validate(values)


def completed_continuation(
    game_number: int,
    root_value: float = -0.99,
    child_q: float = -0.99,
    final_wdl: WdlTarget = WdlTarget(win=0.0, draw=0.0, loss=1.0),
) -> CompletedSelfPlayGame:
    return CompletedSelfPlayGame(
        identity=GameIdentity(
            worker_id=game_number % 4,
            process_instance_id=UUID('00000000-0000-0000-0000-000000000001'),
            game_number=game_number,
        ),
        created_at_seconds=float(game_number),
        generation_seconds=1.0,
        action_ids=(0, 1),
        observations=(
            SearchObservation(
                ply=0,
                model_generation=10,
                policy_target_visits=(GameSearchVisit(action_id=0, visit_count=20),),
                root_value=root_value,
                highest_visited_child_action_id=0,
                highest_visited_child_visit_count=20,
                highest_visited_child_q=child_q,
                selected_action_id=0,
                full_search=True,
                sample_weight=1.0,
                search_budget=20,
                network_root_value=root_value,
                policy_correction=0.0,
                value_correction=0.0,
                search_correction_target=0.0,
                predicted_search_correction=0.0,
                starting_visits=0,
                final_visits=20,
                stop_reason=SearchStopReason.FIXED_LIMIT,
            ),
        ),
        final_wdl=final_wdl,
        termination_reason=TerminationReason.NATURAL,
        is_resignation_continuation=True,
    )


def observe_completed_games(
    calibrator: ResignationCalibrator,
    *games: CompletedSelfPlayGame,
) -> None:
    with calibrator.calibration_batch() as calibration_batch:
        for game in games:
            calibration_batch.observe_completed_game(game)


def test_configuration_rejects_extra_fields_and_misaligned_grid() -> None:
    with pytest.raises(ValidationError, match='extra_forbidden'):
        configuration(unknown=1)
    with pytest.raises(ValidationError, match='integral number'):
        configuration(candidate_threshold_step=0.04)


def test_exact_one_sided_binomial_bound_matches_three_percent_gate() -> None:
    assert one_sided_binomial_upper_bound(0, 100, 0.95) == pytest.approx(0.029513, abs=1e-6)
    assert one_sided_binomial_upper_bound(1, 100, 0.95) > 0.03


def test_calibration_requires_both_root_and_exact_child_q(tmp_path: Path) -> None:
    calibrator = ResignationCalibrator(tmp_path / 'calibration.json', configuration(minimum_evidence_trigger_count=1))
    observe_completed_games(calibrator, completed_continuation(0, root_value=-0.95, child_q=-0.69))
    assert calibrator.state.triggered_continuation_games == ()


def test_adjudicated_continuation_is_not_calibration_evidence(tmp_path: Path) -> None:
    calibrator = ResignationCalibrator(tmp_path / 'calibration.json', configuration(minimum_evidence_trigger_count=1))
    adjudicated = completed_continuation(0).model_copy(
        update={
            'termination_reason': TerminationReason.MAXIMUM_PLIES,
            'final_wdl': WdlTarget(win=0.2, draw=0.6, loss=0.2),
        }
    )

    observe_completed_games(calibrator, adjudicated)
    calibrator.advance_generation(50)

    assert calibrator.state.completed_continuation_games == 1
    assert calibrator.state.triggered_continuation_games == ()
    assert calibrator.published_policy(50).threshold is None


def test_calibration_selects_grid_candidate_and_counts_draw_as_false_nonloss(tmp_path: Path) -> None:
    calibrator = ResignationCalibrator(tmp_path / 'calibration.json', configuration())
    observe_completed_games(
        calibrator,
        *(completed_continuation(game_number) for game_number in range(99)),
        completed_continuation(99, final_wdl=WdlTarget(win=0.0, draw=1.0, loss=0.0)),
    )
    assert calibrator.state.selected_threshold is None

    safe = ResignationCalibrator(tmp_path / 'safe.json', configuration())
    observe_completed_games(safe, *(completed_continuation(game_number) for game_number in range(100)))
    safe.advance_generation(49)
    assert safe.state.selected_threshold == pytest.approx(-0.99)
    assert safe.published_policy(49).threshold is None
    assert safe.published_policy(50).threshold == pytest.approx(-0.99)


def test_triggered_window_and_generation_relaxation_cap(tmp_path: Path) -> None:
    calibrator = ResignationCalibrator(
        tmp_path / 'calibration.json',
        configuration(
            triggered_game_window=3,
            minimum_evidence_trigger_count=1,
            false_nonloss_rate_ceiling=0.99,
        ),
    )
    observe_completed_games(calibrator, *(completed_continuation(game_number) for game_number in range(5)))
    calibrator.advance_generation(10)
    assert tuple(item.game_identity.rsplit(':', 1)[-1] for item in calibrator.state.triggered_continuation_games) == (
        '2',
        '3',
        '4',
    )
    assert calibrator.state.selected_threshold == pytest.approx(-0.99)
    calibrator.advance_generation(11)
    assert calibrator.state.selected_threshold == pytest.approx(-0.98)
    calibrator.advance_generation(11)
    assert calibrator.state.selected_threshold == pytest.approx(-0.98)
    calibrator.advance_generation(12)
    assert calibrator.state.selected_threshold == pytest.approx(-0.97)


def test_central_state_persists_across_restart(tmp_path: Path) -> None:
    path = tmp_path / 'calibration.json'
    parameters = configuration(minimum_evidence_trigger_count=1)
    first = ResignationCalibrator(path, parameters)
    observe_completed_games(first, completed_continuation(0))
    first.advance_generation(50)

    restarted = ResignationCalibrator(path, parameters)
    assert restarted.state == first.state
    assert restarted.published_policy(50) == first.published_policy(50)

    observe_completed_games(restarted, completed_continuation(0))
    assert restarted.state.completed_continuation_games == 1
    assert restarted.state.broadest_candidate_triggers == 1


def test_calibration_batch_uses_one_transaction_and_state_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / 'calibration.json'
    calibrator = ResignationCalibrator(path, configuration())
    statements: list[str] = []
    connect = sqlite3.connect

    def traced_connect(database: Path) -> sqlite3.Connection:
        journal = connect(database)
        journal.set_trace_callback(statements.append)
        return journal

    state_writer = MagicMock(wraps=resignation_module.write_text_atomically)
    monkeypatch.setattr(resignation_module.sqlite3, 'connect', traced_connect)
    monkeypatch.setattr(resignation_module, 'write_text_atomically', state_writer)

    observe_completed_games(
        calibrator,
        completed_continuation(0),
        completed_continuation(1),
        completed_continuation(0),
    )

    transaction_statements = tuple(statement for statement in statements if statement in {'BEGIN ', 'COMMIT'})
    assert transaction_statements == ('BEGIN ', 'COMMIT')
    assert state_writer.call_count == 1
    assert calibrator.state.completed_continuation_games == 2
    assert calibrator.state.broadest_candidate_triggers == 2
