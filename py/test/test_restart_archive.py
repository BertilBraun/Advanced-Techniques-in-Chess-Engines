from dataclasses import replace
from pathlib import Path
from uuid import UUID

import pytest

from src.experiment.configuration import load_experiment_configuration
from src.games.contracts import WdlTarget
from src.self_play.completed_game import (
    CompletedSelfPlayGame,
    GameIdentity,
    SearchObservation,
    SparseSearchVisit,
    TerminationReason,
)
from src.self_play.parameters import RestartStateStartParameters
from src.self_play.restart_archive import RestartStateArchive


def restart_parameters() -> RestartStateStartParameters:
    return RestartStateStartParameters(
        kind='restart_state',
        true_start_probability=0.5,
        candidate_visit_mass=0.85,
        minimum_candidates=2,
        maximum_candidates=3,
        maximum_absolute_root_value=0.3,
        minimum_remaining_plies=15,
        maximum_archive_positions=50_000,
        maximum_age_generations=20,
    )


def test_restart_state_screen_resolves_canonical_parameters() -> None:
    configuration = load_experiment_configuration(Path('configs/screening/go-7x7-overnight/13-restart-states.yaml'))
    assert configuration.game == 'go'

    parameters = configuration.go.self_play.resolve(0, configuration.go.rules.maximum_moves)

    match parameters.start_position:
        case RestartStateStartParameters() as start_position:
            assert start_position == restart_parameters()
        case _:
            pytest.fail('Restart-state screen resolved the wrong start-position variant.')


def completed_game(
    identity_number: int,
    visits: tuple[tuple[int, int], ...] = ((0, 65), (1, 40), (2, 25)),
    *,
    action_count: int = 15,
    observation_ply: int = 0,
    selected_action_id: int = 9,
    root_value: float = 0.3,
    full_search: bool = True,
    minimum_root_visits: int = 10,
    model_generation: int = 1,
    created_at_seconds: float = 1.0,
) -> CompletedSelfPlayGame:
    action_ids = [4] * action_count
    action_ids[observation_ply] = selected_action_id
    return CompletedSelfPlayGame(
        identity=GameIdentity(
            worker_id=0,
            process_instance_id=UUID(int=identity_number),
            game_number=identity_number,
        ),
        created_at_seconds=created_at_seconds,
        generation_seconds=1.0,
        action_ids=tuple(action_ids),
        observations=(
            SearchObservation(
                ply=observation_ply,
                model_generation=model_generation,
                visits=tuple(
                    SparseSearchVisit(action_id=action_id, visit_count=visit_count) for action_id, visit_count in visits
                ),
                root_value=root_value,
                selected_action_id=selected_action_id,
                full_search=full_search,
                sample_weight=1.0,
                search_budget=256,
                minimum_root_visits=minimum_root_visits,
            ),
        ),
        final_wdl=WdlTarget(win=0.0, draw=1.0, loss=0.0),
        termination_reason=TerminationReason.MAXIMUM_PLIES,
    )


def test_archive_uses_preprocessed_visits_and_inclusive_boundaries(tmp_path: Path) -> None:
    archive = RestartStateArchive(tmp_path / 'restart.sqlite3')

    update = archive.archive_completed_game(completed_game(1), restart_parameters())

    assert update.positions == 1
    assert update.candidates == 2
    assert archive.snapshot().positions == 1
    assert archive.snapshot().candidates == 2
    archive.close()


@pytest.mark.parametrize(
    ('game', 'expected_positions'),
    (
        (completed_game(2, visits=((0, 100), (1, 10)), minimum_root_visits=0), 0),
        (completed_game(3, visits=((0, 30), (1, 25), (2, 20), (3, 15), (4, 10)), minimum_root_visits=0), 0),
        (completed_game(4, action_count=14), 0),
        (completed_game(5, root_value=0.3001), 0),
        (completed_game(6, full_search=False), 0),
        (completed_game(7, visits=((0, 55), (1, 30), (2, 15)), minimum_root_visits=0), 1),
        (completed_game(8, visits=((0, 45), (1, 25), (2, 20), (3, 10)), minimum_root_visits=0), 1),
    ),
)
def test_archive_eligibility_and_smallest_prefix_rules(
    tmp_path: Path,
    game: CompletedSelfPlayGame,
    expected_positions: int,
) -> None:
    archive = RestartStateArchive(tmp_path / f'{game.identity.game_number}.sqlite3')

    update = archive.archive_completed_game(game, restart_parameters())

    assert update.positions == expected_positions
    archive.close()


def test_two_connections_reserve_each_source_candidate_once_and_evict_when_exhausted(tmp_path: Path) -> None:
    path = tmp_path / 'restart.sqlite3'
    first = RestartStateArchive(path)
    second = RestartStateArchive(path)
    first.archive_completed_game(completed_game(1), restart_parameters())

    first_reservation = first.reserve(1, restart_parameters())
    second_reservation = second.reserve(1, restart_parameters())

    assert first_reservation is not None
    assert second_reservation is not None
    assert first_reservation.action_id != second_reservation.action_id
    assert first.reserve(1, restart_parameters()) is None
    assert first.snapshot().positions == 0
    assert first.snapshot().exhausted_evictions == 1
    first.close()
    second.close()


def test_played_candidate_is_not_reserved_again(tmp_path: Path) -> None:
    archive = RestartStateArchive(tmp_path / 'restart.sqlite3')
    game = completed_game(1, selected_action_id=0)
    archive.archive_completed_game(game, restart_parameters())

    reservation = archive.reserve(1, restart_parameters())

    assert reservation is not None
    assert reservation.action_id == 1
    assert archive.reserve(1, restart_parameters()) is None
    archive.close()


def test_capacity_and_generation_age_evict_old_positions(tmp_path: Path) -> None:
    archive = RestartStateArchive(tmp_path / 'restart.sqlite3')
    capacity_one = replace(restart_parameters(), maximum_archive_positions=1)
    archive.archive_completed_game(completed_game(1, created_at_seconds=1.0), capacity_one)
    update = archive.archive_completed_game(completed_game(2, created_at_seconds=2.0), capacity_one)

    assert update.capacity_evictions == 1
    assert archive.snapshot().positions == 1
    assert archive.reserve(22, capacity_one) is None
    snapshot = archive.snapshot()
    assert snapshot.expired_evictions == 1
    assert snapshot.positions == 0
    archive.close()


def test_reopen_preserves_claims_and_rejects_duplicate_position_insertion(tmp_path: Path) -> None:
    path = tmp_path / 'restart.sqlite3'
    game = completed_game(1)
    archive = RestartStateArchive(path)
    archive.archive_completed_game(game, restart_parameters())
    assert archive.reserve(1, restart_parameters()) is not None
    archive.close()

    reopened = RestartStateArchive(path)
    update = reopened.archive_completed_game(game, restart_parameters())

    assert update.positions == 0
    assert reopened.reserve(1, restart_parameters()) is not None
    assert reopened.reserve(1, restart_parameters()) is None
    reopened.close()
