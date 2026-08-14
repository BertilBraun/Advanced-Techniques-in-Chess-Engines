from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import UUID

import pytest
from AlphaZeroCpp import GameSearchVisit

from src.experiment.configuration import load_experiment_configuration
from src.games.contracts import WdlTarget
from src.self_play.completed_game import (
    CompletedSelfPlayGame,
    GameIdentity,
    SearchObservation,
    TerminationReason,
)
from src.self_play.parameters import RestartStateStartParameters
from src.self_play.restart_archive import RestartStateArchive, worker_restart_archive_path


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

    parameters = configuration.go.self_play.resolve(0, configuration.go.rules.maximum_moves, 1.0)

    match parameters.start_position:
        case RestartStateStartParameters() as start_position:
            assert start_position == restart_parameters()
        case _:
            pytest.fail('Restart-state screen resolved the wrong start-position variant.')


def completed_game(
    identity_number: int,
    visits: tuple[tuple[int, int], ...] = ((0, 55), (1, 30), (2, 15)),
    *,
    action_count: int = 15,
    observation_ply: int = 0,
    selected_action_id: int = 9,
    root_value: float = 0.3,
    full_search: bool = True,
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
                policy_target_visits=tuple(
                    GameSearchVisit(action_id=action_id, visit_count=visit_count) for action_id, visit_count in visits
                ),
                root_value=root_value,
                highest_visited_child_action_id=max(visits, key=lambda item: item[1])[0],
                highest_visited_child_visit_count=max(item[1] for item in visits),
                highest_visited_child_q=root_value,
                selected_action_id=selected_action_id,
                full_search=full_search,
                sample_weight=1.0,
                search_budget=256,
            ),
        ),
        final_wdl=WdlTarget(win=0.0, draw=1.0, loss=0.0),
        termination_reason=TerminationReason.MAXIMUM_PLIES,
    )


def test_archive_uses_recorded_target_visits_and_inclusive_boundaries(tmp_path: Path) -> None:
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
        (completed_game(2, visits=((0, 100), (1, 10))), 0),
        (completed_game(3, visits=((0, 30), (1, 25), (2, 20), (3, 15), (4, 10))), 0),
        (completed_game(4, action_count=14), 0),
        (completed_game(5, root_value=0.3001), 0),
        (completed_game(6, full_search=False), 0),
        (completed_game(7, visits=((0, 55), (1, 30), (2, 15))), 1),
        (completed_game(8, visits=((0, 45), (1, 25), (2, 20), (3, 10))), 1),
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


def test_worker_archives_use_distinct_stable_paths_without_shared_initialization(tmp_path: Path) -> None:
    completed_games_path = tmp_path / 'completed-games'
    paths = tuple(worker_restart_archive_path(completed_games_path, worker_id) for worker_id in range(24))

    def open_and_close(path: Path) -> None:
        archive = RestartStateArchive(path)
        archive.close()

    with ThreadPoolExecutor(max_workers=24) as executor:
        tuple(executor.map(open_and_close, paths))

    assert len(set(paths)) == 24
    assert paths[0] == completed_games_path / 'restart-states' / 'worker-0.sqlite3'
    assert all(path.is_file() for path in paths)


def test_worker_archive_path_rejects_negative_worker_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='nonnegative'):
        worker_restart_archive_path(tmp_path, -1)


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
