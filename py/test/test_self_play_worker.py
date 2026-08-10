from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import cast
from uuid import UUID

import numpy as np
import pytest

from src.games.implementation import GameImplementation
from src.games.contracts import WdlTarget
from src.self_play.parameters import RandomOpeningStartParameters, ResolvedSelfPlayParameters
from src.self_play.parameters import RestartStateStartParameters
from src.self_play.completed_game import CompletedSelfPlayGame, GameIdentity, SearchObservation
from src.self_play.completed_game import SparseSearchVisit, TerminationReason
from src.self_play.restart_archive import RestartStateArchive
from src.self_play.worker import SelfPlayWorker
from src.training.checkpoint import CheckpointReference


@dataclass(frozen=True)
class FakePosition:
    ply: int


@dataclass
class FakeRoot:
    position: FakePosition
    is_terminal: bool = False
    reset_count: int = 0

    def play(self, action_id: int) -> None:
        assert action_id in (0, 1)
        self.position = FakePosition(self.position.ply + 1)

    def reset(self) -> None:
        self.reset_count += 1

    def discount(self, retained_fraction: float) -> None:
        assert retained_fraction == 0.5


@dataclass(frozen=True)
class FakeVisit:
    action_id: int
    visit_count: int


@dataclass(frozen=True)
class FakeRequest:
    root: FakeRoot
    full_search: bool


@dataclass(frozen=True)
class FakeResult:
    root_value: float
    visits: list[FakeVisit]
    root: FakeRoot


@dataclass(frozen=True)
class FakeBatch:
    results: list[FakeResult]
    simulations_completed: int


@dataclass(frozen=True)
class FakeInferenceStatistics:
    averageNumberOfPositionsInInferenceCall: float = 2.0


class FakeSearch:
    def __init__(self) -> None:
        self.generations: list[int] = []
        self.capacity_changed = False

    def new_root(self, position: FakePosition) -> FakeRoot:
        return FakeRoot(position)

    def request(self, root: FakeRoot, full_search: bool) -> FakeRequest:
        return FakeRequest(root, full_search)

    def search(self, requests: list[FakeRequest], collect_statistics: bool = False) -> FakeBatch:
        assert not collect_statistics
        return FakeBatch(
            [FakeResult(0.25, [FakeVisit(0, 3)], request.root) for request in requests],
            simulations_completed=len(requests) * 3,
        )

    def refresh_model(self, model_generation: int, model_path: str) -> None:
        assert Path(model_path).suffixes == ['.jit', '.pt']
        self.generations.append(model_generation)

    def update_search_schedule(self, search_parameters: ResolvedSelfPlayParameters) -> bool:
        del search_parameters
        return self.capacity_changed

    def inference_statistics(self) -> FakeInferenceStatistics:
        return FakeInferenceStatistics()


class FakeState:
    def initial_position(self) -> FakePosition:
        return FakePosition(0)

    def legal_action_ids(self, position: FakePosition) -> tuple[int, ...]:
        del position
        return (0, 1)

    def child_position(self, position: FakePosition, action_id: int) -> FakePosition:
        assert action_id in (0, 1)
        return FakePosition(position.ply + 1)

    def natural_terminal_wdl(self, position: FakePosition) -> WdlTarget | None:
        del position
        return None

    def adjudicated_wdl(self, position: FakePosition, reason: TerminationReason) -> WdlTarget:
        del position, reason
        return WdlTarget(win=0.0, draw=1.0, loss=0.0)


@dataclass(frozen=True)
class FakeTraining:
    random_seed: int = 5


class FakeGame:
    training = FakeTraining()
    state = FakeState()

    def __init__(
        self,
        maximum_random_opening_plies: int = 0,
        restart_parameters: RestartStateStartParameters | None = None,
    ) -> None:
        self.search = FakeSearch()
        self.maximum_random_opening_plies = maximum_random_opening_plies
        self.restart_parameters = restart_parameters

    def self_play_parameters_at(self, model_generation: int) -> ResolvedSelfPlayParameters:
        del model_generation
        start_position = self.restart_parameters
        if start_position is None:
            start_position = RandomOpeningStartParameters(
                kind='random_opening', maximum_plies=self.maximum_random_opening_plies
            )
        return ResolvedSelfPlayParameters(
            start_position=start_position,
            full_search_probability=1.0,
            parallel_searches=1,
            full_searches=3,
            fast_searches=1,
            minimum_root_visits=0,
            exploration_constant=1.0,
            fpu_reduction=0.0,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
            retained_root_visit_fraction=0.5,
            starting_temperature=1.0,
            final_temperature=1.0,
            greedy_after_ply=1,
            maximum_game_plies=None,
            primary_sample_weight=1.0,
        )

    def create_native_search(
        self,
        device_id: int,
        checkpoint: CheckpointReference,
        parameters: ResolvedSelfPlayParameters,
    ) -> FakeSearch:
        del device_id, checkpoint, parameters
        return self.search

    def native_search_parameters(self, parameters: ResolvedSelfPlayParameters) -> ResolvedSelfPlayParameters:
        return parameters


def checkpoint(path: Path, generation: int) -> CheckpointReference:
    inference_path = path / f'model_{generation}.jit.pt'
    inference_path.write_bytes(f'model {generation}'.encode('ascii'))
    return CheckpointReference(
        generation=generation,
        manifest_path=path / f'checkpoint_{generation}.json',
        model_path=path / f'model_{generation}.pt',
        optimizer_path=path / f'optimizer_{generation}.pt',
        inference_model_path=inference_path,
        inference_model_sha256=hashlib.sha256(inference_path.read_bytes()).hexdigest(),
    )


def test_worker_owns_shared_search_move_selection_and_generation_transition(tmp_path: Path) -> None:
    game = FakeGame()
    worker = SelfPlayWorker(
        cast(GameImplementation, game),
        parallel_game_count=3,
        worker_id=2,
        device_id=0,
        inbox_path=tmp_path,
    )

    with pytest.raises(RuntimeError, match='model must be loaded'):
        worker.run_batch()

    worker.refresh_published_model(checkpoint(tmp_path, 0))
    worker.run_batch()
    worker.refresh_published_model(checkpoint(tmp_path, 1))
    statistics = worker.snapshot_statistics()

    assert [game.root.position.ply for game in worker.active_games] == [1, 1, 1]
    assert [game.root.reset_count for game in worker.active_games] == [1, 1, 1]
    assert all(game.action_ids == [0] for game in worker.active_games)
    assert all(
        game.observations[0].visits == (SparseSearchVisit(action_id=0, visit_count=3),) for game in worker.active_games
    )
    assert game.search.generations == [1]
    assert statistics.model_generation == 1
    assert statistics.completed_searches == 9


def test_worker_replaces_roots_when_search_arena_capacity_changes(tmp_path: Path) -> None:
    game = FakeGame()
    worker = SelfPlayWorker(
        cast(GameImplementation, game),
        parallel_game_count=2,
        worker_id=0,
        device_id=0,
        inbox_path=tmp_path,
    )
    worker.refresh_published_model(checkpoint(tmp_path, 0))
    worker.run_batch()
    original_roots = tuple(active_game.root for active_game in worker.active_games)

    game.search.capacity_changed = True
    worker.refresh_published_model(checkpoint(tmp_path, 1))

    assert all(active_game.root is not original for active_game, original in zip(worker.active_games, original_roots))
    assert [active_game.root.position.ply for active_game in worker.active_games] == [1, 1]
    assert [active_game.action_ids for active_game in worker.active_games] == [[0], [0]]
    assert [active_game.root.reset_count for active_game in worker.active_games] == [0, 0]


class FakeOpeningRandom:
    def __init__(self, opening_lengths: tuple[int, ...]) -> None:
        self.opening_lengths = iter(opening_lengths)

    def integers(self, low: int, high: int) -> int:
        assert (low, high) == (0, 13)
        return next(self.opening_lengths)

    def choice(self, values: tuple[int, ...]) -> int:
        return values[0]


def test_worker_samples_random_opening_length_from_zero_through_configured_maximum(tmp_path: Path) -> None:
    game = FakeGame(maximum_random_opening_plies=12)
    worker = SelfPlayWorker(
        cast(GameImplementation, game),
        parallel_game_count=2,
        worker_id=0,
        device_id=0,
        inbox_path=tmp_path,
    )
    worker.random = cast(np.random.Generator, FakeOpeningRandom((0, 12)))

    worker.refresh_published_model(checkpoint(tmp_path, 0))

    assert [active_game.root.position.ply for active_game in worker.active_games] == [0, 12]
    assert [active_game.action_ids for active_game in worker.active_games] == [[], [0] * 12]


def restart_parameters(true_start_probability: float = 0.5) -> RestartStateStartParameters:
    return RestartStateStartParameters(
        kind='restart_state',
        true_start_probability=true_start_probability,
        candidate_visit_mass=0.85,
        minimum_candidates=2,
        maximum_candidates=3,
        maximum_absolute_root_value=0.3,
        minimum_remaining_plies=15,
        maximum_archive_positions=50_000,
        maximum_age_generations=20,
    )


class FakeRestartRandom:
    def __init__(self, probability_draw: float) -> None:
        self.probability_draw = probability_draw

    def random(self) -> float:
        return self.probability_draw


def restart_source_game() -> CompletedSelfPlayGame:
    action_ids = (0,) * 15
    return CompletedSelfPlayGame(
        identity=GameIdentity(worker_id=0, process_instance_id=UUID(int=1), game_number=0),
        created_at_seconds=1.0,
        generation_seconds=1.0,
        action_ids=action_ids,
        observations=(
            SearchObservation(
                ply=0,
                model_generation=0,
                visits=(
                    SparseSearchVisit(action_id=0, visit_count=60),
                    SparseSearchVisit(action_id=1, visit_count=30),
                    SparseSearchVisit(action_id=2, visit_count=10),
                ),
                root_value=0.0,
                selected_action_id=0,
                full_search=True,
                sample_weight=1.0,
                search_budget=256,
                minimum_root_visits=0,
            ),
        ),
        final_wdl=WdlTarget(win=0.0, draw=1.0, loss=0.0),
        termination_reason=TerminationReason.MAXIMUM_PLIES,
    )


def test_restart_policy_uses_exact_initial_states_without_random_openings(tmp_path: Path) -> None:
    worker = SelfPlayWorker(
        cast(GameImplementation, FakeGame(restart_parameters=restart_parameters(1.0))),
        parallel_game_count=1,
        worker_id=0,
        device_id=0,
        inbox_path=tmp_path / 'inbox',
    )
    worker.random = cast(np.random.Generator, FakeRestartRandom(0.0))

    worker.refresh_published_model(checkpoint(tmp_path, 0))

    assert worker.active_games[0].action_ids == []
    assert worker.active_games[0].root.position == FakePosition(0)
    assert worker.true_starts == 1
    worker.close()


def test_restart_policy_falls_back_to_exact_start_when_archive_is_empty(tmp_path: Path) -> None:
    worker = SelfPlayWorker(
        cast(GameImplementation, FakeGame(restart_parameters=restart_parameters())),
        parallel_game_count=1,
        worker_id=0,
        device_id=0,
        inbox_path=tmp_path / 'inbox',
    )
    worker.random = cast(np.random.Generator, FakeRestartRandom(0.9))

    worker.refresh_published_model(checkpoint(tmp_path, 0))

    assert worker.active_games[0].action_ids == []
    assert worker.empty_restart_fallbacks == 1
    worker.close()


def test_restart_root_runs_full_search_and_plays_reserved_candidate(tmp_path: Path) -> None:
    parameters = restart_parameters()
    archive = RestartStateArchive(tmp_path / 'restart-states.sqlite3')
    archive.archive_completed_game(restart_source_game(), parameters)
    archive.close()
    game = FakeGame(restart_parameters=parameters)
    worker = SelfPlayWorker(
        cast(GameImplementation, game),
        parallel_game_count=1,
        worker_id=0,
        device_id=0,
        inbox_path=tmp_path / 'inbox',
    )
    worker.random = cast(np.random.Generator, FakeRestartRandom(0.9))
    worker.refresh_published_model(checkpoint(tmp_path, 0))

    worker.run_batch()

    active_game = worker.active_games[0]
    assert active_game.action_ids == [1]
    assert active_game.observations[0].full_search
    assert active_game.observations[0].selected_action_id == 1
    assert worker.restart_starts == 1
    worker.close()
