from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import cast

import pytest

from src.games.implementation import GameImplementation
from src.games.contracts import WdlTarget
from src.self_play.parameters import ResolvedSelfPlayParameters
from src.self_play.completed_game import SparseSearchVisit, TerminationReason
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
        assert action_id == 0
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
        return True

    def inference_statistics(self) -> FakeInferenceStatistics:
        return FakeInferenceStatistics()


class FakeState:
    def initial_position(self) -> FakePosition:
        return FakePosition(0)

    def legal_action_ids(self, position: FakePosition) -> tuple[int, ...]:
        del position
        return (0,)

    def child_position(self, position: FakePosition, action_id: int) -> FakePosition:
        assert action_id == 0
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

    def __init__(self) -> None:
        self.search = FakeSearch()

    def self_play_parameters_at(self, model_generation: int) -> ResolvedSelfPlayParameters:
        del model_generation
        return ResolvedSelfPlayParameters(
            random_opening_plies=0,
            full_search_probability=1.0,
            parallel_searches=1,
            full_searches=3,
            fast_searches=1,
            minimum_root_visits=0,
            exploration_constant=1.0,
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
