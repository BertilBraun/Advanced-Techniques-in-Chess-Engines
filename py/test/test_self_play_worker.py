from dataclasses import dataclass
from pathlib import Path

import pytest

from src.self_play.worker import GameSelfPlayPolicy, SelfPlayWorker
from src.self_play.active_game import ContinuingGame
from src.self_play.completed_game import GameIdentity


@dataclass(frozen=True)
class FakeSearchRequest:
    value: int


@dataclass(frozen=True)
class FakeSearchResult:
    value: int


class FakeSelfPlayPolicy(GameSelfPlayPolicy[int, FakeSearchRequest, FakeSearchResult, int]):
    def __init__(self) -> None:
        self.model_generation: int | None = None
        self.refreshed_game_counts: list[int] = []
        self.search_batches: list[tuple[FakeSearchRequest, ...]] = []

    def refresh_model(
        self,
        model_generation: int,
        model_path: Path,
        active_games: tuple[int, ...],
    ) -> None:
        assert model_path.suffix == '.pt'
        self.model_generation = model_generation
        self.refreshed_game_counts.append(len(active_games))

    def snapshot_statistics(self, tensorboard_step: int) -> int:
        return tensorboard_step

    def new_game(self, identity: GameIdentity) -> int:
        del identity
        return 0

    def build_search_request(self, game: int) -> FakeSearchRequest:
        return FakeSearchRequest(game)

    def search_active_games(
        self,
        requests: tuple[FakeSearchRequest, ...],
    ) -> tuple[FakeSearchResult, ...]:
        self.search_batches.append(requests)
        return tuple(FakeSearchResult(request.value + 1) for request in requests)

    def advance_game(
        self,
        game: int,
        request: FakeSearchRequest,
        result: FakeSearchResult,
    ) -> ContinuingGame[int]:
        assert request.value == game
        return ContinuingGame(result.value)


def test_worker_owns_pool_turns_refresh_and_statistics(tmp_path: Path) -> None:
    policy = FakeSelfPlayPolicy()
    worker = SelfPlayWorker(policy, parallel_game_count=3, worker_id=2, inbox_path=tmp_path)

    with pytest.raises(RuntimeError, match='model must be loaded'):
        worker.run_batch()

    worker.refresh_published_model(0, Path('model-0.pt'))
    worker.run_batch()
    worker.refresh_published_model(1, Path('model-1.pt'))

    assert worker.active_games == [1, 1, 1]
    assert policy.refreshed_game_counts == [0, 3]
    assert policy.search_batches == [(FakeSearchRequest(0),) * 3]
    assert worker.snapshot_statistics(17) == 17
