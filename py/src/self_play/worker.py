from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

from src.self_play.active_game import (
    ActiveGamePolicy,
    ActiveGamePool,
    ActiveGameT,
    SearchRequestT,
    SearchResultT,
)


StatisticsT = TypeVar('StatisticsT')


class GameSelfPlayPolicy(
    ActiveGamePolicy[ActiveGameT, SearchRequestT, SearchResultT],
    Generic[ActiveGameT, SearchRequestT, SearchResultT, StatisticsT],
):
    @abstractmethod
    def refresh_model(
        self,
        model_generation: int,
        model_path: Path,
        active_games: tuple[ActiveGameT, ...],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def snapshot_statistics(self, tensorboard_step: int) -> StatisticsT:
        raise NotImplementedError


class SelfPlayWorker(Generic[ActiveGameT, SearchRequestT, SearchResultT, StatisticsT]):
    """Own the game-independent lifecycle of one native self-play worker."""

    def __init__(
        self,
        policy: GameSelfPlayPolicy[
            ActiveGameT,
            SearchRequestT,
            SearchResultT,
            StatisticsT,
        ],
        parallel_game_count: int,
    ) -> None:
        if parallel_game_count <= 0:
            raise ValueError('Self-play requires at least one parallel game.')
        self.policy = policy
        self.parallel_game_count = parallel_game_count
        self._active_games: (
            ActiveGamePool[
                ActiveGameT,
                SearchRequestT,
                SearchResultT,
            ]
            | None
        ) = None

    @property
    def active_games(self) -> list[ActiveGameT]:
        if self._active_games is None:
            raise RuntimeError('A model must be loaded before self-play games are created.')
        return self._active_games.games

    def run_batch(self) -> None:
        if self._active_games is None:
            raise RuntimeError('A model must be loaded before self-play starts.')
        self._active_games.run_turn()

    def refresh_published_model(self, model_generation: int, model_path: Path) -> None:
        active_games = () if self._active_games is None else tuple(self._active_games.games)
        self.policy.refresh_model(model_generation, model_path, active_games)
        if self._active_games is None:
            self._active_games = ActiveGamePool(self.policy, self.parallel_game_count)

    def snapshot_statistics(self, tensorboard_step: int) -> StatisticsT:
        return self.policy.snapshot_statistics(tensorboard_step)
