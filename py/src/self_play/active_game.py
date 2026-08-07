from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar


ActiveGameT = TypeVar('ActiveGameT')
SearchRequestT = TypeVar('SearchRequestT')
SearchResultT = TypeVar('SearchResultT')


class ActiveGamePolicy(ABC, Generic[ActiveGameT, SearchRequestT, SearchResultT]):
    @abstractmethod
    def new_game(self) -> ActiveGameT:
        raise NotImplementedError

    @abstractmethod
    def build_search_request(self, game: ActiveGameT) -> SearchRequestT:
        raise NotImplementedError

    @abstractmethod
    def search_active_games(self, requests: tuple[SearchRequestT, ...]) -> tuple[SearchResultT, ...]:
        raise NotImplementedError

    @abstractmethod
    def advance_game(
        self,
        game: ActiveGameT,
        request: SearchRequestT,
        result: SearchResultT,
    ) -> ActiveGameT:
        raise NotImplementedError


class ActiveGamePool(Generic[ActiveGameT, SearchRequestT, SearchResultT]):
    def __init__(self, policy: ActiveGamePolicy[ActiveGameT, SearchRequestT, SearchResultT], size: int) -> None:
        if size <= 0:
            raise ValueError('Active-game pool size must be positive.')
        self.policy = policy
        self.games = [policy.new_game() for _ in range(size)]

    def run_turn(self, maximum_games: int | None = None) -> None:
        active_count = len(self.games) if maximum_games is None else min(maximum_games, len(self.games))
        if active_count <= 0:
            raise ValueError('A self-play turn must advance at least one active game.')
        selected_games = self.games[:active_count]
        requests = tuple(self.policy.build_search_request(game) for game in selected_games)
        results = self.policy.search_active_games(requests)
        if len(results) != active_count:
            raise RuntimeError('Batched self-play search returned the wrong result count.')
        advanced_games = [
            self.policy.advance_game(game, request, result)
            for game, request, result in zip(selected_games, requests, results, strict=True)
        ]
        self.games[:active_count] = advanced_games
