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

    def run_turn(self) -> None:
        requests = tuple(self.policy.build_search_request(game) for game in self.games)
        results = self.policy.search_active_games(requests)
        if len(results) != len(self.games):
            raise RuntimeError('Batched self-play search returned the wrong result count.')
        self.games = [
            self.policy.advance_game(game, request, result)
            for game, request, result in zip(self.games, requests, results, strict=True)
        ]
