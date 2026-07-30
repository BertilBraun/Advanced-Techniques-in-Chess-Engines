from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from src.az.replay.envelope import ReplayRecord
from src.az.replay.storage import ShardMetadata


ReplayPublisher = Callable[[tuple[ReplayRecord, ...]], ShardMetadata]


@dataclass(frozen=True)
class PublishedReplayShard:
    shard_sequence: int
    game_count: int
    position_count: int
    partial: bool


class CompletedGamePublicationBuffer:
    def __init__(
        self,
        games_per_shard: int,
        publisher: ReplayPublisher,
    ) -> None:
        if games_per_shard <= 0:
            raise ValueError('Games per replay shard must be positive.')
        self._games_per_shard = games_per_shard
        self._publisher = publisher
        self._games: list[tuple[ReplayRecord, ...]] = []

    @property
    def pending_game_count(self) -> int:
        return len(self._games)

    def add_completed_game(
        self,
        records: tuple[ReplayRecord, ...],
    ) -> PublishedReplayShard | None:
        if not records:
            raise ValueError('A completed game publication cannot be empty.')
        self._games.append(records)
        if len(self._games) == self._games_per_shard:
            return self._publish(partial=False)
        return None

    def flush(self) -> PublishedReplayShard | None:
        if not self._games:
            return None
        return self._publish(partial=True)

    def discard(self) -> int:
        game_count = len(self._games)
        self._games.clear()
        return game_count

    def _publish(self, partial: bool) -> PublishedReplayShard:
        records = tuple(record for game in self._games for record in game)
        game_count = len(self._games)
        metadata = self._publisher(records)
        if metadata.position_count != len(records):
            raise ValueError('Replay publisher metadata position count does not match the committed records.')
        self._games.clear()
        return PublishedReplayShard(
            shard_sequence=metadata.sequence,
            game_count=game_count,
            position_count=len(records),
            partial=partial,
        )
