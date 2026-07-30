from __future__ import annotations

from pathlib import Path

from src.az.replay.envelope import ReplayRecord
from src.az.replay.storage import ShardMetadata
from src.az.runtime.publication import CompletedGamePublicationBuffer
from test.unit.go_stage5_helpers import envelope


def _game(first_index: int, position_count: int) -> tuple[ReplayRecord, ...]:
    return tuple(
        ReplayRecord(
            envelope=envelope(first_index + offset),
            payload=f'payload-{first_index + offset}'.encode(),
        )
        for offset in range(position_count)
    )


def test_completed_games_publish_exact_full_shards() -> None:
    published: list[tuple[ReplayRecord, ...]] = []

    def publish(records: tuple[ReplayRecord, ...]) -> ShardMetadata:
        published.append(records)
        return ShardMetadata(Path('shard-0'), 0, len(records), 1)

    buffer = CompletedGamePublicationBuffer(2, publish)

    assert buffer.add_completed_game(_game(1, 2)) is None
    result = buffer.add_completed_game(_game(3, 3))

    assert result is not None
    assert result.game_count == 2
    assert result.position_count == 5
    assert result.shard_sequence == 0
    assert not result.partial
    assert tuple(len(shard) for shard in published) == (5,)
    assert buffer.pending_game_count == 0


def test_graceful_shutdown_flushes_one_valid_partial_shard() -> None:
    published: list[tuple[ReplayRecord, ...]] = []

    def publish(records: tuple[ReplayRecord, ...]) -> ShardMetadata:
        published.append(records)
        return ShardMetadata(Path('shard-1'), 1, len(records), 1)

    buffer = CompletedGamePublicationBuffer(3, publish)
    buffer.add_completed_game(_game(10, 4))

    result = buffer.flush()

    assert result is not None
    assert result.game_count == 1
    assert result.position_count == 4
    assert result.partial
    assert tuple(len(shard) for shard in published) == (4,)
    assert buffer.flush() is None
