from __future__ import annotations

from pathlib import Path
from dataclasses import fields

import pytest

from src.az.games.api import GameIdentifier
from src.az.replay.credits import ReplayCreditJournal
from src.az.replay.envelope import ReplayRecord
from src.az.replay.storage import (
    IncrementalReplayCatalog,
    IndexedReplayShard,
    ReplayShardStorage,
    ShardMetadata,
)
from test.unit.go_stage5_helpers import envelope


def _storage(path: Path) -> ReplayShardStorage:
    return ReplayShardStorage(
        directory=path / 'shards',
        maximum_positions_per_shard=4,
        capacity_positions=12,
        game_identifier=GameIdentifier.GO,
        payload_schema_version=1,
        compression='none',
        credit_journal=ReplayCreditJournal(path / 'credits.azc'),
    )


def _record(index: int) -> ReplayRecord:
    return ReplayRecord(envelope=envelope(index), payload=f'payload-{index}'.encode())


def test_catalog_indexes_each_immutable_shard_once_and_reads_grouped_locations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _storage(tmp_path)
    storage.publish(0, (_record(1), _record(2)))
    catalog = IncrementalReplayCatalog(storage)
    indexed_sequences: list[int] = []
    original = ReplayShardStorage.index_shard

    def counted(
        storage_instance: ReplayShardStorage,
        metadata: ShardMetadata,
    ) -> IndexedReplayShard:
        indexed_sequences.append(metadata.sequence)
        return original(storage_instance, metadata)

    monkeypatch.setattr(ReplayShardStorage, 'index_shard', counted)
    first = catalog.refresh()
    catalog.refresh()
    storage.publish(1, (_record(3), _record(4)))
    second = catalog.refresh()

    assert indexed_sequences == [0, 1]
    assert first.position_count == 2
    assert second.position_count == 4
    assert tuple(field.name for field in fields(type(second))) == (
        'shards',
        'cumulative_position_counts',
        'position_count',
    )
    assert tuple(second.location(index).record_index for index in range(4)) == (
        0,
        1,
        0,
        1,
    )
    selected = (second.location(3), second.location(0), second.location(3))
    loaded = catalog.read(selected)
    assert tuple(item.envelope.sample_id for item in loaded) == tuple(location.sample_id for location in selected)


def test_grouped_read_revalidates_payload_schema_at_storage_boundary(tmp_path: Path) -> None:
    incompatible = ReplayShardStorage(
        directory=tmp_path / 'shards',
        maximum_positions_per_shard=4,
        capacity_positions=12,
        game_identifier=GameIdentifier.GO,
        payload_schema_version=2,
        compression='none',
        credit_journal=ReplayCreditJournal(tmp_path / 'credits.azc'),
    )
    original = _storage(tmp_path)
    metadata = original.publish(0, (_record(1),))
    location = original.index_shard(metadata).records[0]

    with pytest.raises(ValueError, match='payload schema'):
        incompatible.read_locations((location,))
