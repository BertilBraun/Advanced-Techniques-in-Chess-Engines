from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import pytest

from src.az.games.api import GameIdentifier
from src.az.replay import storage as storage_module
from src.az.replay.envelope import ReplayRecord
from src.az.replay.storage import ReplayShardStorage
from test.unit.go_stage5_helpers import envelope


def storage(directory: Path, capacity: int = 4) -> ReplayShardStorage:
    return ReplayShardStorage(
        directory=directory,
        maximum_positions_per_shard=2,
        capacity_positions=capacity,
        game_identifier=GameIdentifier.GO,
        payload_schema_version=1,
        compression='none',
    )


def record(index: int) -> ReplayRecord:
    return ReplayRecord(envelope=envelope(index), payload=f'payload-{index}'.encode())


def test_shard_publish_read_and_metadata(tmp_path: Path) -> None:
    replay = storage(tmp_path)

    metadata = replay.publish(3, (record(1), record(2)))

    assert metadata.sequence == 3
    assert metadata.position_count == 2
    assert tuple(replay.records()) == (record(1), record(2))
    assert not tuple(tmp_path.glob('*.partial'))


def test_storage_evicts_only_complete_oldest_shards(tmp_path: Path) -> None:
    replay = storage(tmp_path, capacity=3)
    replay.publish(1, (record(1), record(2)))
    replay.publish(2, (record(3), record(4)))

    assert [metadata.sequence for metadata in replay.shards()] == [2]
    assert tuple(replay.records()) == (record(3), record(4))


def test_storage_rejects_corrupt_and_truncated_shards(tmp_path: Path) -> None:
    replay = storage(tmp_path)
    metadata = replay.publish(1, (record(1),))
    contents = metadata.path.read_bytes()
    metadata.path.write_bytes(contents[:-1] + bytes([contents[-1] ^ 1]))

    with pytest.raises(ValueError, match='checksum'):
        replay.inspect(metadata.path)
    with pytest.raises(ValueError, match='checksum'):
        replay.read(metadata.path)

    metadata.path.write_bytes(contents[:10])
    with pytest.raises(ValueError, match='truncated'):
        replay.read(metadata.path)


def test_storage_rejects_identity_schema_and_capacity_errors(tmp_path: Path) -> None:
    replay = storage(tmp_path)
    wrong_identity = record(1)
    values = wrong_identity.envelope.model_copy(update={'game_identifier': 'chess'})

    with pytest.raises(ValueError, match='game identity'):
        replay.publish(1, (ReplayRecord(values, b'x'),))
    with pytest.raises(ValueError, match='bounds'):
        replay.publish(1, ())
    with pytest.raises(ValueError, match='only explicit uncompressed'):
        ReplayShardStorage(tmp_path, 2, 4, GameIdentifier.GO, 1, 'zstd')


def test_publish_never_overwrites_existing_shard(tmp_path: Path) -> None:
    replay = storage(tmp_path)
    replay.publish(1, (record(1),))

    with pytest.raises(ValueError, match='increase strictly'):
        replay.publish(1, (record(2),))

    assert tuple(replay.records()) == (record(1),)


def test_shard_framing_is_deterministic(tmp_path: Path) -> None:
    first_directory = tmp_path / 'first'
    second_directory = tmp_path / 'second'

    first = storage(first_directory).publish(1, (record(1), record(2)))
    second = storage(second_directory).publish(1, (record(1), record(2)))

    assert first.path.read_bytes() == second.path.read_bytes()


def test_failed_atomic_publish_leaves_no_visible_or_partial_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay = storage(tmp_path)

    def fail_link(source: Path, destination: Path) -> None:
        raise OSError(f'cannot publish {source} as {destination}')

    monkeypatch.setattr(storage_module.os, 'link', fail_link)

    with pytest.raises(OSError, match='cannot publish'):
        replay.publish(1, (record(1),))

    assert not tuple(tmp_path.iterdir())


def test_read_revalidates_envelope_identity(tmp_path: Path) -> None:
    schema_two_storage = ReplayShardStorage(tmp_path, 2, 4, GameIdentifier.GO, 2, 'none')
    schema_two_envelope = envelope().model_copy(update={'payload_schema_version': 2})
    metadata = schema_two_storage.publish(1, (ReplayRecord(schema_two_envelope, b'x'),))
    go_storage = storage(tmp_path)

    with pytest.raises(ValueError, match='payload schema'):
        go_storage.read(metadata.path)


def test_restart_discovers_sequences_and_rejects_nonmonotonic_publish(tmp_path: Path) -> None:
    storage(tmp_path).publish(4, (record(1),))
    restarted = storage(tmp_path)

    restarted.publish(5, (record(2),))

    assert [metadata.sequence for metadata in restarted.shards()] == [4, 5]
    with pytest.raises(ValueError, match='increase strictly'):
        restarted.publish(3, (record(3),))


def test_capacity_discovery_does_not_rehash_old_shards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay = storage(tmp_path, capacity=4)
    replay.publish(1, (record(1),))
    replay.publish(2, (record(2),))
    checksum_calls = 0
    original = ReplayShardStorage._validate_shard_checksum

    def track_checksum(path: Path, footer_offset: int) -> None:
        nonlocal checksum_calls
        checksum_calls += 1
        original(path, footer_offset)

    monkeypatch.setattr(ReplayShardStorage, '_validate_shard_checksum', staticmethod(track_checksum))

    replay.publish(3, (record(3),))
    assert checksum_calls == 0

    assert len(tuple(replay.records())) == 3
    assert checksum_calls == 3


def test_record_write_rejects_lengths_outside_uint32(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay = storage(tmp_path)
    monkeypatch.setattr(storage_module, 'UINT32_MAXIMUM', 1)

    with pytest.raises(ValueError, match='uint32'):
        replay.publish(1, (record(1),))


def test_corrupt_record_lengths_fail_before_large_read(tmp_path: Path) -> None:
    replay = storage(tmp_path)
    metadata = replay.publish(1, (record(1),))
    contents = bytearray(metadata.path.read_bytes())
    header_offset = len(storage_module.SHARD_MAGIC)
    contents[header_offset : header_offset + storage_module.RECORD_HEADER.size] = struct.pack(
        '<II',
        2**32 - 1,
        2**32 - 1,
    )
    checksum_offset = len(contents) - storage_module.CHECKSUM_SIZE
    contents[checksum_offset:] = hashlib.sha256(contents[:checksum_offset]).digest()
    metadata.path.write_bytes(contents)

    with pytest.raises(ValueError, match='exceed the remaining'):
        replay.read(metadata.path)


def test_public_read_and_inspect_reject_paths_outside_storage(tmp_path: Path) -> None:
    replay = storage(tmp_path / 'replay')
    outside = tmp_path / 'outside.azr'
    outside.write_bytes(b'not a shard')

    with pytest.raises(ValueError, match='inside'):
        replay.read(outside)
    with pytest.raises(ValueError, match='inside'):
        replay.inspect(outside)
