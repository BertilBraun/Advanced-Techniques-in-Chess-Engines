from __future__ import annotations

import hashlib
import os
import struct
import threading
import bisect
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal
from uuid import UUID

from src.az.games.api import GameIdentifier
from src.az.replay.credits import ReplayCreditJournal
from src.az.replay.envelope import ReplayEnvelope, ReplayRecord


SHARD_MAGIC = b'AZRSHRD1'
SHARD_FOOTER_MAGIC = b'AZREND01'
RECORD_HEADER = struct.Struct('<II')
SHARD_COUNT = struct.Struct('<Q')
CHECKSUM_SIZE = hashlib.sha256().digest_size
UINT32_MAXIMUM = 2**32 - 1
FOOTER_SIZE = len(SHARD_FOOTER_MAGIC) + SHARD_COUNT.size + 2 * CHECKSUM_SIZE


@dataclass(frozen=True)
class ShardMetadata:
    path: Path
    sequence: int
    position_count: int
    byte_count: int


@dataclass(frozen=True)
class ReplayRecordLocation:
    shard_sequence: int
    path: Path
    record_index: int
    byte_offset: int
    sample_id: UUID


@dataclass(frozen=True)
class IndexedReplayShard:
    metadata: ShardMetadata
    records: tuple[ReplayRecordLocation, ...]


@dataclass(frozen=True)
class ReplayCatalogSnapshot:
    shards: tuple[IndexedReplayShard, ...]
    cumulative_position_counts: tuple[int, ...]
    position_count: int

    def location(self, global_index: int) -> ReplayRecordLocation:
        if not 0 <= global_index < self.position_count:
            raise ValueError('Replay catalog index is outside the population.')
        shard_index = bisect.bisect_right(self.cumulative_position_counts, global_index)
        previous_count = 0 if shard_index == 0 else self.cumulative_position_counts[shard_index - 1]
        return self.shards[shard_index].records[global_index - previous_count]


def _canonical_envelope(envelope: ReplayEnvelope) -> bytes:
    return envelope.model_dump_json(exclude_none=False).encode('utf-8')


def _record_bytes(record: ReplayRecord) -> bytes:
    envelope = _canonical_envelope(record.envelope)
    if len(envelope) > UINT32_MAXIMUM or len(record.payload) > UINT32_MAXIMUM:
        raise ValueError('Replay record envelope and payload lengths must fit uint32 framing.')
    body = RECORD_HEADER.pack(len(envelope), len(record.payload)) + envelope + record.payload
    return body + hashlib.sha256(body).digest()


def _read_exact(stream: BinaryIO, size: int, description: str) -> bytes:
    contents = stream.read(size)
    if len(contents) != size:
        raise ValueError(f'Replay shard has a truncated {description}.')
    return contents


def _read_record(stream: BinaryIO, footer_offset: int) -> ReplayRecord:
    if footer_offset - stream.tell() < RECORD_HEADER.size + CHECKSUM_SIZE:
        raise ValueError('Replay shard has a truncated record header.')
    header = _read_exact(stream, RECORD_HEADER.size, 'record header')
    envelope_size, payload_size = RECORD_HEADER.unpack(header)
    framed_size = envelope_size + payload_size + CHECKSUM_SIZE
    if framed_size > footer_offset - stream.tell():
        raise ValueError('Replay record lengths exceed the remaining shard body.')
    envelope_bytes = _read_exact(stream, envelope_size, 'record envelope')
    payload = _read_exact(stream, payload_size, 'record payload')
    expected_checksum = _read_exact(stream, CHECKSUM_SIZE, 'record checksum')
    body = header + envelope_bytes + payload
    if hashlib.sha256(body).digest() != expected_checksum:
        raise ValueError('Replay record checksum mismatch.')
    try:
        envelope = ReplayEnvelope.model_validate_json(envelope_bytes)
    except ValueError as error:
        raise ValueError('Replay record contains an invalid envelope.') from error
    return ReplayRecord(envelope=envelope, payload=payload)


class ReplayShardStorage:
    def __init__(
        self,
        directory: Path,
        maximum_positions_per_shard: int,
        capacity_positions: int,
        game_identifier: GameIdentifier,
        payload_schema_version: int,
        compression: Literal['none', 'zstd'],
        credit_journal: ReplayCreditJournal,
    ) -> None:
        if maximum_positions_per_shard <= 0:
            raise ValueError('Maximum positions per shard must be positive.')
        if capacity_positions < maximum_positions_per_shard:
            raise ValueError('Replay capacity must hold at least one maximum-sized shard.')
        if not game_identifier:
            raise ValueError('Replay game identifier cannot be empty.')
        if payload_schema_version <= 0:
            raise ValueError('Replay payload schema version must be positive.')
        if compression != 'none':
            raise ValueError('Replay storage currently supports only explicit uncompressed shards.')
        self._directory = directory
        self._maximum_positions_per_shard = maximum_positions_per_shard
        self._capacity_positions = capacity_positions
        self._game_identifier = game_identifier
        self._payload_schema_version = payload_schema_version
        self._credit_journal = credit_journal
        self._directory.mkdir(parents=True, exist_ok=True)
        recovered = self._discover_shards()
        self._credit_visible_shards(recovered, verify_known=True)
        self._evict_over_capacity(recovered)

    @property
    def credit_journal(self) -> ReplayCreditJournal:
        return self._credit_journal

    def publish(self, sequence: int, records: tuple[ReplayRecord, ...]) -> ShardMetadata:
        if sequence < 0:
            raise ValueError('Shard sequence cannot be negative.')
        if not records or len(records) > self._maximum_positions_per_shard:
            raise ValueError('Shard position count must be within the configured bounds.')
        for record in records:
            if record.envelope.game_identifier != self._game_identifier:
                raise ValueError('Replay envelope game identity does not match the storage.')
            if record.envelope.payload_schema_version != self._payload_schema_version:
                raise ValueError('Replay envelope payload schema does not match the storage.')
        existing = list(self._discover_shards())
        self._credit_visible_shards(tuple(existing), verify_known=False)
        if existing and sequence <= existing[-1].sequence:
            raise ValueError('Replay shard sequence must increase strictly across publications.')
        credit_ids = tuple(record.envelope.replay_credit_id for record in records)
        self._credit_journal.preflight_new_shard(sequence, credit_ids)
        self._evict_over_capacity(tuple(existing))
        existing = list(self._discover_shards())
        destination = self._shard_path(sequence)
        temporary = self._directory / f'.{destination.name}.{os.getpid()}.{threading.get_ident()}.partial'
        try:
            with temporary.open('xb') as stream:
                shard_checksum = hashlib.sha256()
                stream.write(SHARD_MAGIC)
                shard_checksum.update(SHARD_MAGIC)
                for record in records:
                    encoded_record = _record_bytes(record)
                    stream.write(encoded_record)
                    shard_checksum.update(encoded_record)
                footer_body = SHARD_FOOTER_MAGIC + SHARD_COUNT.pack(len(records))
                footer_checksum = hashlib.sha256(footer_body).digest()
                stream.write(footer_body)
                stream.write(footer_checksum)
                shard_checksum.update(footer_body)
                shard_checksum.update(footer_checksum)
                stream.write(shard_checksum.digest())
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, destination)
                _sync_directory(self._directory)
            except FileExistsError as error:
                raise ValueError(f'Replay shard sequence {sequence} already exists.') from error
        finally:
            if temporary.exists():
                temporary.unlink()
        published = ShardMetadata(
            path=destination,
            sequence=sequence,
            position_count=len(records),
            byte_count=destination.stat().st_size,
        )
        self._credit_journal.credit_shard(sequence, credit_ids)
        self._evict_over_capacity((*existing, published))
        return published

    def shards(self) -> tuple[ShardMetadata, ...]:
        return self._discover_shards()

    def records(self) -> Iterator[ReplayRecord]:
        for metadata in self._discover_shards():
            yield from self._iter_records(metadata.path)

    def read(self, path: Path) -> tuple[ReplayRecord, ...]:
        return tuple(self._iter_records(path))

    def read_locations(
        self,
        locations: tuple[ReplayRecordLocation, ...],
    ) -> tuple[ReplayRecord, ...]:
        if not locations:
            raise ValueError('At least one replay record location is required.')
        grouped: dict[Path, list[tuple[int, ReplayRecordLocation]]] = {}
        for result_index, location in enumerate(locations):
            resolved = self._resolve_shard_path(location.path)
            grouped.setdefault(resolved, []).append((result_index, location))
        results: list[ReplayRecord | None] = [None] * len(locations)
        for path, requested in grouped.items():
            footer_offset = path.stat().st_size - FOOTER_SIZE
            with path.open('rb') as stream:
                for result_index, location in sorted(requested, key=lambda item: item[1].byte_offset):
                    if (
                        location.shard_sequence != self._parse_sequence(path)
                        or location.record_index < 0
                        or not len(SHARD_MAGIC) <= location.byte_offset < footer_offset
                    ):
                        raise ValueError('Replay record location is outside its indexed shard.')
                    stream.seek(location.byte_offset)
                    record = _read_record(stream, footer_offset)
                    if record.envelope.sample_id != location.sample_id:
                        raise ValueError('Indexed replay sample identity does not match shard contents.')
                    if record.envelope.game_identifier != self._game_identifier:
                        raise ValueError('Grouped replay game identity does not match the storage.')
                    if record.envelope.payload_schema_version != self._payload_schema_version:
                        raise ValueError('Grouped replay payload schema does not match the storage.')
                    results[result_index] = record
        if any(record is None for record in results):
            raise AssertionError('Grouped replay read did not populate every requested record.')
        return tuple(record for record in results if record is not None)

    def index_shard(self, metadata: ShardMetadata) -> IndexedReplayShard:
        resolved = self._resolve_shard_path(metadata.path)
        expected_count, footer_offset = self._validated_footer(resolved)
        if expected_count != metadata.position_count:
            raise ValueError('Replay shard metadata changed while it was being indexed.')
        locations: list[ReplayRecordLocation] = []
        with resolved.open('rb') as stream:
            if _read_exact(stream, len(SHARD_MAGIC), 'header') != SHARD_MAGIC:
                raise ValueError('Replay shard has an invalid header.')
            for record_index in range(expected_count):
                byte_offset = stream.tell()
                record = _read_record(stream, footer_offset)
                if record.envelope.game_identifier != self._game_identifier:
                    raise ValueError('Replay envelope game identity does not match the storage.')
                if record.envelope.payload_schema_version != self._payload_schema_version:
                    raise ValueError('Replay envelope payload schema does not match the storage.')
                locations.append(
                    ReplayRecordLocation(
                        shard_sequence=metadata.sequence,
                        path=resolved,
                        record_index=record_index,
                        byte_offset=byte_offset,
                        sample_id=record.envelope.sample_id,
                    )
                )
            if stream.tell() != footer_offset:
                raise ValueError('Replay shard record count or framing is invalid.')
        return IndexedReplayShard(metadata=metadata, records=tuple(locations))

    def _iter_records(self, path: Path) -> Iterator[ReplayRecord]:
        resolved = self._resolve_shard_path(path)
        expected_count, footer_offset = self._validated_footer(resolved)
        self._validate_position_count(expected_count)
        with resolved.open('rb') as stream:
            if _read_exact(stream, len(SHARD_MAGIC), 'header') != SHARD_MAGIC:
                raise ValueError('Replay shard has an invalid header.')
            for _ in range(expected_count):
                record = _read_record(stream, footer_offset)
                if record.envelope.game_identifier != self._game_identifier:
                    raise ValueError('Replay envelope game identity does not match the storage.')
                if record.envelope.payload_schema_version != self._payload_schema_version:
                    raise ValueError('Replay envelope payload schema does not match the storage.')
                yield record
            if stream.tell() != footer_offset:
                raise ValueError('Replay shard record count or framing is invalid.')

    def inspect(self, path: Path) -> ShardMetadata:
        resolved = self._resolve_shard_path(path)
        position_count, _ = self._validated_footer(resolved)
        self._validate_position_count(position_count)
        return ShardMetadata(
            path=resolved,
            sequence=self._parse_sequence(resolved),
            position_count=position_count,
            byte_count=resolved.stat().st_size,
        )

    @staticmethod
    def _read_footer(path: Path) -> tuple[int, int]:
        byte_count = path.stat().st_size
        if byte_count < len(SHARD_MAGIC) + FOOTER_SIZE:
            raise ValueError('Replay shard has an invalid or truncated header.')
        footer_offset = byte_count - FOOTER_SIZE
        with path.open('rb') as stream:
            if _read_exact(stream, len(SHARD_MAGIC), 'header') != SHARD_MAGIC:
                raise ValueError('Replay shard has an invalid header.')
            stream.seek(footer_offset)
            footer_magic = _read_exact(stream, len(SHARD_FOOTER_MAGIC), 'footer')
            if footer_magic != SHARD_FOOTER_MAGIC:
                raise ValueError('Replay shard has an invalid or truncated footer.')
            count_bytes = _read_exact(stream, SHARD_COUNT.size, 'position count')
            footer_checksum = _read_exact(stream, CHECKSUM_SIZE, 'footer checksum')
            if hashlib.sha256(footer_magic + count_bytes).digest() != footer_checksum:
                raise ValueError('Replay shard footer checksum mismatch.')
            position_count = SHARD_COUNT.unpack(count_bytes)[0]
        return position_count, footer_offset

    @staticmethod
    def _validate_shard_checksum(path: Path, footer_offset: int) -> None:
        with path.open('rb') as stream:
            checksum_offset = footer_offset + len(SHARD_FOOTER_MAGIC) + SHARD_COUNT.size + CHECKSUM_SIZE
            stream.seek(checksum_offset)
            expected_checksum = _read_exact(stream, CHECKSUM_SIZE, 'shard checksum')
            stream.seek(0)
            shard_checksum = hashlib.sha256()
            remaining = checksum_offset
            while remaining:
                chunk = stream.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise ValueError('Replay shard is truncated before its footer.')
                shard_checksum.update(chunk)
                remaining -= len(chunk)
        if shard_checksum.digest() != expected_checksum:
            raise ValueError('Replay shard checksum mismatch.')

    @classmethod
    def _validated_footer(cls, path: Path) -> tuple[int, int]:
        position_count, footer_offset = cls._read_footer(path)
        cls._validate_shard_checksum(path, footer_offset)
        return position_count, footer_offset

    def _evict_over_capacity(self, shards: tuple[ShardMetadata, ...]) -> None:
        metadata = list(shards)
        total = sum(shard.position_count for shard in metadata)
        deleted = False
        while total > self._capacity_positions:
            oldest = metadata.pop(0)
            oldest.path.unlink()
            total -= oldest.position_count
            deleted = True
        if deleted:
            _sync_directory(self._directory)

    def _credit_visible_shards(
        self,
        shards: tuple[ShardMetadata, ...],
        verify_known: bool,
    ) -> None:
        for shard in shards:
            if not verify_known and self._credit_journal.has_shard(shard.sequence):
                continue
            credit_ids = tuple(record.envelope.replay_credit_id for record in self._iter_records(shard.path))
            self._credit_journal.credit_shard(shard.sequence, credit_ids)

    def _discover_shards(self) -> tuple[ShardMetadata, ...]:
        discovered: list[ShardMetadata] = []
        for path in self._shard_paths():
            position_count = self._read_footer(path)[0]
            self._validate_position_count(position_count)
            discovered.append(
                ShardMetadata(
                    path=path,
                    sequence=self._parse_sequence(path),
                    position_count=position_count,
                    byte_count=path.stat().st_size,
                )
            )
        metadata = tuple(discovered)
        sequences = tuple(shard.sequence for shard in metadata)
        if sequences != tuple(sorted(set(sequences))):
            raise ValueError('Replay shard sequences must be unique and strictly increasing.')
        return metadata

    def _validate_position_count(self, position_count: int) -> None:
        if not 1 <= position_count <= self._maximum_positions_per_shard:
            raise ValueError('Replay shard position count is outside the configured bounds.')

    def _shard_paths(self) -> Iterable[Path]:
        return sorted(self._directory.glob('shard-*.azr'))

    def _shard_path(self, sequence: int) -> Path:
        return self._directory / f'shard-{sequence:020d}.azr'

    def _resolve_shard_path(self, path: Path) -> Path:
        resolved = path.resolve()
        if resolved.parent != self._directory.resolve():
            raise ValueError('Replay shard must be inside the configured directory.')
        return resolved

    @staticmethod
    def _parse_sequence(path: Path) -> int:
        try:
            sequence = int(path.stem.removeprefix('shard-'))
        except ValueError as error:
            raise ValueError(f'Invalid replay shard name: {path.name}.') from error
        if sequence < 0 or path.stem != f'shard-{sequence:020d}':
            raise ValueError(f'Invalid replay shard name: {path.name}.')
        return sequence


def _sync_directory(directory: Path) -> None:
    if os.name == 'nt':
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class IncrementalReplayCatalog:
    """Immutable snapshots backed by shards that are fully indexed exactly once."""

    def __init__(self, storage: ReplayShardStorage) -> None:
        self._storage = storage
        self._indexed: dict[int, IndexedReplayShard] = {}
        self._sample_shards: dict[UUID, int] = {}
        self._snapshot = ReplayCatalogSnapshot(
            shards=(),
            cumulative_position_counts=(),
            position_count=0,
        )

    @property
    def snapshot(self) -> ReplayCatalogSnapshot:
        return self._snapshot

    def refresh(self) -> ReplayCatalogSnapshot:
        visible = self._storage.shards()
        visible_sequences = {metadata.sequence for metadata in visible}
        for sequence in set(self._indexed) - visible_sequences:
            for location in self._indexed[sequence].records:
                del self._sample_shards[location.sample_id]
        self._indexed = {sequence: shard for sequence, shard in self._indexed.items() if sequence in visible_sequences}
        for metadata in visible:
            existing = self._indexed.get(metadata.sequence)
            if existing is not None:
                if existing.metadata != metadata:
                    raise ValueError('An immutable replay shard changed after indexing.')
                continue
            indexed = self._storage.index_shard(metadata)
            for location in indexed.records:
                if location.sample_id in self._sample_shards:
                    raise ValueError('Replay sample identities must be unique across the catalog.')
            self._indexed[metadata.sequence] = indexed
            self._sample_shards.update((location.sample_id, metadata.sequence) for location in indexed.records)
        shards = tuple(self._indexed[metadata.sequence] for metadata in visible)
        position_count = sum(len(shard.records) for shard in shards)
        cumulative: list[int] = []
        running = 0
        for shard in shards:
            running += len(shard.records)
            cumulative.append(running)
        self._snapshot = ReplayCatalogSnapshot(
            shards=shards,
            cumulative_position_counts=tuple(cumulative),
            position_count=position_count,
        )
        return self._snapshot

    def read(self, locations: tuple[ReplayRecordLocation, ...]) -> tuple[ReplayRecord, ...]:
        return self._storage.read_locations(locations)
