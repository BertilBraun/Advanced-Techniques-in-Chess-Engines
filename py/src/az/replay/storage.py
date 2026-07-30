from __future__ import annotations

import hashlib
import os
import struct
import threading
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal

from src.az.games.api import GameIdentifier
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
        self._directory.mkdir(parents=True, exist_ok=True)

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
        if existing and sequence <= existing[-1].sequence:
            raise ValueError('Replay shard sequence must increase strictly across publications.')
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
        self._evict_over_capacity((*existing, published))
        return published

    def shards(self) -> tuple[ShardMetadata, ...]:
        return self._discover_shards()

    def records(self) -> Iterator[ReplayRecord]:
        for metadata in self._discover_shards():
            yield from self._iter_records(metadata.path)

    def read(self, path: Path) -> tuple[ReplayRecord, ...]:
        return tuple(self._iter_records(path))

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
        while total > self._capacity_positions:
            oldest = metadata.pop(0)
            oldest.path.unlink()
            total -= oldest.position_count

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
