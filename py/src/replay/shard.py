from __future__ import annotations

import hashlib
import mmap
import os
import struct
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import BinaryIO, Literal

import numpy as np
from pydantic import Field, model_validator
from src.games.contracts import WdlTarget
from src.replay.columnar import ReplayColumnArray, ReplayColumnViews, build_column_views, flatten_column_views
from src.replay.layout import ReplayColumnDescriptor, ReplayLayout
from src.self_play.completed_game import GameIdentity, SearchObservation, TerminationReason
from src.util.atomic_file import fsync_directory, write_text_atomically
from src.util.frozen_model import FrozenModel

_SHARD_MAGIC = b'AZRSHD01'
_SHARD_SCHEMA_VERSION = 1
_SHARD_ENDIAN_MARKER = 0x0102
_SHARD_HEADER_BYTES = 4_096
_COLUMN_ALIGNMENT = 4_096
_HEADER = struct.Struct('<8sHHI64s64sQ')
_SHA256_PATTERN = r'^[0-9a-f]{64}$'
_DATA_SUFFIX = '.replay-shard.bin'
_MANIFEST_SUFFIX = '.replay-shard.json'
_PENDING_SUFFIX = '.replay-shard.pending.json'


class InboxGameOrder(FrozenModel):
    modified_at_ns: int = Field(ge=0)
    file_name: str = Field(min_length=1)

    def model_post_init(self, __context: object) -> None:
        if Path(self.file_name).name != self.file_name:
            raise ValueError('Replay inbox ordering requires a file basename.')

    @property
    def key(self) -> tuple[int, str]:
        return self.modified_at_ns, self.file_name


class ReplayShardSourceGame(FrozenModel):
    identity: GameIdentity
    order: InboxGameOrder
    source_size: int = Field(ge=0)
    source_sha256: str = Field(pattern=_SHA256_PATTERN)

    def model_post_init(self, __context: object) -> None:
        if self.order.file_name != self.identity.file_name:
            raise ValueError('Replay shard source identity does not match its ordered file name.')


class PendingReplayShardManifest(FrozenModel):
    schema_version: Literal[1] = 1
    sequence: int = Field(ge=0)
    shard_identity: str = Field(pattern=_SHA256_PATTERN)
    layout_digest: str = Field(pattern=_SHA256_PATTERN)
    games: tuple[ReplayShardSourceGame, ...] = Field(min_length=1)

    @model_validator(mode='after')
    def validate_identity_and_order(self) -> PendingReplayShardManifest:
        _validate_source_games(self.games)
        expected_identity = replay_shard_identity(self.layout_digest, self.games)
        if self.shard_identity != expected_identity:
            raise ValueError('Pending replay shard identity does not match its ordered sources.')
        return self

    @classmethod
    def create(
        cls,
        layout: ReplayLayout,
        sequence: int,
        games: tuple[ReplayShardSourceGame, ...],
    ) -> PendingReplayShardManifest:
        return cls(
            sequence=sequence,
            shard_identity=replay_shard_identity(layout.digest, games),
            layout_digest=layout.digest,
            games=games,
        )


class ReplayShardGameMetadata(FrozenModel):
    source: ReplayShardSourceGame
    row_start: int = Field(ge=0)
    row_count: int = Field(ge=0)
    length_plies: int = Field(ge=0)
    termination_reason: TerminationReason
    is_resignation_continuation: bool
    final_wdl: WdlTarget
    observations: tuple[SearchObservation, ...]
    policies_truncated: int = Field(ge=0)
    retained_visit_mass: int = Field(ge=0)
    discarded_visit_mass: int = Field(ge=0)

    def model_post_init(self, __context: object) -> None:
        plies = tuple(observation.ply for observation in self.observations)
        if plies != tuple(sorted(set(plies))) or any(ply > self.length_plies for ply in plies):
            raise ValueError('Replay shard search observations must use unique ordered game plies.')
        trailing = tuple(observation for observation in self.observations if observation.ply == self.length_plies)
        if trailing and (self.termination_reason is not TerminationReason.RESIGNATION or len(trailing) != 1):
            raise ValueError('Only resignation may retain one unplayed final search observation.')
        if trailing and trailing[0].selected_action_id is not None:
            raise ValueError('A resignation observation cannot select an action.')
        if self.termination_reason is TerminationReason.RESIGNATION and self.is_resignation_continuation:
            raise ValueError('Replay shard continuation games cannot terminate by resignation.')


class SealedReplayShardManifest(FrozenModel):
    schema_version: Literal[1] = 1
    sequence: int = Field(ge=0)
    shard_identity: str = Field(pattern=_SHA256_PATTERN)
    layout_digest: str = Field(pattern=_SHA256_PATTERN)
    data_file: str = Field(min_length=1)
    data_size: int = Field(ge=_SHARD_HEADER_BYTES)
    data_sha256: str = Field(pattern=_SHA256_PATTERN)
    row_count: int = Field(ge=0)
    games: tuple[ReplayShardGameMetadata, ...] = Field(min_length=1)

    @model_validator(mode='after')
    def validate_identity_order_and_spans(self) -> SealedReplayShardManifest:
        if Path(self.data_file).name != self.data_file:
            raise ValueError('Replay shard data file must be a basename.')
        if self.data_file != replay_shard_data_name(self.shard_identity):
            raise ValueError('Replay shard data file does not match its identity.')
        sources = tuple(game.source for game in self.games)
        _validate_source_games(sources)
        if self.shard_identity != replay_shard_identity(self.layout_digest, sources):
            raise ValueError('Sealed replay shard identity does not match its ordered sources.')
        next_row = 0
        for game in self.games:
            if game.row_start != next_row:
                raise ValueError('Replay shard game row spans must be contiguous and ordered.')
            next_row += game.row_count
        if next_row != self.row_count:
            raise ValueError('Replay shard game row spans do not cover the shard rows.')
        return self


@dataclass(frozen=True)
class ReplayShardPhysicalColumn:
    descriptor: ReplayColumnDescriptor
    offset: int
    slab_bytes: int


class ReplayShardReader:
    def __init__(
        self,
        manifest_path: Path,
        manifest: SealedReplayShardManifest,
        layout: ReplayLayout,
        file: BinaryIO,
        mapping: mmap.mmap,
        columns: ReplayColumnViews,
    ) -> None:
        self.manifest_path = manifest_path
        self.manifest = manifest
        self.layout = layout
        self._file = file
        self._mapping = mapping
        self._columns: ReplayColumnViews | None = columns
        self._closed = False

    @classmethod
    def open(cls, manifest_path: Path, layout: ReplayLayout) -> ReplayShardReader:
        try:
            manifest = SealedReplayShardManifest.model_validate_json(manifest_path.read_text(encoding='utf-8'))
        except (OSError, UnicodeError, ValueError) as error:
            raise ValueError(f'Replay shard manifest is invalid: {manifest_path}') from error
        if manifest.layout_digest != layout.digest:
            raise ValueError('Replay shard layout does not match the experiment.')
        data_path = manifest_path.parent / manifest.data_file
        if not data_path.is_file():
            raise ValueError('Sealed replay shard data file does not exist.')
        if data_path.stat().st_size != manifest.data_size:
            raise ValueError('Replay shard data size does not match its manifest.')
        if _file_sha256(data_path) != manifest.data_sha256:
            raise ValueError('Replay shard data hash does not match its manifest.')
        file = data_path.open('rb')
        mapping: mmap.mmap | None = None
        try:
            mapping = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
            _validate_header(mapping, layout, manifest.row_count)
            physical_columns = replay_shard_physical_columns(layout, manifest.row_count)
            arrays = tuple(
                ReplayColumnArray(
                    physical.descriptor,
                    np.ndarray(
                        (manifest.row_count, *physical.descriptor.trailing_shape),
                        dtype=physical.descriptor.element_type.numpy_dtype,
                        buffer=mapping,
                        offset=physical.offset,
                    ),
                )
                for physical in physical_columns
            )
            columns = build_column_views(layout, arrays)
            return cls(manifest_path, manifest, layout, file, mapping, columns)
        except BaseException:
            if mapping is not None:
                mapping.close()
            file.close()
            raise

    @property
    def columns(self) -> ReplayColumnViews:
        if self._columns is None:
            raise RuntimeError('Replay shard reader is closed.')
        return self._columns

    def close(self) -> None:
        if self._closed:
            return
        self._columns = None
        self._mapping.close()
        self._file.close()
        self._closed = True

    def __enter__(self) -> ReplayShardReader:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def replay_shard_identity(layout_digest: str, games: tuple[ReplayShardSourceGame, ...]) -> str:
    _validate_sha256(layout_digest, 'layout digest')
    digest = hashlib.sha256()
    digest.update(_encoded_digest_field(layout_digest))
    for game in games:
        digest.update(_encoded_digest_field(game.identity.archive_key))
        digest.update(_encoded_digest_field(str(game.order.modified_at_ns)))
        digest.update(_encoded_digest_field(game.order.file_name))
        digest.update(_encoded_digest_field(str(game.source_size)))
        digest.update(_encoded_digest_field(game.source_sha256))
    return digest.hexdigest()


def replay_shard_data_name(shard_identity: str) -> str:
    _validate_sha256(shard_identity, 'shard identity')
    return f'{shard_identity}{_DATA_SUFFIX}'


def replay_shard_data_path(staging_path: Path, shard_identity: str) -> Path:
    return staging_path / replay_shard_data_name(shard_identity)


def replay_shard_manifest_path(staging_path: Path, shard_identity: str) -> Path:
    _validate_sha256(shard_identity, 'shard identity')
    return staging_path / f'{shard_identity}{_MANIFEST_SUFFIX}'


def pending_replay_shard_manifest_path(staging_path: Path, shard_identity: str) -> Path:
    _validate_sha256(shard_identity, 'shard identity')
    return staging_path / f'{shard_identity}{_PENDING_SUFFIX}'


def sealed_replay_shard_manifest_paths(staging_path: Path) -> tuple[Path, ...]:
    if not staging_path.exists():
        return ()
    return tuple(sorted(staging_path.glob(f'*{_MANIFEST_SUFFIX}'), key=lambda path: path.name))


def replay_shard_physical_columns(
    layout: ReplayLayout,
    row_count: int,
) -> tuple[ReplayShardPhysicalColumn, ...]:
    if row_count < 0:
        raise ValueError('Replay shard row count must be nonnegative.')
    offset = _SHARD_HEADER_BYTES
    physical_columns = []
    for descriptor in layout.columns.columns:
        offset = _align(offset, _COLUMN_ALIGNMENT)
        slab_bytes = row_count * descriptor.row_bytes
        physical_columns.append(ReplayShardPhysicalColumn(descriptor=descriptor, offset=offset, slab_bytes=slab_bytes))
        offset += slab_bytes
    return tuple(physical_columns)


def projected_replay_shard_size(layout: ReplayLayout, row_count: int) -> int:
    physical_columns = replay_shard_physical_columns(layout, row_count)
    if not physical_columns:
        return _SHARD_HEADER_BYTES
    final = physical_columns[-1]
    return final.offset + final.slab_bytes


def write_replay_shard(
    staging_path: Path,
    layout: ReplayLayout,
    pending: PendingReplayShardManifest,
    columns: ReplayColumnViews,
    games: tuple[ReplayShardGameMetadata, ...],
) -> SealedReplayShardManifest:
    if pending.layout_digest != layout.digest:
        raise ValueError('Pending replay shard layout does not match the experiment.')
    if tuple(game.source for game in games) != pending.games:
        raise ValueError('Replay shard game metadata does not match its pending sources.')
    row_count = columns.row_count
    data_file = replay_shard_data_name(pending.shard_identity)
    data_path = replay_shard_data_path(staging_path, pending.shard_identity)
    manifest_path = replay_shard_manifest_path(staging_path, pending.shard_identity)
    data_size = projected_replay_shard_size(layout, row_count)
    provisional_manifest = SealedReplayShardManifest(
        sequence=pending.sequence,
        shard_identity=pending.shard_identity,
        layout_digest=layout.digest,
        data_file=data_file,
        data_size=data_size,
        data_sha256='0' * 64,
        row_count=row_count,
        games=games,
    )
    arrays = flatten_column_views(layout, columns)
    _validate_column_arrays(layout, arrays, row_count)
    if data_path.exists() or manifest_path.exists():
        raise ValueError('Replay shard output already exists.')
    staging_path.mkdir(parents=True, exist_ok=True)
    temporary_path = data_path.with_name(f'.{data_path.name}.{uuid.uuid4().hex}.tmp')
    try:
        _write_data_file(temporary_path, layout, arrays, row_count, data_size)
        os.replace(temporary_path, data_path)
        fsync_directory(staging_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    manifest = provisional_manifest.model_copy(update={'data_sha256': _file_sha256(data_path)})
    write_text_atomically(manifest_path, manifest.model_dump_json() + '\n')
    return manifest


def _write_data_file(
    path: Path,
    layout: ReplayLayout,
    arrays: tuple[ReplayColumnArray, ...],
    row_count: int,
    data_size: int,
) -> None:
    with path.open('x+b') as file:
        file.truncate(data_size)
        mapping = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_WRITE)
        try:
            mapping[:_SHARD_HEADER_BYTES] = bytes(_SHARD_HEADER_BYTES)
            mapping[: _HEADER.size] = _HEADER.pack(
                _SHARD_MAGIC,
                _SHARD_SCHEMA_VERSION,
                _SHARD_ENDIAN_MARKER,
                _SHARD_HEADER_BYTES,
                layout.digest.encode('ascii'),
                _descriptor_digest(layout).encode('ascii'),
                row_count,
            )
            for physical, source in zip(replay_shard_physical_columns(layout, row_count), arrays, strict=True):
                destination = np.ndarray(
                    source.values.shape,
                    dtype=source.values.dtype,
                    buffer=mapping,
                    offset=physical.offset,
                )
                destination[:] = source.values
                del destination
            mapping.flush()
        finally:
            mapping.close()
        file.flush()
        os.fsync(file.fileno())


def _validate_header(mapping: mmap.mmap, layout: ReplayLayout, row_count: int) -> None:
    if len(mapping) != projected_replay_shard_size(layout, row_count):
        raise ValueError('Replay shard data file size does not match its derived slabs.')
    magic, schema, endian, header_bytes, layout_digest, descriptor_digest, encoded_rows = _HEADER.unpack_from(mapping)
    if magic != _SHARD_MAGIC:
        raise ValueError('Replay shard magic is invalid.')
    if schema != _SHARD_SCHEMA_VERSION:
        raise ValueError('Replay shard schema version is unsupported.')
    if endian != _SHARD_ENDIAN_MARKER:
        raise ValueError('Replay shard endian marker is invalid.')
    if header_bytes != _SHARD_HEADER_BYTES:
        raise ValueError('Replay shard header width is invalid.')
    if _decode_ascii(layout_digest, 'layout digest') != layout.digest:
        raise ValueError('Replay shard header layout does not match the experiment.')
    if _decode_ascii(descriptor_digest, 'descriptor digest') != _descriptor_digest(layout):
        raise ValueError('Replay shard canonical column descriptors are invalid.')
    if encoded_rows != row_count:
        raise ValueError('Replay shard header row count does not match its manifest.')
    if any(mapping[_HEADER.size : _SHARD_HEADER_BYTES]):
        raise ValueError('Replay shard reserved header bytes are invalid.')


def _validate_column_arrays(
    layout: ReplayLayout,
    arrays: tuple[ReplayColumnArray, ...],
    row_count: int,
) -> None:
    expected = layout.columns.columns
    if len(arrays) != len(expected):
        raise ValueError('Replay shard columns do not match the canonical layout.')
    for descriptor, column in zip(expected, arrays, strict=True):
        if column.descriptor != descriptor:
            raise ValueError('Replay shard column descriptor does not match the canonical layout.')
        if column.values.shape != (row_count, *descriptor.trailing_shape):
            raise ValueError(f'Replay shard column {descriptor.key.name} has the wrong shape.')
        if column.values.dtype != descriptor.element_type.numpy_dtype:
            raise ValueError(f'Replay shard column {descriptor.key.name} has the wrong dtype.')


def _validate_source_games(games: tuple[ReplayShardSourceGame, ...]) -> None:
    identities = tuple(game.identity.archive_key for game in games)
    if len(set(identities)) != len(identities):
        raise ValueError('Replay shard source game identities must be unique.')
    orders = tuple(game.order.key for game in games)
    if orders != tuple(sorted(orders)) or len(set(orders)) != len(orders):
        raise ValueError('Replay shard source games must use unique deterministic order keys.')


def _descriptor_digest(layout: ReplayLayout) -> str:
    digest = hashlib.sha256()
    for descriptor in layout.columns.columns:
        digest.update(_encoded_digest_field(descriptor.key.name))
        digest.update(_encoded_digest_field(descriptor.element_type.value))
        digest.update(_encoded_digest_field(','.join(str(dimension) for dimension in descriptor.trailing_shape)))
    return digest.hexdigest()


def _encoded_digest_field(value: str) -> bytes:
    encoded = value.encode('utf-8')
    return len(encoded).to_bytes(8, byteorder='little') + encoded


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _validate_sha256(value: str, field: str) -> None:
    if len(value) != 64 or any(character not in '0123456789abcdef' for character in value):
        raise ValueError(f'Replay shard {field} must be a lowercase SHA-256 digest.')


def _decode_ascii(encoded: bytes, field: str) -> str:
    try:
        return encoded.rstrip(b'\x00').decode('ascii')
    except UnicodeDecodeError as error:
        raise ValueError(f'Replay shard {field} is not valid ASCII.') from error


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment
