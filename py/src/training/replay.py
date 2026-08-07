from __future__ import annotations

import hashlib
import os
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import struct
import sys
import time
from typing import Generic, TypeVar

import numpy as np
import numpy.typing as npt

from src.packed_planes import PackedPlanePayload
from src.training.batch import ReplaySampleMetadata, TrainingBatch
from src.self_play.completed_game import CompletedGameRecord, GameIdentity
from src.self_play.value_target import ReplayValueTarget
from src.training.replay_sampling import deterministic_rank_indices
from src.util.atomic_file import fsync_directory, write_bytes_atomically


ARCHIVE_HEADER = b'AZ-COMPLETED-GAMES\x00\x01\n'
ARCHIVE_FILE_PATTERN = 'model-generation-*.games'
_FRAME_HEADER = struct.Struct('>QQQQQQQQ32s')
_MAXIMUM_PACKED_VISIT_VALUE = int(np.iinfo(np.uint16).max)
_REVIEW_CAPACITY = 2_500_000


def pack_visits(visits: Sequence[tuple[int, int]], action_size: int) -> npt.NDArray[np.uint16]:
    if not visits:
        raise ValueError('Packed visits must not be empty.')
    if any(
        action_id < 0 or action_id >= action_size or visit_count <= 0 or visit_count > _MAXIMUM_PACKED_VISIT_VALUE
        for action_id, visit_count in visits
    ):
        raise ValueError('Packed actions or visits lie outside their uint16 ranges.')
    packed = np.asarray(visits, dtype=np.uint16)
    packed.flags.writeable = False
    return packed


@dataclass(frozen=True, eq=False)
class PackedReplaySample:
    encoded_state: PackedPlanePayload
    visits: npt.NDArray[np.uint16]
    value_target: ReplayValueTarget
    metadata: ReplaySampleMetadata
    sample_weight: float
    source_model_generation: int
    source_created_at_seconds: float

    def __post_init__(self) -> None:
        if self.visits.dtype != np.uint16 or self.visits.ndim != 2 or self.visits.shape[1] != 2:
            raise ValueError('Packed visits must have shape (N, 2) and uint16 dtype.')
        if self.visits.flags.writeable:
            raise ValueError('Packed visits must be read-only.')

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PackedReplaySample):
            return NotImplemented
        return (
            self.encoded_state == other.encoded_state
            and np.array_equal(self.visits, other.visits)
            and self.value_target == other.value_target
            and self.metadata == other.metadata
            and self.sample_weight == other.sample_weight
            and self.source_model_generation == other.source_model_generation
            and self.source_created_at_seconds == other.source_created_at_seconds
        )


@dataclass(frozen=True)
class ReplaySnapshot:
    samples: tuple[PackedReplaySample, ...]
    credited_samples: int
    credited_completed_searches: int
    sampler_seed: int
    frozen_at_seconds: float
    evicted_samples: int
    estimated_sample_bytes: int
    encoded_state_value_overhead_bytes: int
    projected_capacity_bytes: int
    projected_review_capacity_bytes: int

    def rank_indices(
        self,
        global_step: int,
        optimizer_steps: int,
        global_batch_size: int,
        world_size: int,
        rank: int,
    ) -> tuple[int, ...]:
        return deterministic_rank_indices(
            len(self.samples),
            self.sampler_seed,
            global_step,
            optimizer_steps,
            global_batch_size,
            world_size,
            rank,
        )


class ReplayPhase(str, Enum):
    INGESTING = 'ingesting'
    FROZEN = 'frozen'


@dataclass(frozen=True)
class ReplayMetrics:
    credited_samples: int
    credited_completed_searches: int
    live_samples: int
    evicted_samples: int
    oldest_source_model_generation: int | None
    newest_source_model_generation: int | None
    mean_source_model_generation: float | None
    oldest_sample_age_seconds: float | None
    mean_sample_age_seconds: float | None
    estimated_sample_bytes: int
    encoded_state_value_overhead_bytes: int
    projected_capacity_bytes: int
    projected_review_capacity_bytes: int


@dataclass(frozen=True)
class ArchiveInspection:
    path: Path
    model_generation: int
    game_count: int
    eligible_sample_count: int
    completed_searches: int
    byte_count: int


@dataclass(frozen=True)
class ArchiveFrameIndex:
    path: Path
    ingestion_sequence: int
    payload_offset: int
    payload_length: int
    payload_digest: bytes
    identity: GameIdentity
    model_generation: int
    eligible_sample_count: int
    completed_searches: int


CompletedGameT = TypeVar('CompletedGameT', bound=CompletedGameRecord)


class ReplayGameImplementation(ABC, Generic[CompletedGameT]):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def action_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def parse_file(self, path: Path) -> CompletedGameT:
        raise NotImplementedError

    @abstractmethod
    def parse_payload(self, payload: bytes) -> CompletedGameT:
        raise NotImplementedError

    @abstractmethod
    def model_generation(self, game: CompletedGameT) -> int:
        raise NotImplementedError

    @abstractmethod
    def archive_counts(self, game: CompletedGameT) -> tuple[int, int]:
        raise NotImplementedError

    @abstractmethod
    def materialize(self, game: CompletedGameT) -> tuple[PackedReplaySample, ...]:
        raise NotImplementedError

    @abstractmethod
    def build_batch(
        self,
        snapshot: ReplaySnapshot,
        sample_indices: Sequence[int],
        global_step: int,
        rank: int,
        sample_position_offset: int,
    ) -> TrainingBatch:
        raise NotImplementedError


class ReplayTrainingBatchLoader(Generic[CompletedGameT]):
    def __init__(
        self,
        implementation: ReplayGameImplementation[CompletedGameT],
        snapshot: ReplaySnapshot,
        global_step: int,
        optimizer_steps: int,
        global_batch_size: int,
        world_size: int,
        rank: int,
        pin_memory: bool,
    ) -> None:
        self.implementation = implementation
        self.snapshot = snapshot
        self.global_step = global_step
        self.optimizer_steps = optimizer_steps
        self.rank = rank
        self.local_batch_size = global_batch_size // world_size
        self.indices = snapshot.rank_indices(
            global_step,
            optimizer_steps,
            global_batch_size,
            world_size,
            rank,
        )
        self.pin_memory = pin_memory
        self.preparation_seconds = 0.0

    def __iter__(self) -> Iterator[TrainingBatch]:
        for offset in range(0, len(self.indices), self.local_batch_size):
            started_at = time.perf_counter()
            batch = self.implementation.build_batch(
                self.snapshot,
                self.indices[offset : offset + self.local_batch_size],
                self.global_step,
                self.rank,
                offset,
            )
            if self.pin_memory:
                batch = batch.pin_memory()
            self.preparation_seconds += time.perf_counter() - started_at
            yield batch

    def __len__(self) -> int:
        return self.optimizer_steps


class Replay(Generic[CompletedGameT]):
    def __init__(
        self,
        implementation: ReplayGameImplementation[CompletedGameT],
        capacity: int,
        sampler_seed: int,
    ) -> None:
        if capacity <= 0:
            raise ValueError('Replay capacity must be positive.')
        self.implementation = implementation
        self.capacity = capacity
        self.sampler_seed = sampler_seed
        self.phase = ReplayPhase.INGESTING
        self._samples: deque[PackedReplaySample] = deque()
        self._credited_samples = 0
        self._credited_completed_searches = 0
        self._evicted_samples = 0

    def begin_ingestion(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError('Replay capacity must be positive.')
        self.phase = ReplayPhase.INGESTING
        self.capacity = capacity
        self._evict_to_capacity()

    def ingest_game(self, game: CompletedGameT) -> int:
        if self.phase is not ReplayPhase.INGESTING:
            raise RuntimeError('Replay ingestion is allowed only during the ingestion phase.')
        samples = self.implementation.materialize(game)
        _, completed_searches = self.implementation.archive_counts(game)
        self._samples.extend(samples)
        self._credited_samples += len(samples)
        self._credited_completed_searches += completed_searches
        self._evict_to_capacity()
        return len(samples)

    def rebuild(
        self,
        games: Iterator[CompletedGameT],
        credited_samples: int,
        credited_completed_searches: int,
    ) -> None:
        if self.phase is not ReplayPhase.INGESTING or self._samples or self._credited_samples:
            raise RuntimeError('Replay rebuild requires a new replay in the ingestion phase.')
        for game in games:
            self._samples.extend(self.implementation.materialize(game))
            self._evict_to_capacity()
        if credited_samples < len(self._samples) or credited_completed_searches < 0:
            raise ValueError('Replay recovery totals are inconsistent with the retained samples.')
        self._credited_samples = credited_samples
        self._credited_completed_searches = credited_completed_searches
        self._evicted_samples = credited_samples - len(self._samples)

    def freeze(self) -> ReplaySnapshot:
        if self.phase is not ReplayPhase.INGESTING:
            raise RuntimeError('Replay is already frozen.')
        self.phase = ReplayPhase.FROZEN
        samples = tuple(self._samples)
        return ReplaySnapshot(
            samples=samples,
            credited_samples=self._credited_samples,
            credited_completed_searches=self._credited_completed_searches,
            sampler_seed=self.sampler_seed,
            frozen_at_seconds=time.time(),
            evicted_samples=self._evicted_samples,
            estimated_sample_bytes=sum(_estimated_sample_bytes(sample) for sample in samples),
            encoded_state_value_overhead_bytes=_encoded_state_value_overhead_bytes(samples),
            projected_capacity_bytes=_projected_capacity_bytes(samples, self.capacity),
            projected_review_capacity_bytes=_projected_capacity_bytes(samples, _REVIEW_CAPACITY),
        )

    def metrics(self, measured_at_seconds: float) -> ReplayMetrics:
        samples = tuple(self._samples)
        generations = tuple(sample.source_model_generation for sample in samples)
        ages = tuple(max(0.0, measured_at_seconds - sample.source_created_at_seconds) for sample in samples)
        return ReplayMetrics(
            credited_samples=self._credited_samples,
            credited_completed_searches=self._credited_completed_searches,
            live_samples=len(samples),
            evicted_samples=self._evicted_samples,
            oldest_source_model_generation=min(generations) if generations else None,
            newest_source_model_generation=max(generations) if generations else None,
            mean_source_model_generation=float(np.mean(generations)) if generations else None,
            oldest_sample_age_seconds=max(ages) if ages else None,
            mean_sample_age_seconds=float(np.mean(ages)) if ages else None,
            estimated_sample_bytes=sum(_estimated_sample_bytes(sample) for sample in samples),
            encoded_state_value_overhead_bytes=_encoded_state_value_overhead_bytes(samples),
            projected_capacity_bytes=_projected_capacity_bytes(samples, self.capacity),
            projected_review_capacity_bytes=_projected_capacity_bytes(samples, _REVIEW_CAPACITY),
        )

    def _evict_to_capacity(self) -> None:
        while len(self._samples) > self.capacity:
            self._samples.popleft()
            self._evicted_samples += 1


class ReplayMaintainer(Generic[CompletedGameT]):
    def __init__(
        self,
        run_path: Path,
        implementation: ReplayGameImplementation[CompletedGameT],
        capacity: int,
        sampler_seed: int,
    ) -> None:
        self.run_path = run_path
        self.inbox_path = run_path / 'completed-games' / 'inbox'
        self.archive_path = run_path / 'completed-games' / 'archive'
        self.implementation = implementation
        self.replay = Replay(implementation, capacity, sampler_seed)
        self._archived_digests: dict[GameIdentity, bytes] = {}
        self._next_ingestion_sequence = 0
        self._recover_and_rebuild()

    def maintain(self, capacity: int) -> tuple[ReplaySnapshot, ReplayMetrics]:
        self.replay.begin_ingestion(capacity)
        for inbox_file in sorted(self.inbox_path.glob('*.json')):
            game = self.implementation.parse_file(inbox_file)
            payload = canonical_game_payload(game)
            payload_digest = hashlib.sha256(payload).digest()
            archived_digest = self._archived_digests.get(game.identity)
            if archived_digest is None:
                append_archive_record(
                    self.archive_file(self.implementation.model_generation(game)),
                    payload,
                    self._next_ingestion_sequence,
                    game.identity,
                    self.implementation.model_generation(game),
                    *self.implementation.archive_counts(game),
                )
                self._archived_digests[game.identity] = payload_digest
                self.replay.ingest_game(game)
                self._next_ingestion_sequence += 1
            elif archived_digest != payload_digest:
                raise ValueError(f'Archived completed game has conflicting identity {game.identity.archive_key}.')
            inbox_file.unlink()
            fsync_directory(inbox_file.parent)
        snapshot = self.replay.freeze()
        return snapshot, self.replay.metrics(snapshot.frozen_at_seconds)

    def archive_file(self, model_generation: int) -> Path:
        return self.archive_path / f'model-generation-{model_generation:020d}.games'

    def _recover_and_rebuild(self) -> None:
        frame_indexes: list[ArchiveFrameIndex] = []
        for archive_file in sorted(self.archive_path.glob(ARCHIVE_FILE_PATTERN)):
            for frame_index in index_archive(archive_file, recover_incomplete=True):
                if archive_file != self.archive_file(frame_index.model_generation):
                    raise ValueError(f'Completed game is stored in the wrong model-generation archive: {archive_file}')
                previous_digest = self._archived_digests.setdefault(frame_index.identity, frame_index.payload_digest)
                if previous_digest != frame_index.payload_digest:
                    raise ValueError(f'Archive contains conflicting game identity {frame_index.identity.archive_key}.')
                frame_indexes.append(frame_index)
        ordered_indexes = tuple(sorted(frame_indexes, key=lambda item: item.ingestion_sequence))
        if tuple(frame.ingestion_sequence for frame in ordered_indexes) != tuple(range(len(ordered_indexes))):
            raise ValueError('Archive ingestion sequence is not contiguous.')
        retained_indexes: list[ArchiveFrameIndex] = []
        retained_sample_count = 0
        for frame_index in reversed(ordered_indexes):
            if retained_sample_count >= self.replay.capacity:
                break
            if frame_index.eligible_sample_count:
                retained_indexes.append(frame_index)
                retained_sample_count += frame_index.eligible_sample_count
        self.replay.rebuild(
            (self._read_frame(frame_index) for frame_index in reversed(retained_indexes)),
            credited_samples=sum(frame.eligible_sample_count for frame in ordered_indexes),
            credited_completed_searches=sum(frame.completed_searches for frame in ordered_indexes),
        )
        self._next_ingestion_sequence = len(frame_indexes)

    def _read_frame(self, frame_index: ArchiveFrameIndex) -> CompletedGameT:
        payload = read_frame_payload(frame_index)
        game = self.implementation.parse_payload(payload)
        eligible_sample_count, completed_searches = self.implementation.archive_counts(game)
        if (
            game.identity != frame_index.identity
            or self.implementation.model_generation(game) != frame_index.model_generation
            or eligible_sample_count != frame_index.eligible_sample_count
            or completed_searches != frame_index.completed_searches
        ):
            raise ValueError(f'Archive frame metadata disagrees with its payload: {frame_index.path}')
        return game


def canonical_game_payload(game: CompletedGameRecord) -> bytes:
    return game.model_dump_json().encode('utf-8')


def append_archive_record(
    path: Path,
    payload: bytes,
    ingestion_sequence: int,
    identity: GameIdentity,
    model_generation: int,
    eligible_sample_count: int,
    completed_searches: int,
) -> None:
    if ingestion_sequence < 0:
        raise ValueError('Archive ingestion sequence must be nonnegative.')
    if not path.exists():
        write_bytes_atomically(path, ARCHIVE_HEADER)
    index_archive(path, recover_incomplete=True)
    frame = _FRAME_HEADER.pack(
        ingestion_sequence,
        len(payload),
        model_generation,
        eligible_sample_count,
        completed_searches,
        identity.run_id,
        identity.worker_id,
        identity.game_number,
        hashlib.sha256(payload).digest(),
    )
    with path.open('ab') as archive:
        archive.write(frame + payload)
        archive.flush()
        os.fsync(archive.fileno())


def index_archive(path: Path, recover_incomplete: bool) -> tuple[ArchiveFrameIndex, ...]:
    if not path.is_file():
        raise ValueError(f'Archive does not exist: {path}')
    frame_indexes: list[ArchiveFrameIndex] = []
    mode = 'r+b' if recover_incomplete else 'rb'
    with path.open(mode) as archive:
        if archive.read(len(ARCHIVE_HEADER)) != ARCHIVE_HEADER:
            raise ValueError(f'Unsupported archive header: {path}')
        archive_size = os.fstat(archive.fileno()).st_size
        valid_end = len(ARCHIVE_HEADER)
        while True:
            frame_header = archive.read(_FRAME_HEADER.size)
            if not frame_header:
                break
            if len(frame_header) != _FRAME_HEADER.size:
                if not recover_incomplete:
                    raise ValueError(f'Archive has an incomplete final frame header: {path}')
                archive.truncate(valid_end)
                break
            (
                ingestion_sequence,
                payload_length,
                model_generation,
                eligible_sample_count,
                completed_searches,
                run_id,
                worker_id,
                game_number,
                payload_digest,
            ) = _FRAME_HEADER.unpack(frame_header)
            payload_offset = archive.tell()
            payload_end = payload_offset + payload_length
            if payload_end > archive_size:
                if not recover_incomplete:
                    raise ValueError(f'Archive has an incomplete final frame payload: {path}')
                archive.truncate(valid_end)
                break
            archive.seek(payload_end)
            frame_indexes.append(
                ArchiveFrameIndex(
                    path=path,
                    ingestion_sequence=ingestion_sequence,
                    payload_offset=payload_offset,
                    payload_length=payload_length,
                    payload_digest=payload_digest,
                    identity=GameIdentity(run_id=run_id, worker_id=worker_id, game_number=game_number),
                    model_generation=model_generation,
                    eligible_sample_count=eligible_sample_count,
                    completed_searches=completed_searches,
                )
            )
            valid_end = payload_end
        if recover_incomplete:
            archive.flush()
            os.fsync(archive.fileno())
    return tuple(frame_indexes)


def read_frame_payload(frame_index: ArchiveFrameIndex) -> bytes:
    with frame_index.path.open('rb') as archive:
        archive.seek(frame_index.payload_offset)
        payload = archive.read(frame_index.payload_length)
    if len(payload) != frame_index.payload_length or hashlib.sha256(payload).digest() != frame_index.payload_digest:
        raise ValueError(f'Archive frame checksum failed: {frame_index.path}')
    return payload


def _estimated_sample_bytes(sample: PackedReplaySample) -> int:
    return sample.encoded_state.memory_bytes() + sample.visits.nbytes + sys.getsizeof(sample)


def _encoded_state_value_overhead_bytes(samples: Sequence[PackedReplaySample]) -> int:
    if not samples:
        return 0
    payload_bytes = len(samples[0].encoded_state.payload)
    return max(0, int(np.mean(tuple(sample.encoded_state.memory_bytes() - payload_bytes for sample in samples))))


def _projected_capacity_bytes(samples: Sequence[PackedReplaySample], capacity: int) -> int:
    if capacity < 0:
        raise ValueError('Replay capacity must be nonnegative.')
    if not samples:
        return 0
    return int(round(np.mean(tuple(_estimated_sample_bytes(sample) for sample in samples)) * capacity))
