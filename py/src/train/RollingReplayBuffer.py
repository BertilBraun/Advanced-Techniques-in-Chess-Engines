from __future__ import annotations

import hashlib
import os
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import AbstractContextManager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import TracebackType

import h5py
import numpy as np
import numpy.typing as npt
import torch
from pydantic import BaseModel, ConfigDict, Field

from src.Encoding import decode_board_states
from src.games.chess.ChessGame import BOARD_LENGTH, ChessGame, DictMove, index_to_square, square_to_index
from src.self_play.SelfPlayDataset import ReplaySampleMetadata, SelfPlayDataset, TrainingBatch
from src.self_play.value_target import (
    REPLAY_SCHEMA_VERSION,
    FinalOutcome,
    ReplayValueTarget,
    TerminationReason,
)
from src.settings import CurrentGame


DEFAULT_REPLAY_CAPACITY = 2_500_000
COMPACTION_TARGET_POSITIONS = 100_000
COMPACTION_COPY_CHUNK_ROWS = 4_096
PRESENTATION_CREDITS_PER_UNIQUE_SAMPLE = 4
ROLLING_REPLAY_INDEX_SCHEMA_VERSION = 3
COMPACTION_MANIFEST_SCHEMA_VERSION = 1

REPLAY_DATASETS = (
    'states',
    'visit_counts',
    'final_outcomes',
    'mcts_root_values',
    'outcome_target_eligible',
    'termination_reasons',
    'plies',
    'current_player_piece_counts',
    'opponent_piece_counts',
)


class TerminationCounts(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    natural: int
    resignation: int
    ply_cap: int
    material_adjudication: int
    diagnostic: int

    @classmethod
    def from_targets(cls, targets: list[ReplayValueTarget]) -> TerminationCounts:
        reasons = np.fromiter((int(target.termination_reason) for target in targets), dtype=np.uint8)
        return cls(
            natural=int(np.count_nonzero(reasons == int(TerminationReason.NATURAL))),
            resignation=int(np.count_nonzero(reasons == int(TerminationReason.RESIGNATION))),
            ply_cap=int(np.count_nonzero(reasons == int(TerminationReason.PLY_CAP))),
            material_adjudication=int(np.count_nonzero(reasons == int(TerminationReason.MATERIAL_ADJUDICATION))),
            diagnostic=int(np.count_nonzero(reasons == int(TerminationReason.DIAGNOSTIC))),
        )


class ReplayShardManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    schema_version: int
    shard_id: str
    game_count: int
    unique_sample_count: int
    producing_worker: int
    minimum_model_version: int
    maximum_model_version: int
    termination_counts: TerminationCounts
    content_sha256: str
    creation_timestamp_seconds: float
    hdf5_file_name: str

    @property
    def manifest_file_name(self) -> str:
        return f'{self.shard_id}.manifest.json'


class ReplayPayloadKind(str, Enum):
    PRODUCER_SHARD = 'producer_shard'
    COMPACTED_CONTAINER = 'compacted_container'


class LogicalReplaySegment(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    segment_id: str
    source_manifest: ReplayShardManifest
    physical_payload_id: str
    physical_offset: int
    unique_sample_count: int


class PhysicalReplayPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    payload_id: str
    kind: ReplayPayloadKind
    hdf5_file_name: str
    sidecar_file_name: str
    unique_sample_count: int
    content_sha256: str


class CompactionSourceRange(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    segment_id: str
    source_shard_id: str
    container_offset: int
    unique_sample_count: int
    source_content_sha256: str


class ReplayCompactionManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    schema_version: int
    container_id: str
    target_unique_positions: int
    unique_sample_count: int
    source_ranges: tuple[CompactionSourceRange, ...]
    content_sha256: str
    creation_timestamp_seconds: float
    hdf5_file_name: str


class ActiveCompactionPlan(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    container_id: str
    source_segment_ids: tuple[str, ...]
    total_rows: int
    temporary_hdf5_file_name: str
    final_hdf5_file_name: str
    manifest_file_name: str


class RollingReplayIndexState(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    schema_version: int
    sampler_seed: int
    credited_unique_samples: int = Field(ge=0)
    live_segments: tuple[LogicalReplaySegment, ...]
    physical_payloads: tuple[PhysicalReplayPayload, ...]
    retired_payload_ids: tuple[str, ...]
    active_compaction: ActiveCompactionPlan | None


@dataclass(frozen=True)
class ReplayIngestResult:
    unique_samples: int
    presentation_credits: int


@dataclass(frozen=True)
class GlobalSampleReference:
    segment_id: str
    sample_index: int
    physical_payload_id: str
    physical_hdf5_file_name: str
    physical_sample_index: int
    global_batch_position: int

    def __post_init__(self) -> None:
        if not self.segment_id or not self.physical_payload_id or not self.physical_hdf5_file_name:
            raise ValueError('Replay sample references require logical and physical payload identities.')
        if self.sample_index < 0 or self.physical_sample_index < 0 or self.global_batch_position < 0:
            raise ValueError('Replay sample reference indices must be nonnegative.')


@dataclass(frozen=True)
class RankSamplePartition:
    rank: int
    references: tuple[GlobalSampleReference, ...]


@dataclass(frozen=True)
class ReplayQuantumRequest:
    global_step: int
    optimizer_steps: int
    global_batch_size: int
    world_size: int
    rank: int

    @property
    def global_sample_count(self) -> int:
        return self.optimizer_steps * self.global_batch_size

    @property
    def local_batch_size(self) -> int:
        if self.global_batch_size % self.world_size:
            raise ValueError('Global batch size must divide evenly across ranks.')
        return self.global_batch_size // self.world_size


@dataclass(frozen=True)
class ReplayTrainingQuantum:
    global_step: int
    rank: int
    local_batch_size: int
    full_batch: TrainingBatch
    decode_statistics: ReplayDecodeStatistics

    def __post_init__(self) -> None:
        if self.global_step < 0 or self.rank < 0:
            raise ValueError('Replay quantum step and rank must be nonnegative.')
        if self.local_batch_size <= 0 or len(self.full_batch) % self.local_batch_size:
            raise ValueError('Replay quantum must divide evenly into local optimizer batches.')

    @property
    def optimizer_steps(self) -> int:
        return len(self.full_batch) // self.local_batch_size

    def optimizer_batches(self) -> Iterator[TrainingBatch]:
        for offset in range(0, len(self.full_batch), self.local_batch_size):
            yield self.full_batch.row_view(offset, offset + self.local_batch_size)


@dataclass(frozen=True)
class ReplayDecodeStatistics:
    payload_open_count: int
    selected_rows: int
    rows_read: int
    selected_bytes: int
    bytes_read: int

    @property
    def row_read_amplification(self) -> float:
        return self.rows_read / self.selected_rows

    @property
    def byte_read_amplification(self) -> float:
        return self.bytes_read / self.selected_bytes


class CompactionStepStatus(str, Enum):
    WAITING_FOR_MORE_SHARDS = 'waiting_for_more_shards'
    COMMITTED_CONTAINER = 'committed_container'


@dataclass(frozen=True)
class CompactionStepResult:
    status: CompactionStepStatus
    compacted_source_shards: int
    compacted_unique_positions: int
    container_id: str | None


@dataclass(frozen=True)
class _ActiveLeasePins:
    segment_ids: frozenset[str]
    payload_ids: frozenset[str]


def _atomic_write_json(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    with temporary_path.open('x', encoding='utf-8') as file:
        file.write(payload)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary_path, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _arrays_payload_bytes(arrays: Sequence[npt.NDArray[np.generic]]) -> int:
    return sum(array.nbytes for array in arrays)


def commit_replay_shard(
    dataset: SelfPlayDataset,
    replay_inbox: Path,
    producing_worker: int,
    minimum_model_version: int,
    maximum_model_version: int,
    shard_id: str | None = None,
) -> ReplayShardManifest:
    if not dataset:
        raise ValueError('Cannot commit an empty replay shard.')
    if minimum_model_version < 0 or maximum_model_version < minimum_model_version:
        raise ValueError('Replay shard model-version range is invalid.')

    resolved_shard_id = shard_id if shard_id is not None else uuid.uuid4().hex
    final_hdf5_path = replay_inbox / f'{resolved_shard_id}.hdf5'
    manifest_path = replay_inbox / f'{resolved_shard_id}.manifest.json'
    if final_hdf5_path.exists() or manifest_path.exists():
        raise ValueError(f'Replay shard {resolved_shard_id} already exists.')

    replay_inbox.mkdir(parents=True, exist_ok=True)
    if not dataset.save_to_path(final_hdf5_path):
        raise RuntimeError(f'Failed to write replay shard {resolved_shard_id}.')

    with h5py.File(final_hdf5_path, 'r') as file:
        SelfPlayDataset._require_current_schema(file, final_hdf5_path)
        unique_sample_count = int(file['states'].shape[0])
    if unique_sample_count != len(dataset):
        final_hdf5_path.unlink()
        raise RuntimeError('Committed replay shard sample count does not match its source dataset.')

    manifest = ReplayShardManifest(
        schema_version=REPLAY_SCHEMA_VERSION,
        shard_id=resolved_shard_id,
        game_count=dataset.stats.num_games,
        unique_sample_count=unique_sample_count,
        producing_worker=producing_worker,
        minimum_model_version=minimum_model_version,
        maximum_model_version=maximum_model_version,
        termination_counts=TerminationCounts.from_targets(dataset.value_targets),
        content_sha256=file_sha256(final_hdf5_path),
        creation_timestamp_seconds=time.time(),
        hdf5_file_name=final_hdf5_path.name,
    )
    _atomic_write_json(manifest_path, manifest.model_dump_json(indent=2))
    return manifest


class ReplayBatchLease(AbstractContextManager['ReplayBatchLease']):
    def __init__(
        self,
        replay_buffer: RollingReplayBuffer,
        lease_id: str,
        partitions: tuple[RankSamplePartition, ...],
    ) -> None:
        self._replay_buffer = replay_buffer
        self.lease_id = lease_id
        self.partitions = partitions
        self._released = False

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()

    def release(self) -> None:
        if self._released:
            return
        self._replay_buffer.release_lease(self.lease_id)
        self._released = True


class RollingReplayBuffer:
    """Disk replay owned by one writer and refreshed by phase-separated read ranks.

    Cross-process leases are intentionally not persisted. The scheduler must finish
    every rank quantum before the writer ingests, evicts, or compacts payloads.
    """

    def __init__(
        self,
        replay_inbox: Path,
        index_path: Path,
        capacity: int = DEFAULT_REPLAY_CAPACITY,
        sampler_seed: int = 0,
        read_only: bool = False,
    ) -> None:
        if capacity <= 0:
            raise ValueError('Replay capacity must be positive.')
        self.replay_inbox = replay_inbox
        self.index_path = index_path
        self.capacity = capacity
        self.read_only = read_only
        self._lock = threading.RLock()
        self._active_leases: dict[str, _ActiveLeasePins] = {}
        self._compaction_reserved_payload_ids: set[str] = set()
        self._state = self._load_or_create_state(sampler_seed)
        self._prefix = np.asarray([0], dtype=np.int64)
        self._segments_by_id: dict[str, LogicalReplaySegment] = {}
        self._payloads_by_id: dict[str, PhysicalReplayPayload] = {}
        self._last_decode_statistics = ReplayDecodeStatistics(
            payload_open_count=0,
            selected_rows=0,
            rows_read=0,
            selected_bytes=0,
            bytes_read=0,
        )
        self._rebuild_lookup()
        if not self.read_only:
            self._recover_interrupted_compaction()
            self._finish_retired_payloads()
            self._remove_orphan_compaction_artifacts()
            self._evict_outside_capacity()

    @property
    def unique_sample_count(self) -> int:
        return int(self._prefix[-1])

    @property
    def credited_unique_sample_count(self) -> int:
        """Return the durable number of positions that have ever earned replay credit."""
        return self._state.credited_unique_samples

    @property
    def shard_count(self) -> int:
        return len(self._state.live_segments)

    @property
    def physical_payload_count(self) -> int:
        return len(self._state.physical_payloads)

    @property
    def compacted_container_count(self) -> int:
        return sum(payload.kind is ReplayPayloadKind.COMPACTED_CONTAINER for payload in self._state.physical_payloads)

    @property
    def last_decode_payload_open_count(self) -> int:
        return self._last_decode_statistics.payload_open_count

    @property
    def last_decode_statistics(self) -> ReplayDecodeStatistics:
        return self._last_decode_statistics

    @property
    def metadata_memory_bytes(self) -> int:
        return len(self._state.model_dump_json().encode('utf-8')) + self._prefix.nbytes

    def refresh_index_for_read(self) -> None:
        """Refresh an idle read rank after the writer completes a mutation phase."""
        with self._lock:
            if not self.read_only:
                raise ValueError('Only read-only replay ranks refresh from the writer index.')
            if self._active_leases:
                raise RuntimeError('A read rank cannot refresh its replay index during an active quantum.')
            self._state = self._load_existing_state()
            self._rebuild_lookup()

    def discover_committed_shards(self) -> ReplayIngestResult:
        with self._lock:
            self._require_writer()
            known_ids = {segment.source_manifest.shard_id for segment in self._state.live_segments}
            discovered: list[ReplayShardManifest] = []
            for manifest_path in sorted(self.replay_inbox.glob('*.manifest.json')):
                manifest = ReplayShardManifest.model_validate_json(manifest_path.read_text(encoding='utf-8'))
                if manifest.shard_id in known_ids:
                    continue
                self._validate_source_manifest(manifest)
                discovered.append(manifest)
                known_ids.add(manifest.shard_id)

            if not discovered:
                return ReplayIngestResult(unique_samples=0, presentation_credits=0)

            new_segments = tuple(
                LogicalReplaySegment(
                    segment_id=manifest.shard_id,
                    source_manifest=manifest,
                    physical_payload_id=manifest.shard_id,
                    physical_offset=0,
                    unique_sample_count=manifest.unique_sample_count,
                )
                for manifest in discovered
            )
            new_payloads = tuple(
                PhysicalReplayPayload(
                    payload_id=manifest.shard_id,
                    kind=ReplayPayloadKind.PRODUCER_SHARD,
                    hdf5_file_name=manifest.hdf5_file_name,
                    sidecar_file_name=manifest.manifest_file_name,
                    unique_sample_count=manifest.unique_sample_count,
                    content_sha256=manifest.content_sha256,
                )
                for manifest in discovered
            )
            live_segments = tuple(
                sorted(
                    (*self._state.live_segments, *new_segments),
                    key=lambda segment: (
                        segment.source_manifest.creation_timestamp_seconds,
                        segment.segment_id,
                    ),
                )
            )
            self._persist_state(
                self._state_with(
                    credited_unique_samples=self._state.credited_unique_samples
                    + sum(manifest.unique_sample_count for manifest in discovered),
                    live_segments=live_segments,
                    physical_payloads=(*self._state.physical_payloads, *new_payloads),
                )
            )
            self._evict_outside_capacity()
            unique_samples = sum(manifest.unique_sample_count for manifest in discovered)
            return ReplayIngestResult(
                unique_samples=unique_samples,
                presentation_credits=unique_samples * PRESENTATION_CREDITS_PER_UNIQUE_SAMPLE,
            )

    def compact_one_idle_container(self) -> CompactionStepResult:
        with self._lock:
            self._require_writer()
            if self._state.active_compaction is not None:
                raise RuntimeError('A replay compaction is already active.')
            source_segments = self._select_compaction_sources()
            if not source_segments:
                return CompactionStepResult(
                    status=CompactionStepStatus.WAITING_FOR_MORE_SHARDS,
                    compacted_source_shards=0,
                    compacted_unique_positions=0,
                    container_id=None,
                )
            plan = self._create_compaction_plan(source_segments)
            source_payload_ids = {segment.physical_payload_id for segment in source_segments}
            self._compaction_reserved_payload_ids.update(source_payload_ids)
            self._persist_state(
                self._state_with(
                    active_compaction=plan,
                    preserve_active_compaction=False,
                )
            )

        try:
            manifest = self._stream_compaction(plan, source_segments)
            with self._lock:
                self._commit_compaction(plan, manifest)
                self._compaction_reserved_payload_ids.difference_update(source_payload_ids)
                self._finish_retired_payloads()
            return CompactionStepResult(
                status=CompactionStepStatus.COMMITTED_CONTAINER,
                compacted_source_shards=len(source_segments),
                compacted_unique_positions=plan.total_rows,
                container_id=plan.container_id,
            )
        except Exception:
            with self._lock:
                self._compaction_reserved_payload_ids.difference_update(source_payload_ids)
                self._abort_active_compaction(plan)
            raise

    def lease_quantum(
        self,
        global_step: int,
        global_sample_count: int,
        world_size: int,
        global_batch_size: int | None = None,
    ) -> ReplayBatchLease:
        if global_step < 0:
            raise ValueError('Global step must be nonnegative.')
        if global_sample_count <= 0:
            raise ValueError('Global sample count must be positive.')
        if world_size <= 0 or global_sample_count % world_size:
            raise ValueError('Global sample count must divide evenly across ranks.')
        resolved_batch_size = global_sample_count if global_batch_size is None else global_batch_size
        if resolved_batch_size <= 0 or resolved_batch_size % world_size or global_sample_count % resolved_batch_size:
            raise ValueError('Global batch size must divide the quantum and evenly partition across ranks.')

        with self._lock:
            if self.unique_sample_count == 0:
                raise ValueError('Cannot sample from an empty replay buffer.')
            generator = np.random.default_rng(
                np.random.SeedSequence((self._state.sampler_seed, global_step, global_sample_count))
            )
            if self.unique_sample_count >= global_sample_count:
                global_indices = generator.choice(
                    self.unique_sample_count,
                    size=global_sample_count,
                    replace=False,
                )
            elif self.unique_sample_count >= resolved_batch_size:
                global_indices = np.concatenate(
                    tuple(
                        generator.choice(
                            self.unique_sample_count,
                            size=resolved_batch_size,
                            replace=False,
                        )
                        for _ in range(global_sample_count // resolved_batch_size)
                    )
                )
            else:
                raise ValueError(
                    f'Live replay has {self.unique_sample_count} positions but a duplicate-free '
                    f'global batch requires {resolved_batch_size}.'
                )
            references = tuple(
                self._reference_for_index(int(global_index), position)
                for position, global_index in enumerate(global_indices)
            )
            partitions = tuple(
                RankSamplePartition(rank=rank, references=references[rank::world_size]) for rank in range(world_size)
            )
            lease_id = uuid.uuid4().hex
            self._active_leases[lease_id] = _ActiveLeasePins(
                segment_ids=frozenset(reference.segment_id for reference in references),
                payload_ids=frozenset(reference.physical_payload_id for reference in references),
            )
            return ReplayBatchLease(self, lease_id, partitions)

    def release_lease(self, lease_id: str) -> None:
        with self._lock:
            if lease_id not in self._active_leases:
                raise ValueError(f'Unknown replay lease {lease_id}.')
            del self._active_leases[lease_id]
            if self.read_only:
                return
            self._finish_retired_payloads()
            self._evict_outside_capacity()

    def decode_partition(
        self,
        partition: RankSamplePartition,
        global_step: int,
    ) -> TrainingBatch:
        if not partition.references:
            raise ValueError('Cannot decode an empty rank partition.')
        grouped: dict[tuple[str, str], list[tuple[int, GlobalSampleReference]]] = defaultdict(list)
        for output_position, reference in enumerate(partition.references):
            grouped[(reference.physical_payload_id, reference.physical_hdf5_file_name)].append(
                (output_position, reference)
            )

        encoded_states: list[bytes | None] = [None] * len(partition.references)
        visit_counts: list[npt.NDArray[np.uint16] | None] = [None] * len(partition.references)
        value_targets: list[ReplayValueTarget | None] = [None] * len(partition.references)
        sample_metadata: list[ReplaySampleMetadata | None] = [None] * len(partition.references)

        payload_open_count = 0
        rows_read = 0
        selected_bytes = 0
        bytes_read = 0
        for (_, hdf5_file_name), output_references in grouped.items():
            sorted_references = sorted(
                output_references,
                key=lambda item: item[1].physical_sample_index,
            )
            unique_indices = np.asarray(
                sorted({reference.physical_sample_index for _, reference in sorted_references}),
                dtype=np.int64,
            )
            first_index = int(unique_indices[0])
            stop_index = int(unique_indices[-1]) + 1
            selection = unique_indices - first_index
            rows_read += stop_index - first_index
            payload_path = self.replay_inbox / hdf5_file_name
            if not payload_path.exists():
                raise RuntimeError(f'Stale replay reference points to deleted payload {hdf5_file_name}.')
            with h5py.File(payload_path, 'r') as file:
                if stop_index > int(file['states'].shape[0]):
                    raise RuntimeError(f'Stale replay reference exceeds payload {hdf5_file_name}.')
                payload_open_count += 1
                read_states = np.asarray(file['states'][first_index:stop_index])
                read_visits = np.asarray(file['visit_counts'][first_index:stop_index])
                read_outcomes = np.asarray(
                    file['final_outcomes'][first_index:stop_index],
                    dtype=np.uint8,
                )
                read_root_values = np.asarray(
                    file['mcts_root_values'][first_index:stop_index],
                    dtype=np.float32,
                )
                read_eligibility = np.asarray(
                    file['outcome_target_eligible'][first_index:stop_index],
                    dtype=np.bool_,
                )
                read_reasons = np.asarray(
                    file['termination_reasons'][first_index:stop_index],
                    dtype=np.uint8,
                )
                read_plies = np.asarray(
                    file['plies'][first_index:stop_index],
                    dtype=np.int32,
                )
                read_current_counts = np.asarray(
                    file['current_player_piece_counts'][first_index:stop_index],
                    dtype=np.uint8,
                )
                read_opponent_counts = np.asarray(
                    file['opponent_piece_counts'][first_index:stop_index],
                    dtype=np.uint8,
                )

            read_arrays = (
                read_states,
                read_visits,
                read_outcomes,
                read_root_values,
                read_eligibility,
                read_reasons,
                read_plies,
                read_current_counts,
                read_opponent_counts,
            )
            bytes_read += _arrays_payload_bytes(read_arrays)
            states = read_states[selection]
            visits = read_visits[selection]
            outcomes = read_outcomes[selection]
            root_values = read_root_values[selection]
            eligibility = read_eligibility[selection]
            reasons = read_reasons[selection]
            plies = read_plies[selection]
            current_counts = read_current_counts[selection]
            opponent_counts = read_opponent_counts[selection]
            selected_bytes += _arrays_payload_bytes(
                (
                    states,
                    visits,
                    outcomes,
                    root_values,
                    eligibility,
                    reasons,
                    plies,
                    current_counts,
                    opponent_counts,
                )
            )

            source_positions = {
                int(sample_index): source_position for source_position, sample_index in enumerate(unique_indices)
            }
            for output_position, reference in sorted_references:
                source_position = source_positions[reference.physical_sample_index]
                encoded_states[output_position] = bytes(states[source_position])
                visit_counts[output_position] = visits[source_position][visits[source_position, :, 1] > 0].astype(
                    np.uint16, copy=False
                )
                value_targets[output_position] = ReplayValueTarget(
                    final_outcome=FinalOutcome(int(outcomes[source_position])),
                    mcts_root_value=float(root_values[source_position]),
                    termination_reason=TerminationReason(int(reasons[source_position])),
                    outcome_target_eligible=bool(eligibility[source_position]),
                )
                sample_metadata[output_position] = ReplaySampleMetadata(
                    ply=int(plies[source_position]),
                    current_player_piece_count=int(current_counts[source_position]),
                    opponent_piece_count=int(opponent_counts[source_position]),
                )

        self._last_decode_statistics = ReplayDecodeStatistics(
            payload_open_count=payload_open_count,
            selected_rows=len(partition.references),
            rows_read=rows_read,
            selected_bytes=selected_bytes,
            bytes_read=bytes_read,
        )
        complete_states = [state for state in encoded_states if state is not None]
        complete_visits = [visits for visits in visit_counts if visits is not None]
        complete_targets = [target for target in value_targets if target is not None]
        complete_metadata = [metadata for metadata in sample_metadata if metadata is not None]
        if not all(
            len(values) == len(partition.references)
            for values in (
                complete_states,
                complete_visits,
                complete_targets,
                complete_metadata,
            )
        ):
            raise RuntimeError('Replay decode did not populate every requested sample.')
        return _decode_with_deterministic_symmetry(
            complete_states,
            complete_visits,
            complete_targets,
            complete_metadata,
            self._state.sampler_seed,
            global_step,
            partition,
        )

    def _load_or_create_state(self, sampler_seed: int) -> RollingReplayIndexState:
        if self.index_path.exists():
            return self._load_existing_state()
        if self.read_only:
            raise ValueError(f'Replay index {self.index_path} does not exist for a read-only rank.')
        state = RollingReplayIndexState(
            schema_version=ROLLING_REPLAY_INDEX_SCHEMA_VERSION,
            sampler_seed=sampler_seed,
            credited_unique_samples=0,
            live_segments=(),
            physical_payloads=(),
            retired_payload_ids=(),
            active_compaction=None,
        )
        _atomic_write_json(self.index_path, state.model_dump_json(indent=2))
        return state

    def _load_existing_state(self) -> RollingReplayIndexState:
        state = RollingReplayIndexState.model_validate_json(self.index_path.read_text(encoding='utf-8'))
        if state.schema_version != ROLLING_REPLAY_INDEX_SCHEMA_VERSION:
            raise ValueError(
                f'Replay index schema {state.schema_version} is unsupported; '
                f'expected {ROLLING_REPLAY_INDEX_SCHEMA_VERSION}.'
            )
        return state

    def _require_writer(self) -> None:
        if self.read_only:
            raise RuntimeError('Read-only replay ranks cannot ingest, evict, or compact payloads.')

    def _state_with(
        self,
        credited_unique_samples: int | None = None,
        live_segments: tuple[LogicalReplaySegment, ...] | None = None,
        physical_payloads: tuple[PhysicalReplayPayload, ...] | None = None,
        retired_payload_ids: tuple[str, ...] | None = None,
        active_compaction: ActiveCompactionPlan | None = None,
        preserve_active_compaction: bool = True,
    ) -> RollingReplayIndexState:
        resolved_compaction = self._state.active_compaction if preserve_active_compaction else active_compaction
        return RollingReplayIndexState(
            schema_version=ROLLING_REPLAY_INDEX_SCHEMA_VERSION,
            sampler_seed=self._state.sampler_seed,
            credited_unique_samples=(
                self._state.credited_unique_samples if credited_unique_samples is None else credited_unique_samples
            ),
            live_segments=self._state.live_segments if live_segments is None else live_segments,
            physical_payloads=(self._state.physical_payloads if physical_payloads is None else physical_payloads),
            retired_payload_ids=(
                self._state.retired_payload_ids if retired_payload_ids is None else retired_payload_ids
            ),
            active_compaction=resolved_compaction,
        )

    def _persist_state(self, state: RollingReplayIndexState) -> None:
        _atomic_write_json(self.index_path, state.model_dump_json(indent=2))
        self._state = state
        self._rebuild_lookup()

    def _rebuild_lookup(self) -> None:
        self._segments_by_id = {segment.segment_id: segment for segment in self._state.live_segments}
        self._payloads_by_id = {payload.payload_id: payload for payload in self._state.physical_payloads}
        lengths = np.fromiter(
            (segment.unique_sample_count for segment in self._state.live_segments),
            dtype=np.int64,
            count=len(self._state.live_segments),
        )
        self._prefix = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(lengths)))

    def _validate_source_manifest(self, manifest: ReplayShardManifest) -> None:
        if manifest.schema_version != REPLAY_SCHEMA_VERSION:
            raise ValueError(
                f'Shard {manifest.shard_id} uses schema {manifest.schema_version}; expected {REPLAY_SCHEMA_VERSION}.'
            )
        hdf5_path = self.replay_inbox / manifest.hdf5_file_name
        if not hdf5_path.exists():
            raise ValueError(f'Manifest {manifest.shard_id} references missing HDF5 payload.')
        if file_sha256(hdf5_path) != manifest.content_sha256:
            raise ValueError(f'Shard {manifest.shard_id} content hash does not match its manifest.')
        with h5py.File(hdf5_path, 'r') as file:
            if int(file['states'].shape[0]) != manifest.unique_sample_count:
                raise ValueError(f'Shard {manifest.shard_id} sample count does not match its manifest.')

    def _reference_for_index(
        self,
        global_index: int,
        global_batch_position: int,
    ) -> GlobalSampleReference:
        segment_index = int(np.searchsorted(self._prefix, global_index + 1) - 1)
        segment = self._state.live_segments[segment_index]
        payload = self._payloads_by_id[segment.physical_payload_id]
        sample_index = global_index - int(self._prefix[segment_index])
        return GlobalSampleReference(
            segment_id=segment.segment_id,
            sample_index=sample_index,
            physical_payload_id=payload.payload_id,
            physical_hdf5_file_name=payload.hdf5_file_name,
            physical_sample_index=segment.physical_offset + sample_index,
            global_batch_position=global_batch_position,
        )

    def _select_compaction_sources(self) -> tuple[LogicalReplaySegment, ...]:
        selected: list[LogicalReplaySegment] = []
        selected_count = 0
        for segment in self._state.live_segments:
            payload = self._payloads_by_id[segment.physical_payload_id]
            if payload.kind is not ReplayPayloadKind.PRODUCER_SHARD:
                selected = []
                selected_count = 0
                continue
            selected.append(segment)
            selected_count += segment.unique_sample_count
            if selected_count >= COMPACTION_TARGET_POSITIONS:
                return tuple(selected)
        return ()

    def _create_compaction_plan(
        self,
        source_segments: tuple[LogicalReplaySegment, ...],
    ) -> ActiveCompactionPlan:
        container_id = f'container-{uuid.uuid4().hex}'
        return ActiveCompactionPlan(
            container_id=container_id,
            source_segment_ids=tuple(segment.segment_id for segment in source_segments),
            total_rows=sum(segment.unique_sample_count for segment in source_segments),
            temporary_hdf5_file_name=f'tmp/.{container_id}.hdf5.tmp',
            final_hdf5_file_name=f'containers/{container_id}.hdf5',
            manifest_file_name=f'containers/{container_id}.container.json',
        )

    def _stream_compaction(
        self,
        plan: ActiveCompactionPlan,
        source_segments: tuple[LogicalReplaySegment, ...],
    ) -> ReplayCompactionManifest:
        temporary_path = self.replay_inbox / plan.temporary_hdf5_file_name
        final_path = self.replay_inbox / plan.final_hdf5_file_name
        manifest_path = self.replay_inbox / plan.manifest_file_name
        temporary_path.parent.mkdir(parents=True, exist_ok=True)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        if temporary_path.exists() or final_path.exists() or manifest_path.exists():
            raise RuntimeError(f'Compaction artifacts for {plan.container_id} already exist.')

        source_payloads = tuple(self._payloads_by_id[segment.physical_payload_id] for segment in source_segments)
        max_visit_width = 0
        first_source_path = self.replay_inbox / source_payloads[0].hdf5_file_name
        with h5py.File(first_source_path, 'r') as first_source:
            dataset_dtypes = {dataset_name: first_source[dataset_name].dtype for dataset_name in REPLAY_DATASETS}
        for payload in source_payloads:
            with h5py.File(self.replay_inbox / payload.hdf5_file_name, 'r') as source:
                max_visit_width = max(max_visit_width, int(source['visit_counts'].shape[1]))

        with h5py.File(temporary_path, 'w') as destination:
            destination.create_dataset(
                'states',
                shape=(plan.total_rows,),
                dtype=dataset_dtypes['states'],
                chunks=True,
            )
            destination.create_dataset(
                'visit_counts',
                shape=(plan.total_rows, max_visit_width, 2),
                dtype=dataset_dtypes['visit_counts'],
                chunks=True,
                fillvalue=0,
            )
            for dataset_name in REPLAY_DATASETS[2:]:
                destination.create_dataset(
                    dataset_name,
                    shape=(plan.total_rows,),
                    dtype=dataset_dtypes[dataset_name],
                    chunks=True,
                )

            destination_offset = 0
            for segment, payload in zip(source_segments, source_payloads):
                source_path = self.replay_inbox / payload.hdf5_file_name
                with h5py.File(source_path, 'r') as source:
                    for source_offset in range(
                        0,
                        segment.unique_sample_count,
                        COMPACTION_COPY_CHUNK_ROWS,
                    ):
                        row_count = min(
                            COMPACTION_COPY_CHUNK_ROWS,
                            segment.unique_sample_count - source_offset,
                        )
                        source_rows = slice(
                            segment.physical_offset + source_offset,
                            segment.physical_offset + source_offset + row_count,
                        )
                        destination_rows = slice(
                            destination_offset + source_offset,
                            destination_offset + source_offset + row_count,
                        )
                        destination['states'][destination_rows] = source['states'][source_rows]
                        source_visits = source['visit_counts'][source_rows]
                        destination['visit_counts'][
                            destination_rows,
                            : source_visits.shape[1],
                            :,
                        ] = source_visits
                        for dataset_name in REPLAY_DATASETS[2:]:
                            destination[dataset_name][destination_rows] = source[dataset_name][source_rows]
                destination_offset += segment.unique_sample_count

            destination.attrs['replay_schema_version'] = REPLAY_SCHEMA_VERSION
            destination.attrs['metadata'] = str(SelfPlayDataset._get_current_metadata())
            destination.flush()

        with temporary_path.open('r+b') as file:
            os.fsync(file.fileno())
        content_hash = file_sha256(temporary_path)
        os.replace(temporary_path, final_path)
        source_ranges: list[CompactionSourceRange] = []
        offset = 0
        for segment in source_segments:
            source_ranges.append(
                CompactionSourceRange(
                    segment_id=segment.segment_id,
                    source_shard_id=segment.source_manifest.shard_id,
                    container_offset=offset,
                    unique_sample_count=segment.unique_sample_count,
                    source_content_sha256=segment.source_manifest.content_sha256,
                )
            )
            offset += segment.unique_sample_count
        manifest = ReplayCompactionManifest(
            schema_version=COMPACTION_MANIFEST_SCHEMA_VERSION,
            container_id=plan.container_id,
            target_unique_positions=COMPACTION_TARGET_POSITIONS,
            unique_sample_count=plan.total_rows,
            source_ranges=tuple(source_ranges),
            content_sha256=content_hash,
            creation_timestamp_seconds=time.time(),
            hdf5_file_name=plan.final_hdf5_file_name,
        )
        _atomic_write_json(manifest_path, manifest.model_dump_json(indent=2))
        return manifest

    def _commit_compaction(
        self,
        plan: ActiveCompactionPlan,
        manifest: ReplayCompactionManifest,
    ) -> None:
        if self._state.active_compaction != plan:
            raise RuntimeError('Replay compaction plan changed before commit.')
        source_ids = set(plan.source_segment_ids)
        source_segments = tuple(segment for segment in self._state.live_segments if segment.segment_id in source_ids)
        if tuple(segment.segment_id for segment in source_segments) != plan.source_segment_ids:
            raise RuntimeError('Replay compaction source order changed before commit.')

        offsets = {source_range.segment_id: source_range.container_offset for source_range in manifest.source_ranges}
        remapped_segments = tuple(
            (
                LogicalReplaySegment(
                    segment_id=segment.segment_id,
                    source_manifest=segment.source_manifest,
                    physical_payload_id=manifest.container_id,
                    physical_offset=offsets[segment.segment_id],
                    unique_sample_count=segment.unique_sample_count,
                )
                if segment.segment_id in source_ids
                else segment
            )
            for segment in self._state.live_segments
        )
        source_payload_ids = {segment.physical_payload_id for segment in source_segments}
        container_payload = PhysicalReplayPayload(
            payload_id=manifest.container_id,
            kind=ReplayPayloadKind.COMPACTED_CONTAINER,
            hdf5_file_name=manifest.hdf5_file_name,
            sidecar_file_name=plan.manifest_file_name,
            unique_sample_count=manifest.unique_sample_count,
            content_sha256=manifest.content_sha256,
        )
        self._persist_state(
            self._state_with(
                live_segments=remapped_segments,
                physical_payloads=(*self._state.physical_payloads, container_payload),
                retired_payload_ids=tuple(dict.fromkeys((*self._state.retired_payload_ids, *source_payload_ids))),
                active_compaction=None,
                preserve_active_compaction=False,
            )
        )

    def _abort_active_compaction(self, plan: ActiveCompactionPlan) -> None:
        self._delete_compaction_artifacts(plan)
        if self._state.active_compaction == plan:
            self._persist_state(
                self._state_with(
                    active_compaction=None,
                    preserve_active_compaction=False,
                )
            )

    def _recover_interrupted_compaction(self) -> None:
        plan = self._state.active_compaction
        if plan is None:
            return
        self._delete_compaction_artifacts(plan)
        self._persist_state(
            self._state_with(
                active_compaction=None,
                preserve_active_compaction=False,
            )
        )

    def _delete_compaction_artifacts(self, plan: ActiveCompactionPlan) -> None:
        (self.replay_inbox / plan.temporary_hdf5_file_name).unlink(missing_ok=True)
        (self.replay_inbox / plan.final_hdf5_file_name).unlink(missing_ok=True)
        (self.replay_inbox / plan.manifest_file_name).unlink(missing_ok=True)

    def _remove_orphan_compaction_artifacts(self) -> None:
        referenced_files = {payload.hdf5_file_name for payload in self._state.physical_payloads}
        referenced_files.update(payload.sidecar_file_name for payload in self._state.physical_payloads)
        containers_path = self.replay_inbox / 'containers'
        if containers_path.exists():
            for path in containers_path.iterdir():
                relative_name = path.relative_to(self.replay_inbox).as_posix()
                if relative_name not in referenced_files:
                    path.unlink()
        temporary_path = self.replay_inbox / 'tmp'
        if temporary_path.exists():
            for path in temporary_path.glob('.container-*.hdf5.tmp'):
                path.unlink()

    def _evict_outside_capacity(self) -> None:
        live_segments = list(self._state.live_segments)
        active_segment_ids = (
            set().union(*(pins.segment_ids for pins in self._active_leases.values())) if self._active_leases else set()
        )
        if self._state.active_compaction is not None:
            active_segment_ids.update(self._state.active_compaction.source_segment_ids)
        evicted: list[LogicalReplaySegment] = []
        total = sum(segment.unique_sample_count for segment in live_segments)
        while total > self.capacity and live_segments:
            candidate = live_segments[0]
            if candidate.segment_id in active_segment_ids:
                break
            evicted.append(live_segments.pop(0))
            total -= candidate.unique_sample_count
        if not evicted:
            return

        remaining_payload_ids = {segment.physical_payload_id for segment in live_segments}
        newly_retired = {
            segment.physical_payload_id
            for segment in evicted
            if segment.physical_payload_id not in remaining_payload_ids
        }
        self._persist_state(
            self._state_with(
                live_segments=tuple(live_segments),
                retired_payload_ids=tuple(dict.fromkeys((*self._state.retired_payload_ids, *newly_retired))),
            )
        )
        self._finish_retired_payloads()

    def _finish_retired_payloads(self) -> None:
        if not self._state.retired_payload_ids:
            return
        active_payload_ids = (
            set().union(*(pins.payload_ids for pins in self._active_leases.values())) if self._active_leases else set()
        )
        active_payload_ids.update(self._compaction_reserved_payload_ids)
        live_payload_ids = {segment.physical_payload_id for segment in self._state.live_segments}
        deletable_ids = {
            payload_id
            for payload_id in self._state.retired_payload_ids
            if payload_id not in active_payload_ids and payload_id not in live_payload_ids
        }
        if not deletable_ids:
            return
        for payload_id in deletable_ids:
            payload = self._payloads_by_id[payload_id]
            (self.replay_inbox / payload.hdf5_file_name).unlink(missing_ok=True)
            (self.replay_inbox / payload.sidecar_file_name).unlink(missing_ok=True)
        self._persist_state(
            self._state_with(
                physical_payloads=tuple(
                    payload for payload in self._state.physical_payloads if payload.payload_id not in deletable_ids
                ),
                retired_payload_ids=tuple(
                    payload_id for payload_id in self._state.retired_payload_ids if payload_id not in deletable_ids
                ),
            )
        )


def _chess_mirror_action_map() -> npt.NDArray[np.int64]:
    game = ChessGame()
    mapping = np.empty(game.action_size, dtype=np.int64)
    for action, move in game.index2move.items():
        from_row, from_column = square_to_index(move.from_square)
        to_row, to_column = square_to_index(move.to_square)
        mirrored_move = DictMove(
            from_square=index_to_square(from_row, BOARD_LENGTH - 1 - from_column),
            to_square=index_to_square(to_row, BOARD_LENGTH - 1 - to_column),
            promotion=move.promotion,
        )
        mapping[action] = game.move2index[mirrored_move]
    return mapping


CHESS_MIRROR_ACTION_MAP = _chess_mirror_action_map()


def _decode_with_deterministic_symmetry(
    encoded_states: list[bytes],
    visit_counts: list[npt.NDArray[np.uint16]],
    value_targets: list[ReplayValueTarget],
    sample_metadata: list[ReplaySampleMetadata],
    sampler_seed: int,
    global_step: int,
    partition: RankSamplePartition,
) -> TrainingBatch:
    states = decode_board_states(encoded_states).astype(np.float32, copy=False)
    policies = np.zeros((len(encoded_states), CurrentGame.action_size), dtype=np.float32)
    for row, visits in enumerate(visit_counts):
        policies[row, visits[:, 0]] = visits[:, 1]
    policy_totals = np.sum(policies, axis=1, keepdims=True)
    if np.any(policy_totals <= 0):
        raise ValueError('Visit counts must contain a positive total.')
    policies /= policy_totals

    mirrored = np.fromiter(
        (
            _sample_is_mirrored(
                sampler_seed,
                global_step,
                partition.rank,
                reference.global_batch_position,
            )
            for reference in partition.references
        ),
        dtype=np.bool_,
        count=len(partition.references),
    )
    states[mirrored] = np.flip(states[mirrored], axis=3)
    mirrored_policies = policies[mirrored].copy()
    policies[mirrored] = 0.0
    mirrored_rows = np.flatnonzero(mirrored)
    policies[
        mirrored_rows[:, np.newaxis],
        CHESS_MIRROR_ACTION_MAP[np.newaxis, :],
    ] = mirrored_policies

    return TrainingBatch(
        states=torch.from_numpy(states),
        policy_targets=torch.from_numpy(policies),
        final_outcomes=torch.from_numpy(
            np.fromiter(
                (int(target.final_outcome) for target in value_targets),
                dtype=np.int64,
            )
        ),
        mcts_root_values=torch.from_numpy(
            np.fromiter(
                (target.mcts_root_value for target in value_targets),
                dtype=np.float32,
            )
        ),
        outcome_target_eligible=torch.from_numpy(
            np.fromiter(
                (target.outcome_target_eligible for target in value_targets),
                dtype=np.bool_,
            )
        ),
        termination_reasons=torch.from_numpy(
            np.fromiter(
                (int(target.termination_reason) for target in value_targets),
                dtype=np.int64,
            )
        ),
        plies=torch.from_numpy(np.fromiter((metadata.ply for metadata in sample_metadata), dtype=np.int32)),
        current_player_piece_counts=torch.from_numpy(
            np.fromiter(
                (metadata.current_player_piece_count for metadata in sample_metadata),
                dtype=np.int8,
            )
        ),
        opponent_piece_counts=torch.from_numpy(
            np.fromiter(
                (metadata.opponent_piece_count for metadata in sample_metadata),
                dtype=np.int8,
            )
        ),
    )


def _sample_is_mirrored(
    sampler_seed: int,
    global_step: int,
    rank: int,
    global_batch_position: int,
) -> bool:
    mask = (1 << 64) - 1
    value = (
        (sampler_seed & mask)
        ^ (((global_step + 1) * 0x9E3779B97F4A7C15) & mask)
        ^ (((rank + 1) * 0xBF58476D1CE4E5B9) & mask)
        ^ (((global_batch_position + 1) * 0x94D049BB133111EB) & mask)
    )
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    value ^= value >> 31
    return bool(value & 1)


def decode_rank_quantum(
    replay_buffer: RollingReplayBuffer,
    request: ReplayQuantumRequest,
) -> ReplayTrainingQuantum:
    if request.optimizer_steps <= 0:
        raise ValueError('Replay quantum optimizer steps must be positive.')
    if not 0 <= request.rank < request.world_size:
        raise ValueError('Replay request rank must be within its world size.')
    with replay_buffer.lease_quantum(
        global_step=request.global_step,
        global_sample_count=request.global_sample_count,
        world_size=request.world_size,
        global_batch_size=request.global_batch_size,
    ) as lease:
        full_batch = replay_buffer.decode_partition(
            lease.partitions[request.rank],
            request.global_step,
        )
    return ReplayTrainingQuantum(
        global_step=request.global_step,
        rank=request.rank,
        local_batch_size=request.local_batch_size,
        full_batch=full_batch,
        decode_statistics=replay_buffer.last_decode_statistics,
    )


def prefetch_rank_quanta(
    replay_buffer: RollingReplayBuffer,
    requests: Sequence[ReplayQuantumRequest],
) -> Iterator[ReplayTrainingQuantum]:
    if not requests:
        return
    with ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix='replay-quantum-prefetch',
    ) as executor:
        future: Future[ReplayTrainingQuantum] = executor.submit(
            decode_rank_quantum,
            replay_buffer,
            requests[0],
        )
        for request_index in range(len(requests)):
            quantum = future.result()
            if request_index + 1 < len(requests):
                future = executor.submit(
                    decode_rank_quantum,
                    replay_buffer,
                    requests[request_index + 1],
                )
            yield quantum
