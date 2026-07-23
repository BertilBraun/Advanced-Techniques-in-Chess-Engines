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
from pathlib import Path
from types import TracebackType

import h5py
import numpy as np
import numpy.typing as npt
import torch
from pydantic import BaseModel, ConfigDict

from src.Encoding import decode_board_states
from src.games.chess.ChessGame import BOARD_LENGTH, ChessGame, DictMove, index_to_square, square_to_index
from src.self_play.SelfPlayDataset import (
    ReplaySampleMetadata,
    SelfPlayDataset,
    TrainingBatch,
)
from src.self_play.value_target import (
    REPLAY_SCHEMA_VERSION,
    FinalOutcome,
    ReplayValueTarget,
    TerminationReason,
)
from src.settings import CurrentGame


DEFAULT_REPLAY_CAPACITY = 2_500_000
PRESENTATION_CREDITS_PER_UNIQUE_SAMPLE = 4
REPLAY_INDEX_SCHEMA_VERSION = 1


class TerminationCounts(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    natural: int
    resignation: int
    ply_cap: int
    material_adjudication: int
    diagnostic: int

    @classmethod
    def from_targets(cls, targets: list[ReplayValueTarget]) -> TerminationCounts:
        values = np.fromiter((int(target.termination_reason) for target in targets), dtype=np.uint8)
        return cls(
            natural=int(np.count_nonzero(values == int(TerminationReason.NATURAL))),
            resignation=int(np.count_nonzero(values == int(TerminationReason.RESIGNATION))),
            ply_cap=int(np.count_nonzero(values == int(TerminationReason.PLY_CAP))),
            material_adjudication=int(np.count_nonzero(values == int(TerminationReason.MATERIAL_ADJUDICATION))),
            diagnostic=int(np.count_nonzero(values == int(TerminationReason.DIAGNOSTIC))),
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


class ReplayIndexState(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    schema_version: int
    sampler_seed: int
    live_shards: tuple[ReplayShardManifest, ...]
    evicted_shard_ids: tuple[str, ...]


@dataclass(frozen=True)
class ReplayIngestResult:
    unique_samples: int
    presentation_credits: int


@dataclass(frozen=True)
class GlobalSampleReference:
    shard_id: str
    sample_index: int
    global_batch_position: int


@dataclass(frozen=True)
class RankSamplePartition:
    rank: int
    references: tuple[GlobalSampleReference, ...]


@dataclass(frozen=True)
class ReplayBatchRequest:
    global_step: int
    global_sample_count: int
    world_size: int
    rank: int


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
        replay_buffer: DiskReplayBuffer,
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


class DiskReplayBuffer:
    def __init__(
        self,
        replay_inbox: Path,
        index_path: Path,
        capacity: int = DEFAULT_REPLAY_CAPACITY,
        sampler_seed: int = 0,
    ) -> None:
        if capacity <= 0:
            raise ValueError('Replay capacity must be positive.')
        self.replay_inbox = replay_inbox
        self.index_path = index_path
        self.capacity = capacity
        self._lock = threading.RLock()
        self._active_leases: dict[str, frozenset[str]] = {}
        self._pending_evictions: set[str] = set()
        self._state = self._load_or_create_state(sampler_seed)
        self._prefix = np.asarray([0], dtype=np.int64)
        self._shards_by_id: dict[str, ReplayShardManifest] = {}
        self._rebuild_lookup()
        self._finish_recorded_evictions()

    @property
    def unique_sample_count(self) -> int:
        return int(self._prefix[-1])

    @property
    def shard_count(self) -> int:
        return len(self._state.live_shards)

    @property
    def metadata_memory_bytes(self) -> int:
        manifest_bytes = sum(len(manifest.model_dump_json().encode('utf-8')) for manifest in self._state.live_shards)
        return manifest_bytes + self._prefix.nbytes

    def discover_committed_shards(self) -> ReplayIngestResult:
        with self._lock:
            known_ids = {manifest.shard_id for manifest in self._state.live_shards}
            known_ids.update(self._state.evicted_shard_ids)
            discovered: list[ReplayShardManifest] = []
            for manifest_path in sorted(self.replay_inbox.glob('*.manifest.json')):
                manifest = ReplayShardManifest.model_validate_json(manifest_path.read_text(encoding='utf-8'))
                if manifest.shard_id in known_ids:
                    continue
                self._validate_manifest(manifest)
                discovered.append(manifest)
                known_ids.add(manifest.shard_id)

            if not discovered:
                return ReplayIngestResult(unique_samples=0, presentation_credits=0)

            live_shards = tuple(
                sorted(
                    (*self._state.live_shards, *discovered),
                    key=lambda manifest: (manifest.creation_timestamp_seconds, manifest.shard_id),
                )
            )
            self._replace_state(live_shards=live_shards)
            self._evict_outside_capacity()
            unique_samples = sum(manifest.unique_sample_count for manifest in discovered)
            return ReplayIngestResult(
                unique_samples=unique_samples,
                presentation_credits=unique_samples * PRESENTATION_CREDITS_PER_UNIQUE_SAMPLE,
            )

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
        if world_size <= 0 or global_sample_count % world_size != 0:
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
            shard_ids = frozenset(reference.shard_id for reference in references)
            lease_id = uuid.uuid4().hex
            self._active_leases[lease_id] = shard_ids
            return ReplayBatchLease(self, lease_id, partitions)

    def release_lease(self, lease_id: str) -> None:
        with self._lock:
            if lease_id not in self._active_leases:
                raise ValueError(f'Unknown replay lease {lease_id}.')
            del self._active_leases[lease_id]
            self._evict_outside_capacity()

    def decode_partition(
        self,
        partition: RankSamplePartition,
        global_step: int,
    ) -> TrainingBatch:
        if not partition.references:
            raise ValueError('Cannot decode an empty rank partition.')
        grouped: dict[str, list[tuple[int, GlobalSampleReference]]] = defaultdict(list)
        for output_position, reference in enumerate(partition.references):
            grouped[reference.shard_id].append((output_position, reference))

        encoded_states: list[bytes | None] = [None] * len(partition.references)
        visit_counts: list[npt.NDArray[np.uint16] | None] = [None] * len(partition.references)
        value_targets: list[ReplayValueTarget | None] = [None] * len(partition.references)
        sample_metadata: list[ReplaySampleMetadata | None] = [None] * len(partition.references)

        for shard_id, output_references in grouped.items():
            manifest = self._shards_by_id[shard_id]
            sorted_references = sorted(output_references, key=lambda item: item[1].sample_index)
            unique_indices = np.asarray(
                sorted({reference.sample_index for _, reference in sorted_references}),
                dtype=np.int64,
            )
            with h5py.File(self.replay_inbox / manifest.hdf5_file_name, 'r') as file:
                states = np.asarray(file['states'][unique_indices])
                visits = np.asarray(file['visit_counts'][unique_indices])
                outcomes = np.asarray(file['final_outcomes'][unique_indices], dtype=np.uint8)
                root_values = np.asarray(file['mcts_root_values'][unique_indices], dtype=np.float32)
                eligibility = np.asarray(file['outcome_target_eligible'][unique_indices], dtype=np.bool_)
                reasons = np.asarray(file['termination_reasons'][unique_indices], dtype=np.uint8)
                plies = np.asarray(file['plies'][unique_indices], dtype=np.int32)
                current_counts = np.asarray(
                    file['current_player_piece_counts'][unique_indices],
                    dtype=np.uint8,
                )
                opponent_counts = np.asarray(file['opponent_piece_counts'][unique_indices], dtype=np.uint8)

            source_positions = {
                int(sample_index): source_position for source_position, sample_index in enumerate(unique_indices)
            }
            for output_position, reference in sorted_references:
                source_position = source_positions[reference.sample_index]
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

        complete_states = [state for state in encoded_states if state is not None]
        complete_visits = [visits for visits in visit_counts if visits is not None]
        complete_targets = [target for target in value_targets if target is not None]
        complete_metadata = [metadata for metadata in sample_metadata if metadata is not None]
        if not all(
            len(values) == len(partition.references)
            for values in (complete_states, complete_visits, complete_targets, complete_metadata)
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

    def _load_or_create_state(self, sampler_seed: int) -> ReplayIndexState:
        if self.index_path.exists():
            state = ReplayIndexState.model_validate_json(self.index_path.read_text(encoding='utf-8'))
            if state.schema_version != REPLAY_INDEX_SCHEMA_VERSION:
                raise ValueError(
                    f'Replay index schema {state.schema_version} is unsupported; '
                    f'expected {REPLAY_INDEX_SCHEMA_VERSION}.'
                )
            return state
        state = ReplayIndexState(
            schema_version=REPLAY_INDEX_SCHEMA_VERSION,
            sampler_seed=sampler_seed,
            live_shards=(),
            evicted_shard_ids=(),
        )
        _atomic_write_json(self.index_path, state.model_dump_json(indent=2))
        return state

    def _validate_manifest(self, manifest: ReplayShardManifest) -> None:
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

    def _reference_for_index(self, global_index: int, position: int) -> GlobalSampleReference:
        shard_index = int(np.searchsorted(self._prefix, global_index + 1) - 1)
        manifest = self._state.live_shards[shard_index]
        return GlobalSampleReference(
            shard_id=manifest.shard_id,
            sample_index=global_index - int(self._prefix[shard_index]),
            global_batch_position=position,
        )

    def _replace_state(
        self,
        live_shards: tuple[ReplayShardManifest, ...],
        evicted_shard_ids: tuple[str, ...] | None = None,
    ) -> None:
        self._state = ReplayIndexState(
            schema_version=REPLAY_INDEX_SCHEMA_VERSION,
            sampler_seed=self._state.sampler_seed,
            live_shards=live_shards,
            evicted_shard_ids=(self._state.evicted_shard_ids if evicted_shard_ids is None else evicted_shard_ids),
        )
        _atomic_write_json(self.index_path, self._state.model_dump_json(indent=2))
        self._rebuild_lookup()

    def _rebuild_lookup(self) -> None:
        self._shards_by_id = {manifest.shard_id: manifest for manifest in self._state.live_shards}
        lengths = np.fromiter(
            (manifest.unique_sample_count for manifest in self._state.live_shards),
            dtype=np.int64,
            count=len(self._state.live_shards),
        )
        self._prefix = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(lengths)))

    def _evict_outside_capacity(self) -> None:
        live_shards = list(self._state.live_shards)
        active_shard_ids = set().union(*self._active_leases.values()) if self._active_leases else set()
        evicted: list[ReplayShardManifest] = []
        total = sum(manifest.unique_sample_count for manifest in live_shards)
        while total > self.capacity and live_shards:
            candidate = live_shards[0]
            if candidate.shard_id in active_shard_ids:
                self._pending_evictions.add(candidate.shard_id)
                break
            evicted.append(live_shards.pop(0))
            total -= candidate.unique_sample_count
        if not evicted:
            return
        evicted_ids = tuple((*self._state.evicted_shard_ids, *(manifest.shard_id for manifest in evicted)))
        self._replace_state(tuple(live_shards), evicted_ids)
        self._finish_recorded_evictions()
        for manifest in evicted:
            self._pending_evictions.discard(manifest.shard_id)

    def _finish_recorded_evictions(self) -> None:
        if not self._state.evicted_shard_ids:
            return
        for shard_id in self._state.evicted_shard_ids:
            hdf5_path = self.replay_inbox / f'{shard_id}.hdf5'
            manifest_path = self.replay_inbox / f'{shard_id}.manifest.json'
            hdf5_path.unlink(missing_ok=True)
            manifest_path.unlink(missing_ok=True)
        self._replace_state(self._state.live_shards, ())


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
    policies[mirrored_rows[:, np.newaxis], CHESS_MIRROR_ACTION_MAP[np.newaxis, :]] = mirrored_policies

    return TrainingBatch(
        states=torch.from_numpy(states),
        policy_targets=torch.from_numpy(policies),
        final_outcomes=torch.from_numpy(
            np.fromiter((int(target.final_outcome) for target in value_targets), dtype=np.int64)
        ),
        mcts_root_values=torch.from_numpy(
            np.fromiter((target.mcts_root_value for target in value_targets), dtype=np.float32)
        ),
        outcome_target_eligible=torch.from_numpy(
            np.fromiter((target.outcome_target_eligible for target in value_targets), dtype=np.bool_)
        ),
        termination_reasons=torch.from_numpy(
            np.fromiter((int(target.termination_reason) for target in value_targets), dtype=np.int64)
        ),
        plies=torch.from_numpy(np.fromiter((metadata.ply for metadata in sample_metadata), dtype=np.int32)),
        current_player_piece_counts=torch.from_numpy(
            np.fromiter((metadata.current_player_piece_count for metadata in sample_metadata), dtype=np.int8)
        ),
        opponent_piece_counts=torch.from_numpy(
            np.fromiter((metadata.opponent_piece_count for metadata in sample_metadata), dtype=np.int8)
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


def prefetch_replay_batches(
    replay_buffer: DiskReplayBuffer,
    requests: Sequence[ReplayBatchRequest],
    maximum_prefetched_batches: int,
) -> Iterator[TrainingBatch]:
    if maximum_prefetched_batches <= 0:
        raise ValueError('Maximum prefetched batches must be positive.')

    def decode(request: ReplayBatchRequest) -> TrainingBatch:
        if not 0 <= request.rank < request.world_size:
            raise ValueError('Replay request rank must be within its world size.')
        with replay_buffer.lease_quantum(
            request.global_step,
            request.global_sample_count,
            request.world_size,
        ) as lease:
            return replay_buffer.decode_partition(
                lease.partitions[request.rank],
                request.global_step,
            )

    with ThreadPoolExecutor(
        max_workers=maximum_prefetched_batches,
        thread_name_prefix='replay-prefetch',
    ) as executor:
        futures: dict[int, Future[TrainingBatch]] = {}
        next_submission = 0
        next_result = 0
        while next_result < len(requests):
            while next_submission < len(requests) and len(futures) < maximum_prefetched_batches:
                futures[next_submission] = executor.submit(decode, requests[next_submission])
                next_submission += 1
            yield futures.pop(next_result).result()
            next_result += 1


def slice_rank_partition(
    partition: RankSamplePartition,
    local_batch_size: int,
) -> tuple[RankSamplePartition, ...]:
    if local_batch_size <= 0:
        raise ValueError('Local batch size must be positive.')
    if len(partition.references) % local_batch_size:
        raise ValueError('Rank partition must divide evenly into local batches.')
    return tuple(
        RankSamplePartition(
            rank=partition.rank,
            references=partition.references[offset : offset + local_batch_size],
        )
        for offset in range(0, len(partition.references), local_batch_size)
    )


def prefetch_leased_replay_batches(
    replay_buffer: DiskReplayBuffer,
    partitions: Sequence[RankSamplePartition],
    global_step: int,
    maximum_prefetched_batches: int,
) -> Iterator[TrainingBatch]:
    if maximum_prefetched_batches <= 0:
        raise ValueError('Maximum prefetched batches must be positive.')

    with ThreadPoolExecutor(
        max_workers=maximum_prefetched_batches,
        thread_name_prefix='leased-replay-prefetch',
    ) as executor:
        futures: dict[int, Future[TrainingBatch]] = {}
        next_submission = 0
        next_result = 0
        while next_result < len(partitions):
            while next_submission < len(partitions) and len(futures) < maximum_prefetched_batches:
                futures[next_submission] = executor.submit(
                    replay_buffer.decode_partition,
                    partitions[next_submission],
                    global_step,
                )
                next_submission += 1
            yield futures.pop(next_result).result()
            next_result += 1
