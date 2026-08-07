from __future__ import annotations

import hashlib
import os
import struct
import time
from collections import deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from AlphaZeroCpp import GoPlayer, GoRules

from src.games.go.contract import GoStateContract, GoSymmetryIndex, NativeGoPosition
from src.packed_planes import PackedPlanePayload
from src.self_play.SelfPlayDataset import ReplaySampleMetadata, TrainingBatch
from src.self_play.go_completed_game import (
    GoCompletedGame,
    GoGameIdentity,
    go_completed_game_from_path,
)
from src.self_play.value_target import ReplayValueTarget, TerminationReason, outcome_from_sample_perspective
from src.train.replay_sampling import deterministic_rank_indices
from src.util.atomic_file import fsync_directory, write_bytes_atomically


GO_ARCHIVE_HEADER = b'AZ-GO-GAMES\x00\x01\n'
_FRAME_HEADER = struct.Struct('>QQ32s')
_ARCHIVE_PATTERN = 'model-generation-*.games'


def pack_go_visits(visits: Sequence[tuple[int, int]], action_size: int) -> npt.NDArray[np.uint16]:
    if not visits:
        raise ValueError('Packed Go visits must not be empty.')
    if any(
        not 0 <= action_id < action_size or visit_count <= 0 or visit_count > int(np.iinfo(np.uint16).max)
        for action_id, visit_count in visits
    ):
        raise ValueError('Packed Go actions or visits lie outside their uint16 ranges.')
    packed = np.asarray(visits, dtype=np.uint16)
    packed.flags.writeable = False
    return packed


@dataclass(frozen=True, eq=False)
class GoReplaySample:
    encoded_state: PackedPlanePayload
    visits: npt.NDArray[np.uint16]
    value_target: ReplayValueTarget
    metadata: ReplaySampleMetadata
    sample_weight: float
    source_model_generation: int
    source_created_at_seconds: float

    def __post_init__(self) -> None:
        if self.visits.dtype != np.uint16 or self.visits.ndim != 2 or self.visits.shape[1] != 2:
            raise ValueError('Packed Go visits must have shape (N, 2) and uint16 dtype.')
        if self.visits.flags.writeable:
            raise ValueError('Packed Go visits must be read-only.')

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GoReplaySample):
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
class GoReplaySnapshot:
    contract: GoStateContract
    samples: tuple[GoReplaySample, ...]
    credited_samples: int
    credited_completed_searches: int
    sampler_seed: int
    frozen_at_seconds: float
    evicted_samples: int

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


class GoReplay:
    def __init__(self, contract: GoStateContract, capacity: int, sampler_seed: int) -> None:
        if capacity <= 0:
            raise ValueError('Go replay capacity must be positive.')
        self.contract = contract
        self.capacity = capacity
        self.sampler_seed = sampler_seed
        self.phase = ReplayPhase.INGESTING
        self._samples: deque[GoReplaySample] = deque()
        self._credited_samples = 0
        self._credited_completed_searches = 0
        self._evicted_samples = 0

    def begin_ingestion(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError('Go replay capacity must be positive.')
        self.phase = ReplayPhase.INGESTING
        self.capacity = capacity
        self._evict()

    def ingest_game(self, game: GoCompletedGame) -> int:
        if self.phase is not ReplayPhase.INGESTING:
            raise RuntimeError('Go replay ingestion is allowed only during ingestion.')
        samples = materialize_go_game(game)
        self._samples.extend(samples)
        self._credited_samples += len(samples)
        if samples:
            self._credited_completed_searches += sum(
                observation.search_budget for observation in game.observations if observation.sample_eligible
            )
        self._evict()
        return len(samples)

    def freeze(self) -> GoReplaySnapshot:
        if self.phase is not ReplayPhase.INGESTING:
            raise RuntimeError('Go replay is already frozen.')
        self.phase = ReplayPhase.FROZEN
        return GoReplaySnapshot(
            contract=self.contract,
            samples=tuple(self._samples),
            credited_samples=self._credited_samples,
            credited_completed_searches=self._credited_completed_searches,
            sampler_seed=self.sampler_seed,
            frozen_at_seconds=time.time(),
            evicted_samples=self._evicted_samples,
        )

    def _evict(self) -> None:
        while len(self._samples) > self.capacity:
            self._samples.popleft()
            self._evicted_samples += 1


@dataclass(frozen=True)
class GoArchiveFrame:
    path: Path
    ingestion_sequence: int
    payload_offset: int
    payload_length: int
    payload_digest: bytes


def _index_archive(path: Path, recover_incomplete: bool) -> tuple[GoArchiveFrame, ...]:
    mode = 'r+b' if recover_incomplete else 'rb'
    frames: list[GoArchiveFrame] = []
    with path.open(mode) as archive:
        if archive.read(len(GO_ARCHIVE_HEADER)) != GO_ARCHIVE_HEADER:
            raise ValueError(f'Unsupported Go archive header: {path}')
        archive_size = os.fstat(archive.fileno()).st_size
        valid_end = len(GO_ARCHIVE_HEADER)
        while True:
            header = archive.read(_FRAME_HEADER.size)
            if not header:
                break
            if len(header) != _FRAME_HEADER.size:
                if not recover_incomplete:
                    raise ValueError(f'Go archive has an incomplete final frame header: {path}')
                archive.truncate(valid_end)
                break
            sequence, payload_length, digest = _FRAME_HEADER.unpack(header)
            payload_offset = archive.tell()
            payload_end = payload_offset + payload_length
            if payload_end > archive_size:
                if not recover_incomplete:
                    raise ValueError(f'Go archive has an incomplete final frame payload: {path}')
                archive.truncate(valid_end)
                break
            frames.append(GoArchiveFrame(path, sequence, payload_offset, payload_length, digest))
            archive.seek(payload_end)
            valid_end = payload_end
        if recover_incomplete:
            archive.flush()
            os.fsync(archive.fileno())
    return tuple(frames)


def _read_frame(frame: GoArchiveFrame) -> GoCompletedGame:
    with frame.path.open('rb') as archive:
        archive.seek(frame.payload_offset)
        payload = archive.read(frame.payload_length)
    if len(payload) != frame.payload_length or hashlib.sha256(payload).digest() != frame.payload_digest:
        raise ValueError(f'Go archive frame checksum failed: {frame.path}')
    return GoCompletedGame.model_validate_json(payload)


def append_go_archive(path: Path, game: GoCompletedGame, ingestion_sequence: int) -> None:
    payload = game.model_dump_json().encode('utf-8')
    if not path.exists():
        write_bytes_atomically(path, GO_ARCHIVE_HEADER)
    _index_archive(path, recover_incomplete=True)
    with path.open('ab') as archive:
        archive.write(_FRAME_HEADER.pack(ingestion_sequence, len(payload), hashlib.sha256(payload).digest()))
        archive.write(payload)
        archive.flush()
        os.fsync(archive.fileno())


def read_go_archive(path: Path, recover_incomplete: bool = False) -> tuple[GoCompletedGame, ...]:
    return tuple(_read_frame(frame) for frame in _index_archive(path, recover_incomplete))


class GoReplayMaintainer:
    def __init__(self, run_path: Path, contract: GoStateContract, capacity: int, sampler_seed: int) -> None:
        self.run_path = run_path
        self.inbox_path = run_path / 'completed-games' / 'inbox'
        self.archive_path = run_path / 'completed-games' / 'archive'
        self.replay = GoReplay(contract, capacity, sampler_seed)
        self._archived_digests: dict[GoGameIdentity, bytes] = {}
        self._next_sequence = 0
        self._recover()

    def archive_file(self, model_generation: int) -> Path:
        return self.archive_path / f'model-generation-{model_generation:020d}.games'

    def maintain(self, capacity: int) -> GoReplaySnapshot:
        self.replay.begin_ingestion(capacity)
        for path in sorted(self.inbox_path.glob('*.json')):
            game = go_completed_game_from_path(path)
            if game.representation.board_size != self.replay.contract.board_size:
                raise ValueError('Inbox Go game board size disagrees with the replay contract.')
            payload_digest = hashlib.sha256(game.model_dump_json().encode('utf-8')).digest()
            existing = self._archived_digests.get(game.identity)
            if existing is None:
                append_go_archive(self.archive_file(game.model_generation), game, self._next_sequence)
                self._archived_digests[game.identity] = payload_digest
                self.replay.ingest_game(game)
                self._next_sequence += 1
            elif existing != payload_digest:
                raise ValueError(f'Archived Go game has conflicting identity {game.identity.archive_key}.')
            path.unlink()
            fsync_directory(path.parent)
        return self.replay.freeze()

    def _recover(self) -> None:
        indexed: list[tuple[GoArchiveFrame, GoCompletedGame]] = []
        for path in sorted(self.archive_path.glob(_ARCHIVE_PATTERN)):
            for frame in _index_archive(path, recover_incomplete=True):
                game = _read_frame(frame)
                if path != self.archive_file(game.model_generation):
                    raise ValueError(f'Go game is stored in the wrong generation archive: {path}')
                digest = frame.payload_digest
                existing = self._archived_digests.setdefault(game.identity, digest)
                if existing != digest:
                    raise ValueError(f'Archive contains conflicting Go identity {game.identity.archive_key}.')
                indexed.append((frame, game))
        indexed.sort(key=lambda pair: pair[0].ingestion_sequence)
        if tuple(frame.ingestion_sequence for frame, _ in indexed) != tuple(range(len(indexed))):
            raise ValueError('Go archive ingestion sequence is not contiguous.')
        for _, game in indexed:
            self.replay.ingest_game(game)
        self._next_sequence = len(indexed)


def rebuild_go_replay(run_path: Path, contract: GoStateContract, capacity: int, sampler_seed: int) -> GoReplaySnapshot:
    return GoReplayMaintainer(run_path, contract, capacity, sampler_seed).maintain(capacity)


def materialize_go_game(game: GoCompletedGame) -> tuple[GoReplaySample, ...]:
    contract = GoStateContract(game.representation.board_size, game.representation.history_length)
    rules = GoRules(game.rules.komi_half_points, game.rules.maximum_moves)
    position: NativeGoPosition = contract.initial_position(rules)
    observations = {observation.ply: observation for observation in game.observations}
    samples: list[GoReplaySample] = []
    for ply, action in enumerate(game.actions):
        observation = observations.get(ply)
        if observation is not None:
            legal_actions = tuple(sorted(position.legal_actions()))
            if legal_actions != observation.legal_action_ids:
                raise ValueError(f'Completed Go legal actions disagree at ply {ply}.')
            if action != observation.selected_action_id:
                raise ValueError(f'Completed Go selected action disagrees at ply {ply}.')
            if observation.sample_eligible:
                visits = tuple(
                    (visit.action_id, visit.visit_count - observation.minimum_visit_count)
                    for visit in observation.visits
                    if visit.visit_count > observation.minimum_visit_count
                )
                if not visits:
                    raise ValueError(f'Eligible Go observation has no visits after filtering at ply {ply}.')
                black_count = len(position.black_points(0))
                white_count = len(position.white_points(0))
                sample_player = contract.player_sign(position.player)
                final_score = outcome_from_sample_perspective(
                    game.final_score,
                    final_current_player=game.final_current_player,
                    sample_current_player=sample_player,
                )
                samples.append(
                    GoReplaySample(
                        encoded_state=contract.packed_position(position),
                        visits=pack_go_visits(visits, contract.action_size),
                        value_target=ReplayValueTarget.from_scores(
                            final_score,
                            observation.root_value,
                            TerminationReason.NATURAL,
                        ),
                        metadata=ReplaySampleMetadata(
                            ply=ply,
                            current_player_piece_count=black_count
                            if position.player == GoPlayer.BLACK
                            else white_count,
                            opponent_piece_count=white_count if position.player == GoPlayer.BLACK else black_count,
                        ),
                        sample_weight=observation.sample_weight,
                        source_model_generation=observation.model_generation,
                        source_created_at_seconds=game.created_at_seconds,
                    )
                )
        position = position.child(action)
    if not position.is_terminal or contract.player_sign(position.player) != game.final_current_player:
        raise ValueError('Completed Go game does not reconstruct to its declared terminal player.')
    terminal = position.terminal_result()
    if terminal.reason.name.lower() != game.termination_reason.value:
        raise ValueError('Completed Go termination reason disagrees with reconstructed moves.')
    if position.terminal_value() != game.final_score:
        raise ValueError('Completed Go final score disagrees with the reconstructed position.')
    return tuple(reversed(samples))


def _symmetry_index(sampler_seed: int, global_step: int, rank: int, sample_position: int) -> GoSymmetryIndex:
    sequence = np.random.SeedSequence((sampler_seed, global_step, rank, sample_position))
    return GoSymmetryIndex(int(np.random.default_rng(sequence).integers(0, 8)))


def build_go_training_batch(
    snapshot: GoReplaySnapshot,
    sample_indices: Sequence[int],
    global_step: int,
    rank: int,
    sample_position_offset: int = 0,
) -> TrainingBatch:
    samples = tuple(snapshot.samples[index] for index in sample_indices)
    batch_size = len(samples)
    states = np.empty(
        (batch_size, snapshot.contract.channels, snapshot.contract.board_size, snapshot.contract.board_size),
        dtype=np.float32,
    )
    policies = np.zeros((batch_size, snapshot.contract.action_size), dtype=np.float32)
    transformed_states: list[PackedPlanePayload] = []
    for row, sample in enumerate(samples):
        symmetry = _symmetry_index(snapshot.sampler_seed, global_step, rank, sample_position_offset + row)
        transformed_states.append(snapshot.contract.transform_state(sample.encoded_state, symmetry))
        actions = tuple(snapshot.contract.transform_action(int(action), symmetry) for action in sample.visits[:, 0])
        counts = sample.visits[:, 1].astype(np.float32, copy=False)
        policies[row, actions] = counts / counts.sum()
    snapshot.contract.decode_batch_into(tuple(transformed_states), states)
    return TrainingBatch(
        states=torch.from_numpy(states),
        policy_targets=torch.from_numpy(policies),
        final_outcomes=torch.tensor([int(sample.value_target.final_outcome) for sample in samples]),
        mcts_root_values=torch.tensor([sample.value_target.mcts_root_value for sample in samples]),
        outcome_target_eligible=torch.tensor(
            [sample.value_target.outcome_target_eligible for sample in samples], dtype=torch.bool
        ),
        material_result_scores=torch.zeros(batch_size, dtype=torch.float32),
        material_target_eligible=torch.zeros(batch_size, dtype=torch.bool),
        termination_reasons=torch.tensor([int(sample.value_target.termination_reason) for sample in samples]),
        plies=torch.tensor([sample.metadata.ply for sample in samples], dtype=torch.int32),
        current_player_piece_counts=torch.tensor(
            [sample.metadata.current_player_piece_count for sample in samples], dtype=torch.int8
        ),
        opponent_piece_counts=torch.tensor(
            [sample.metadata.opponent_piece_count for sample in samples], dtype=torch.int8
        ),
        occurrence_counts=torch.ones(batch_size, dtype=torch.int32),
        sample_weights=torch.tensor([sample.sample_weight for sample in samples], dtype=torch.float32),
    )


class GoTrainingBatchLoader:
    def __init__(
        self,
        snapshot: GoReplaySnapshot,
        global_step: int,
        optimizer_steps: int,
        global_batch_size: int,
        world_size: int,
        rank: int,
        pin_memory: bool,
    ) -> None:
        self.snapshot = snapshot
        self.global_step = global_step
        self.rank = rank
        self.local_batch_size = global_batch_size // world_size
        self.indices = snapshot.rank_indices(global_step, optimizer_steps, global_batch_size, world_size, rank)
        self.optimizer_steps = optimizer_steps
        self.pin_memory = pin_memory

    def __iter__(self) -> Iterator[TrainingBatch]:
        for offset in range(0, len(self.indices), self.local_batch_size):
            batch = build_go_training_batch(
                self.snapshot,
                self.indices[offset : offset + self.local_batch_size],
                self.global_step,
                self.rank,
                offset,
            )
            yield batch.pin_memory() if self.pin_memory else batch

    def __len__(self) -> int:
        return self.optimizer_steps
