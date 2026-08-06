from __future__ import annotations

import hashlib
import os
from collections import deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import struct
import time

import chess
import numpy as np
import torch

from src.Encoding import decode_board_states_into, encode_board_state
from src.games.chess.ChessBoard import ChessBoard
from src.games.chess.ChessGame import BOARD_LENGTH, ChessGame, DictMove, index_to_square, square_to_index
from src.self_play.SelfPlayDataset import ReplaySampleMetadata, TrainingBatch
from src.self_play.chess_completed_game import ChessCompletedGame, completed_game_from_path
from src.self_play.value_target import ReplayValueTarget, outcome_from_sample_perspective
from src.settings import CurrentGame
from src.util.atomic_file import fsync_directory, write_bytes_atomically


CHESS_ARCHIVE_HEADER = b'AZ-CHESS-GAMES\x00\x01\n'
_FRAME_HEADER = struct.Struct('>Q32s')
_ARCHIVE_FILE_PATTERN = 'model-generation-*.games'


@dataclass(frozen=True)
class ChessReplaySample:
    encoded_state: bytes
    visits: tuple[tuple[int, int], ...]
    value_target: ReplayValueTarget
    metadata: ReplaySampleMetadata
    sample_weight: float
    source_model_generation: int
    source_created_at_seconds: float


@dataclass(frozen=True)
class ChessReplaySnapshot:
    samples: tuple[ChessReplaySample, ...]
    credited_samples: int
    credited_completed_searches: int
    sampler_seed: int
    frozen_at_seconds: float
    evicted_samples: int
    estimated_sample_bytes: int

    def rank_indices(
        self,
        global_step: int,
        optimizer_steps: int,
        global_batch_size: int,
        world_size: int,
        rank: int,
    ) -> tuple[int, ...]:
        if global_step < 0 or optimizer_steps <= 0 or global_batch_size <= 0:
            raise ValueError('Replay sampling counters and sizes are invalid.')
        if world_size <= 0 or not 0 <= rank < world_size or global_batch_size % world_size:
            raise ValueError('Replay rank partition is invalid.')
        global_sample_count = optimizer_steps * global_batch_size
        replay_size = len(self.samples)
        if replay_size >= global_sample_count:
            generator = np.random.default_rng(np.random.SeedSequence((self.sampler_seed, global_step)))
            global_indices = generator.choice(replay_size, size=global_sample_count, replace=False)
        elif replay_size >= global_batch_size:
            generator = np.random.default_rng(np.random.SeedSequence((self.sampler_seed, global_step)))
            global_indices = np.concatenate(
                tuple(
                    generator.choice(replay_size, size=global_batch_size, replace=False) for _ in range(optimizer_steps)
                )
            )
        else:
            raise ValueError(
                f'Live replay has {replay_size} positions but a duplicate-free global batch requires '
                f'{global_batch_size}.'
            )
        local_batch_size = global_batch_size // world_size
        matrix = global_indices.reshape(optimizer_steps, world_size, local_batch_size)
        return tuple(int(index) for index in matrix[:, rank, :].reshape(-1))


class ChessTrainingBatchLoader:
    def __init__(
        self,
        snapshot: ChessReplaySnapshot,
        global_step: int,
        optimizer_steps: int,
        global_batch_size: int,
        world_size: int,
        rank: int,
        pin_memory: bool,
    ) -> None:
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
            batch = build_chess_training_batch(
                self.snapshot,
                self.indices[offset : offset + self.local_batch_size],
                self.global_step,
                self.rank,
            )
            if self.pin_memory:
                batch = batch.pin_memory()
            self.preparation_seconds += time.perf_counter() - started_at
            yield batch

    def __len__(self) -> int:
        return self.optimizer_steps


class ReplayPhase(str, Enum):
    INGESTING = 'ingesting'
    FROZEN = 'frozen'


@dataclass(frozen=True)
class ChessReplayMetrics:
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


@dataclass(frozen=True)
class ChessArchiveInspection:
    path: Path
    model_generation: int
    game_count: int
    eligible_sample_count: int
    completed_searches: int
    byte_count: int


class ChessReplay:
    def __init__(self, capacity: int, sampler_seed: int) -> None:
        if capacity <= 0:
            raise ValueError('Chess replay capacity must be positive.')
        self.capacity = capacity
        self.sampler_seed = sampler_seed
        self.phase = ReplayPhase.INGESTING
        self._samples: deque[ChessReplaySample] = deque()
        self._credited_samples = 0
        self._credited_completed_searches = 0
        self._evicted_samples = 0

    def begin_ingestion(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError('Chess replay capacity must be positive.')
        self.phase = ReplayPhase.INGESTING
        self.capacity = capacity
        self._evict_to_capacity()

    def ingest_game(self, game: ChessCompletedGame) -> int:
        if self.phase is not ReplayPhase.INGESTING:
            raise RuntimeError('Chess replay ingestion is allowed only during the ingestion phase.')
        samples = materialize_chess_game(game)
        self._samples.extend(samples)
        self._credited_samples += len(samples)
        self._credited_completed_searches += sum(
            observation.search_budget for observation in game.observations if observation.sample_eligible
        )
        self._evict_to_capacity()
        return len(samples)

    def freeze(self) -> ChessReplaySnapshot:
        if self.phase is not ReplayPhase.INGESTING:
            raise RuntimeError('Chess replay is already frozen.')
        self.phase = ReplayPhase.FROZEN
        samples = tuple(self._samples)
        return ChessReplaySnapshot(
            samples=samples,
            credited_samples=self._credited_samples,
            credited_completed_searches=self._credited_completed_searches,
            sampler_seed=self.sampler_seed,
            frozen_at_seconds=time.time(),
            evicted_samples=self._evicted_samples,
            estimated_sample_bytes=sum(len(sample.encoded_state) + len(sample.visits) * 8 + 64 for sample in samples),
        )

    def metrics(self, measured_at_seconds: float) -> ChessReplayMetrics:
        samples = tuple(self._samples)
        generations = tuple(sample.source_model_generation for sample in samples)
        ages = tuple(max(0.0, measured_at_seconds - sample.source_created_at_seconds) for sample in samples)
        estimated_sample_bytes = sum(len(sample.encoded_state) + len(sample.visits) * 8 + 64 for sample in samples)
        return ChessReplayMetrics(
            credited_samples=self._credited_samples,
            credited_completed_searches=self._credited_completed_searches,
            live_samples=len(samples),
            evicted_samples=self._evicted_samples,
            oldest_source_model_generation=min(generations) if generations else None,
            newest_source_model_generation=max(generations) if generations else None,
            mean_source_model_generation=float(np.mean(generations)) if generations else None,
            oldest_sample_age_seconds=max(ages) if ages else None,
            mean_sample_age_seconds=float(np.mean(ages)) if ages else None,
            estimated_sample_bytes=estimated_sample_bytes,
        )

    def _evict_to_capacity(self) -> None:
        while len(self._samples) > self.capacity:
            self._samples.popleft()
            self._evicted_samples += 1


class ChessReplayMaintainer:
    def __init__(self, run_path: Path, capacity: int, sampler_seed: int) -> None:
        self.run_path = run_path
        self.inbox_path = run_path / 'completed-games' / 'inbox'
        self.archive_path = run_path / 'completed-games' / 'archive'
        self.replay = ChessReplay(capacity, sampler_seed)
        self._archived_payloads: dict[str, bytes] = {}
        self._latest_model_generation = -1
        self._recover_and_rebuild()

    def maintain(self, capacity: int) -> tuple[ChessReplaySnapshot, ChessReplayMetrics]:
        self.replay.begin_ingestion(capacity)
        for inbox_file in sorted(self.inbox_path.glob('*.json')):
            game = completed_game_from_path(inbox_file)
            payload = canonical_game_payload(game)
            archived_payload = self._archived_payloads.get(game.identity.archive_key)
            if archived_payload is None:
                if game.model_generation < self._latest_model_generation:
                    raise ValueError('Completed games cannot arrive after a newer model-generation archive.')
                append_chess_archive_record(self.archive_file(game.model_generation), payload)
                self._archived_payloads[game.identity.archive_key] = payload
                self.replay.ingest_game(game)
                self._latest_model_generation = game.model_generation
            elif archived_payload != payload:
                raise ValueError(f'Archived completed game has conflicting identity {game.identity.archive_key}.')
            inbox_file.unlink()
            fsync_directory(inbox_file.parent)
        snapshot = self.replay.freeze()
        return snapshot, self.replay.metrics(snapshot.frozen_at_seconds)

    def archive_file(self, model_generation: int) -> Path:
        return self.archive_path / f'model-generation-{model_generation:020d}.games'

    def _recover_and_rebuild(self) -> None:
        for archive_file in sorted(self.archive_path.glob(_ARCHIVE_FILE_PATTERN)):
            for game in read_chess_archive(archive_file, recover_incomplete=True):
                expected_archive = self.archive_file(game.model_generation)
                if archive_file != expected_archive:
                    raise ValueError(f'Completed game is stored in the wrong model-generation archive: {archive_file}')
                payload = canonical_game_payload(game)
                previous = self._archived_payloads.setdefault(game.identity.archive_key, payload)
                if previous != payload:
                    raise ValueError(f'Archive contains conflicting game identity {game.identity.archive_key}.')
                self.replay.ingest_game(game)
                self._latest_model_generation = game.model_generation


def canonical_game_payload(game: ChessCompletedGame) -> bytes:
    return game.model_dump_json().encode('utf-8')


def append_chess_archive_record(path: Path, payload: bytes) -> None:
    if not path.exists():
        write_bytes_atomically(path, CHESS_ARCHIVE_HEADER)
    read_chess_archive(path, recover_incomplete=True)
    frame = _FRAME_HEADER.pack(len(payload), hashlib.sha256(payload).digest()) + payload
    with path.open('ab') as archive:
        archive.write(frame)
        archive.flush()
        os.fsync(archive.fileno())


def read_chess_archive(path: Path, recover_incomplete: bool = False) -> tuple[ChessCompletedGame, ...]:
    if not path.is_file():
        raise ValueError(f'Chess archive does not exist: {path}')
    records: list[ChessCompletedGame] = []
    mode = 'r+b' if recover_incomplete else 'rb'
    with path.open(mode) as archive:
        header = archive.read(len(CHESS_ARCHIVE_HEADER))
        if header != CHESS_ARCHIVE_HEADER:
            raise ValueError(f'Unsupported chess archive header: {path}')
        valid_end = len(CHESS_ARCHIVE_HEADER)
        while True:
            frame_header = archive.read(_FRAME_HEADER.size)
            if not frame_header:
                break
            if len(frame_header) < _FRAME_HEADER.size:
                if not recover_incomplete:
                    raise ValueError(f'Chess archive has an incomplete final frame header: {path}')
                archive.truncate(valid_end)
                break
            payload_length, expected_digest = _FRAME_HEADER.unpack(frame_header)
            payload = archive.read(payload_length)
            if len(payload) < payload_length:
                if not recover_incomplete:
                    raise ValueError(f'Chess archive has an incomplete final frame payload: {path}')
                archive.truncate(valid_end)
                break
            if hashlib.sha256(payload).digest() != expected_digest:
                raise ValueError(f'Chess archive frame checksum failed: {path}')
            records.append(ChessCompletedGame.model_validate_json(payload))
            valid_end = archive.tell()
        if recover_incomplete:
            archive.flush()
            os.fsync(archive.fileno())
    return tuple(records)


def inspect_chess_archives(run_path: Path) -> tuple[ChessArchiveInspection, ...]:
    archive_path = run_path / 'completed-games' / 'archive'
    inspections: list[ChessArchiveInspection] = []
    for path in sorted(archive_path.glob(_ARCHIVE_FILE_PATTERN)):
        records = read_chess_archive(path)
        generations = {record.model_generation for record in records}
        if len(generations) != 1:
            raise ValueError(f'Chess archive mixes model generations: {path}')
        model_generation = generations.pop()
        inspections.append(
            ChessArchiveInspection(
                path=path,
                model_generation=model_generation,
                game_count=len(records),
                eligible_sample_count=sum(
                    observation.sample_eligible for record in records for observation in record.observations
                ),
                completed_searches=sum(
                    observation.search_budget
                    for record in records
                    for observation in record.observations
                    if observation.sample_eligible
                ),
                byte_count=path.stat().st_size,
            )
        )
    return tuple(inspections)


def rebuild_chess_replay(run_path: Path, capacity: int, sampler_seed: int) -> ChessReplaySnapshot:
    maintainer = ChessReplayMaintainer(run_path, capacity, sampler_seed)
    snapshot, _ = maintainer.maintain(capacity)
    return snapshot


def materialize_chess_game(game: ChessCompletedGame) -> tuple[ChessReplaySample, ...]:
    board = ChessBoard.from_fen(game.initial_fen)
    observations_by_ply = {observation.ply: observation for observation in game.observations}
    materialized_by_ply: dict[int, ChessReplaySample] = {}
    moves: list[str] = []
    for ply, move_uci in enumerate(game.moves_uci):
        observation = observations_by_ply.get(ply)
        if observation is not None:
            legal_actions = tuple(sorted(CurrentGame.encode_move(move, board) for move in board.get_valid_moves()))
            if legal_actions != observation.legal_action_ids:
                raise ValueError(f'Completed-game legal actions disagree at ply {ply}.')
            selected_move = chess.Move.from_uci(move_uci)
            if selected_move not in board.get_valid_moves():
                raise ValueError(f'Completed game contains illegal move {move_uci} at ply {ply}.')
            if CurrentGame.encode_move(selected_move, board) != observation.selected_action_id:
                raise ValueError(f'Completed-game selected action disagrees at ply {ply}.')
            if observation.sample_eligible:
                visits = tuple(
                    (visit.action_id, visit.visit_count - observation.minimum_visit_count)
                    for visit in observation.visits
                    if visit.visit_count > observation.minimum_visit_count
                )
                if not visits:
                    raise ValueError(f'Eligible chess observation has no visits after filtering at ply {ply}.')
                canonical_state = CurrentGame.get_canonical_board(board).astype(np.int8, copy=False)
                current_piece_count, opponent_piece_count = CurrentGame.replay_piece_counts(canonical_state)
                final_score = outcome_from_sample_perspective(
                    game.final_score,
                    final_current_player=game.final_current_player,
                    sample_current_player=board.current_player,
                )
                materialized_by_ply[ply] = ChessReplaySample(
                    encoded_state=encode_board_state(canonical_state),
                    visits=visits,
                    value_target=ReplayValueTarget.from_scores(
                        final_score=final_score,
                        mcts_root_value=observation.root_value,
                        termination_reason=game.termination_reason,
                    ),
                    metadata=ReplaySampleMetadata(
                        ply=ply,
                        current_player_piece_count=current_piece_count,
                        opponent_piece_count=opponent_piece_count,
                        starting_fen=game.initial_fen,
                        moves_uci=tuple(moves),
                    ),
                    sample_weight=observation.sample_weight,
                    source_model_generation=observation.model_generation,
                    source_created_at_seconds=game.created_at_seconds,
                )
        move = chess.Move.from_uci(move_uci)
        if move not in board.get_valid_moves():
            raise ValueError(f'Completed game contains illegal move {move_uci} at ply {ply}.')
        board.make_move(move)
        moves.append(move_uci)
    missing_observations = set(observations_by_ply) - set(materialized_by_ply)
    ineligible_plies = {observation.ply for observation in game.observations if not observation.sample_eligible}
    if missing_observations - ineligible_plies:
        raise ValueError('Completed game contains observations that were not materialized.')
    return tuple(materialized_by_ply[ply] for ply in sorted(materialized_by_ply, reverse=True))


def build_chess_training_batch(
    snapshot: ChessReplaySnapshot,
    sample_indices: Sequence[int],
    global_step: int,
    rank: int,
) -> TrainingBatch:
    samples = tuple(snapshot.samples[index] for index in sample_indices)
    batch_size = len(samples)
    states = np.empty((batch_size, *CurrentGame.representation_shape), dtype=np.float32)
    decode_board_states_into(tuple(sample.encoded_state for sample in samples), states)
    policies = np.zeros((batch_size, CurrentGame.action_size), dtype=np.float32)
    for row, sample in enumerate(samples):
        actions = np.fromiter((action for action, _ in sample.visits), dtype=np.int64)
        counts = np.fromiter((count for _, count in sample.visits), dtype=np.float32)
        policies[row, actions] = counts / counts.sum()
    mirrored = np.fromiter(
        (sample_is_mirrored(snapshot.sampler_seed, global_step, rank, position) for position in range(batch_size)),
        dtype=np.bool_,
        count=batch_size,
    )
    states[mirrored] = np.flip(states[mirrored], axis=3)
    mirrored_policies = policies[mirrored].copy()
    policies[mirrored] = 0.0
    mirrored_rows = np.flatnonzero(mirrored)
    policies[mirrored_rows[:, np.newaxis], CHESS_MIRROR_ACTION_MAP[np.newaxis, :]] = mirrored_policies
    return TrainingBatch(
        states=torch.from_numpy(states),
        policy_targets=torch.from_numpy(policies),
        final_outcomes=torch.tensor([int(sample.value_target.final_outcome) for sample in samples]),
        mcts_root_values=torch.tensor([sample.value_target.mcts_root_value for sample in samples]),
        outcome_target_eligible=torch.tensor(
            [sample.value_target.outcome_target_eligible for sample in samples], dtype=torch.bool
        ),
        material_result_scores=torch.tensor(
            [sample.value_target.material_result_score for sample in samples], dtype=torch.float32
        ),
        material_target_eligible=torch.tensor(
            [sample.value_target.material_target_eligible for sample in samples], dtype=torch.bool
        ),
        termination_reasons=torch.tensor([int(sample.value_target.termination_reason) for sample in samples]),
        plies=torch.tensor([sample.metadata.ply for sample in samples], dtype=torch.int32),
        current_player_piece_counts=torch.tensor(
            [sample.metadata.current_player_piece_count for sample in samples], dtype=torch.int8
        ),
        opponent_piece_counts=torch.tensor(
            [sample.metadata.opponent_piece_count for sample in samples], dtype=torch.int8
        ),
        occurrence_counts=torch.tensor([sample.metadata.occurrence_count for sample in samples], dtype=torch.int32),
        sample_weights=torch.tensor([sample.sample_weight for sample in samples], dtype=torch.float32),
    )


def training_batch_loader(
    snapshot: ChessReplaySnapshot,
    global_step: int,
    optimizer_steps: int,
    global_batch_size: int,
    world_size: int,
    rank: int,
    pin_memory: bool,
) -> ChessTrainingBatchLoader:
    return ChessTrainingBatchLoader(
        snapshot,
        global_step,
        optimizer_steps,
        global_batch_size,
        world_size,
        rank,
        pin_memory=pin_memory,
    )


def sample_is_mirrored(sampler_seed: int, global_step: int, rank: int, sample_position: int) -> bool:
    mask = (1 << 64) - 1
    value = (
        (sampler_seed & mask)
        ^ (((global_step + 1) * 0x9E3779B97F4A7C15) & mask)
        ^ (((rank + 1) * 0xBF58476D1CE4E5B9) & mask)
        ^ (((sample_position + 1) * 0x94D049BB133111EB) & mask)
    )
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    value ^= value >> 31
    return bool(value & 1)


def _chess_mirror_action_map() -> np.ndarray:
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
