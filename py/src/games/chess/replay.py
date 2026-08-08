from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import chess
import numpy as np
import numpy.typing as npt
import torch

from src.games.chess.encoding import decode_board_states_into, encode_board_state, get_board_result_score
from src.games.chess.contract import CHESS_STATE_CONTRACT
from src.games.chess.board import ChessBoard
from src.training.batch import ReplaySampleMetadata, TrainingBatch
from src.games.chess.completed_game import ChessCompletedGame
from src.self_play.completed_game_record import completed_game_from_path
from src.self_play.value_target import ReplayValueTarget, TerminationReason, outcome_from_sample_perspective
from src.training.replay import (
    ARCHIVE_HEADER,
    ArchiveFrameIndex,
    ArchiveInspection,
    PackedReplaySample,
    ReplayGameImplementation,
    ReplayMaintainer,
    ReplayPhase,
    ReplaySnapshot,
    ReplayTrainingBatchLoader,
    append_archive_record,
    canonical_game_payload,
    index_archive,
    pack_visits,
    read_frame_payload,
)


CHESS_ARCHIVE_HEADER = ARCHIVE_HEADER

__all__ = [
    'CHESS_ARCHIVE_HEADER',
    'ChessArchiveFrameIndex',
    'ReplayPhase',
    'append_chess_archive_record',
    'build_chess_training_batch',
    'canonical_game_payload',
    'inspect_chess_archives',
    'materialize_chess_game',
    'pack_chess_visits',
    'read_chess_archive',
    'rebuild_chess_replay',
    'training_batch_loader',
]


def pack_chess_visits(visits: Sequence[tuple[int, int]]) -> npt.NDArray[np.uint16]:
    return pack_visits(visits, CHESS_STATE_CONTRACT.action_size)


ChessArchiveInspection = ArchiveInspection
ChessArchiveFrameIndex = ArchiveFrameIndex


def append_chess_archive_record(path: Path, payload: bytes, ingestion_sequence: int) -> None:
    game = ChessCompletedGame.model_validate_json(payload)
    eligible_sample_count, completed_searches = _chess_archive_counts(game)
    append_archive_record(
        path,
        payload,
        ingestion_sequence,
        game.identity,
        game.model_generation,
        eligible_sample_count,
        completed_searches,
    )


def read_chess_archive(path: Path, recover_incomplete: bool = False) -> tuple[ChessCompletedGame, ...]:
    return tuple(_read_indexed_chess_archive_frame(frame) for frame in index_archive(path, recover_incomplete))


def _chess_archive_counts(game: ChessCompletedGame) -> tuple[int, int]:
    return (
        sum(observation.sample_eligible for observation in game.observations),
        sum(observation.search_budget for observation in game.observations),
    )


def _read_indexed_chess_archive_frame(frame_index: ChessArchiveFrameIndex) -> ChessCompletedGame:
    game = ChessCompletedGame.model_validate_json(read_frame_payload(frame_index))
    if (
        game.identity != frame_index.identity
        or game.model_generation != frame_index.model_generation
        or _chess_archive_counts(game) != (frame_index.eligible_sample_count, frame_index.completed_searches)
    ):
        raise ValueError(f'Archive frame metadata disagrees with its payload: {frame_index.path}')
    return game


def inspect_chess_archives(run_path: Path) -> tuple[ChessArchiveInspection, ...]:
    inspections: list[ChessArchiveInspection] = []
    for path in sorted((run_path / 'completed-games' / 'archive').glob('model-generation-*.games')):
        frame_indexes = index_archive(path, recover_incomplete=False)
        for frame_index in frame_indexes:
            _read_indexed_chess_archive_frame(frame_index)
        generations = {frame_index.model_generation for frame_index in frame_indexes}
        if len(generations) != 1:
            raise ValueError(f'Chess archive mixes model generations: {path}')
        inspections.append(
            ChessArchiveInspection(
                path=path,
                model_generation=generations.pop(),
                game_count=len(frame_indexes),
                eligible_sample_count=sum(frame.eligible_sample_count for frame in frame_indexes),
                completed_searches=sum(frame.completed_searches for frame in frame_indexes),
                byte_count=path.stat().st_size,
            )
        )
    return tuple(inspections)


def rebuild_chess_replay(run_path: Path, capacity: int, sampler_seed: int) -> ReplaySnapshot:
    maintainer = ReplayMaintainer(run_path, CHESS_REPLAY_IMPLEMENTATION, capacity, sampler_seed)
    snapshot, _ = maintainer.maintain(capacity)
    return snapshot


def materialize_chess_game(game: ChessCompletedGame) -> tuple[PackedReplaySample, ...]:
    board = ChessBoard.from_fen(game.initial_fen)
    observations_by_ply = {observation.ply: observation for observation in game.observations}
    materialized_by_ply: dict[int, PackedReplaySample] = {}
    for ply in range(len(game.moves_uci) + 1):
        observation = observations_by_ply.get(ply)
        if observation is not None:
            legal_actions = tuple(
                sorted(CHESS_STATE_CONTRACT.encode_move(move, board) for move in board.get_valid_moves())
            )
            if legal_actions != observation.legal_action_ids:
                raise ValueError(f'Completed-game legal actions disagree at ply {ply}.')
            if ply < len(game.moves_uci):
                selected_move = chess.Move.from_uci(game.moves_uci[ply])
                if CHESS_STATE_CONTRACT.encode_move(selected_move, board) != observation.selected_action_id:
                    raise ValueError(f'Completed-game selected action disagrees at ply {ply}.')
            if observation.sample_eligible:
                visits = tuple(
                    (visit.action_id, visit.visit_count - observation.minimum_visit_count)
                    for visit in observation.visits
                    if visit.visit_count > observation.minimum_visit_count
                )
                if not visits:
                    raise ValueError(f'Eligible chess observation has no visits after filtering at ply {ply}.')
                canonical_state = CHESS_STATE_CONTRACT.canonical_board(board).astype(np.int8, copy=False)
                current_piece_count, opponent_piece_count = CHESS_STATE_CONTRACT.replay_piece_counts(canonical_state)
                final_score = outcome_from_sample_perspective(
                    game.final_score,
                    final_current_player=game.final_current_player,
                    sample_current_player=board.current_player,
                )
                materialized_by_ply[ply] = PackedReplaySample(
                    encoded_state=encode_board_state(canonical_state),
                    visits=pack_chess_visits(visits),
                    value_target=ReplayValueTarget.from_scores(
                        final_score=final_score,
                        mcts_root_value=observation.root_value,
                        termination_reason=game.termination_reason,
                    ),
                    metadata=ReplaySampleMetadata(
                        ply=ply,
                        current_player_piece_count=current_piece_count,
                        opponent_piece_count=opponent_piece_count,
                    ),
                    sample_weight=observation.sample_weight,
                    source_model_generation=observation.model_generation,
                    source_created_at_seconds=game.created_at_seconds,
                )
        if ply == len(game.moves_uci):
            break
        move_uci = game.moves_uci[ply]
        move = chess.Move.from_uci(move_uci)
        if move not in board.get_valid_moves():
            raise ValueError(f'Completed game contains illegal move {move_uci} at ply {ply}.')
        board.make_move(move)
    if board.current_player != game.final_current_player:
        raise ValueError('Completed-game final player does not match reconstructed moves.')
    if game.termination_reason is TerminationReason.NATURAL:
        natural_result = get_board_result_score(board)
        if natural_result is None or natural_result != game.final_score:
            raise ValueError('Completed-game natural result does not match reconstructed moves.')
    missing_observations = set(observations_by_ply) - set(materialized_by_ply)
    ineligible_plies = {observation.ply for observation in game.observations if not observation.sample_eligible}
    if missing_observations - ineligible_plies:
        raise ValueError('Completed game contains observations that were not materialized.')
    return tuple(materialized_by_ply[ply] for ply in sorted(materialized_by_ply, reverse=True))


def build_chess_training_batch(
    snapshot: ReplaySnapshot,
    sample_indices: Sequence[int],
    global_step: int,
    rank: int,
    sample_position_offset: int = 0,
) -> TrainingBatch:
    samples = tuple(snapshot.samples[index] for index in sample_indices)
    batch_size = len(samples)
    states = np.empty(
        (
            batch_size,
            CHESS_STATE_CONTRACT.representation.channels,
            CHESS_STATE_CONTRACT.representation.rows,
            CHESS_STATE_CONTRACT.representation.columns,
        ),
        dtype=np.float32,
    )
    decode_board_states_into(tuple(sample.encoded_state for sample in samples), states)
    policies = np.zeros((batch_size, CHESS_STATE_CONTRACT.action_size), dtype=np.float32)
    for row, sample in enumerate(samples):
        actions = sample.visits[:, 0].astype(np.int64, copy=False)
        counts = sample.visits[:, 1].astype(np.float32, copy=False)
        policies[row, actions] = counts / counts.sum()
    mirrored = np.fromiter(
        (
            sample_is_mirrored(
                snapshot.sampler_seed,
                global_step,
                rank,
                sample_position_offset + position,
            )
            for position in range(batch_size)
        ),
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
    snapshot: ReplaySnapshot,
    global_step: int,
    optimizer_steps: int,
    global_batch_size: int,
    world_size: int,
    rank: int,
    pin_memory: bool,
) -> ReplayTrainingBatchLoader[ChessCompletedGame]:
    return ReplayTrainingBatchLoader(
        CHESS_REPLAY_IMPLEMENTATION,
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
    return np.fromiter(
        (
            CHESS_STATE_CONTRACT.transform_action_id(action_id, 1)
            for action_id in range(CHESS_STATE_CONTRACT.action_size)
        ),
        dtype=np.int64,
        count=CHESS_STATE_CONTRACT.action_size,
    )


CHESS_MIRROR_ACTION_MAP = _chess_mirror_action_map()


class ChessReplayImplementation(ReplayGameImplementation[ChessCompletedGame]):
    @property
    def name(self) -> str:
        return 'chess'

    @property
    def action_size(self) -> int:
        return CHESS_STATE_CONTRACT.action_size

    def parse_file(self, path: Path) -> ChessCompletedGame:
        game = completed_game_from_path(path)
        if not isinstance(game, ChessCompletedGame):
            raise ValueError(f'Expected a chess completed game: {path}')
        return game

    def parse_payload(self, payload: bytes) -> ChessCompletedGame:
        return ChessCompletedGame.model_validate_json(payload)

    def model_generation(self, game: ChessCompletedGame) -> int:
        return game.model_generation

    def archive_counts(self, game: ChessCompletedGame) -> tuple[int, int]:
        return _chess_archive_counts(game)

    def materialize(self, game: ChessCompletedGame) -> tuple[PackedReplaySample, ...]:
        return materialize_chess_game(game)

    def build_batch(
        self,
        snapshot: ReplaySnapshot,
        sample_indices: Sequence[int],
        global_step: int,
        rank: int,
        sample_position_offset: int,
    ) -> TrainingBatch:
        return build_chess_training_batch(
            snapshot,
            sample_indices,
            global_step,
            rank,
            sample_position_offset,
        )


CHESS_REPLAY_IMPLEMENTATION = ChessReplayImplementation()
