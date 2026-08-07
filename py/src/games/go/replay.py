from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from AlphaZeroCpp import GoPlayer, GoRules

from src.games.go.contract import GoStateContract, GoSymmetryIndex, NativeGoPosition
from src.packed_planes import PackedPlanePayload
from src.training.batch import ReplaySampleMetadata, TrainingBatch
from src.self_play.completed_game_record import completed_game_from_path
from src.games.go.completed_game import GoCompletedGame
from src.self_play.value_target import ReplayValueTarget, TerminationReason, outcome_from_sample_perspective
from src.training.replay import (
    ARCHIVE_HEADER,
    PackedReplaySample,
    ReplayGameImplementation,
    ReplayMaintainer,
    ReplaySnapshot,
    ReplayTrainingBatchLoader,
    append_archive_record,
    index_archive,
    pack_visits,
    read_frame_payload,
)


GO_ARCHIVE_HEADER = ARCHIVE_HEADER


def pack_go_visits(visits: Sequence[tuple[int, int]], action_size: int) -> npt.NDArray[np.uint16]:
    return pack_visits(visits, action_size)


def append_go_archive(path: Path, game: GoCompletedGame, ingestion_sequence: int) -> None:
    payload = game.model_dump_json().encode('utf-8')
    eligible_sample_count, completed_searches = _go_archive_counts(game)
    append_archive_record(
        path,
        payload,
        ingestion_sequence,
        game.identity,
        game.model_generation,
        eligible_sample_count,
        completed_searches,
    )


def read_go_archive(path: Path, recover_incomplete: bool = False) -> tuple[GoCompletedGame, ...]:
    return tuple(
        GoCompletedGame.model_validate_json(read_frame_payload(frame))
        for frame in index_archive(path, recover_incomplete)
    )


def rebuild_go_replay(run_path: Path, contract: GoStateContract, capacity: int, sampler_seed: int) -> ReplaySnapshot:
    maintainer = ReplayMaintainer(run_path, GoReplayImplementation(contract), capacity, sampler_seed)
    snapshot, _ = maintainer.maintain(capacity)
    return snapshot


def _go_archive_counts(game: GoCompletedGame) -> tuple[int, int]:
    eligible = tuple(observation for observation in game.observations if observation.sample_eligible)
    return len(eligible), sum(observation.search_budget for observation in eligible)


def materialize_go_game(game: GoCompletedGame) -> tuple[PackedReplaySample, ...]:
    contract = GoStateContract(game.representation.board_size, game.representation.history_length)
    rules = GoRules(game.rules.komi_half_points, game.rules.maximum_moves)
    position: NativeGoPosition = contract.initial_position(rules)
    observations = {observation.ply: observation for observation in game.observations}
    samples: list[PackedReplaySample] = []
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
                    PackedReplaySample(
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
    contract: GoStateContract,
    snapshot: ReplaySnapshot,
    sample_indices: Sequence[int],
    global_step: int,
    rank: int,
    sample_position_offset: int = 0,
) -> TrainingBatch:
    samples = tuple(snapshot.samples[index] for index in sample_indices)
    batch_size = len(samples)
    states = np.empty(
        (batch_size, contract.channels, contract.board_size, contract.board_size),
        dtype=np.float32,
    )
    policies = np.zeros((batch_size, contract.action_size), dtype=np.float32)
    transformed_states: list[PackedPlanePayload] = []
    for row, sample in enumerate(samples):
        symmetry = _symmetry_index(snapshot.sampler_seed, global_step, rank, sample_position_offset + row)
        transformed_states.append(contract.transform_state(sample.encoded_state, symmetry))
        actions = tuple(contract.transform_action(int(action), symmetry) for action in sample.visits[:, 0])
        counts = sample.visits[:, 1].astype(np.float32, copy=False)
        policies[row, actions] = counts / counts.sum()
    contract.decode_batch_into(tuple(transformed_states), states)
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


class GoReplayImplementation(ReplayGameImplementation[GoCompletedGame]):
    def __init__(self, contract: GoStateContract) -> None:
        self.contract = contract

    @property
    def name(self) -> str:
        return 'go'

    @property
    def action_size(self) -> int:
        return self.contract.action_size

    def parse_file(self, path: Path) -> GoCompletedGame:
        game = completed_game_from_path(path)
        if not isinstance(game, GoCompletedGame):
            raise ValueError(f'Expected a Go completed game: {path}')
        self._validate_representation(game)
        return game

    def parse_payload(self, payload: bytes) -> GoCompletedGame:
        game = GoCompletedGame.model_validate_json(payload)
        self._validate_representation(game)
        return game

    def model_generation(self, game: GoCompletedGame) -> int:
        return game.model_generation

    def archive_counts(self, game: GoCompletedGame) -> tuple[int, int]:
        return _go_archive_counts(game)

    def materialize(self, game: GoCompletedGame) -> tuple[PackedReplaySample, ...]:
        self._validate_representation(game)
        return materialize_go_game(game)

    def build_batch(
        self,
        snapshot: ReplaySnapshot,
        sample_indices: Sequence[int],
        global_step: int,
        rank: int,
        sample_position_offset: int,
    ) -> TrainingBatch:
        return build_go_training_batch(
            self.contract,
            snapshot,
            sample_indices,
            global_step,
            rank,
            sample_position_offset,
        )

    def batch_loader(
        self,
        snapshot: ReplaySnapshot,
        global_step: int,
        optimizer_steps: int,
        global_batch_size: int,
        world_size: int,
        rank: int,
        pin_memory: bool,
    ) -> ReplayTrainingBatchLoader[GoCompletedGame]:
        return ReplayTrainingBatchLoader(
            self,
            snapshot,
            global_step,
            optimizer_steps,
            global_batch_size,
            world_size,
            rank,
            pin_memory,
        )

    def _validate_representation(self, game: GoCompletedGame) -> None:
        if (
            game.representation.board_size != self.contract.board_size
            or game.representation.history_length != self.contract.history_length
        ):
            raise ValueError('Go game representation disagrees with the replay implementation.')
