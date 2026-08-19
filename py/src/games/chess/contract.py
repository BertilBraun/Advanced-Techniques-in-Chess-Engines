from __future__ import annotations

from typing import Protocol

import numpy as np
from src.games.contracts import GameStateContract, Player, WdlTarget
from src.games.representation import (
    NetworkDimensions,
    PackedPlaneLayout,
    PackedPlanePayload,
    RepresentationDimensions,
    decode_packed_planes,
    encode_packed_planes,
)
from src.replay.contracts import (
    EligibleLegalMovesTarget,
    EligibleNextPolicyTarget,
    EligibleRemainingGameLengthTarget,
    EligibleScalarAuxiliaryTarget,
    ReplaySample,
    SparsePolicyTarget,
)
from src.self_play.completed_game import SearchVisitCounts, TerminationReason


class ChessPosition(Protocol):
    @property
    def fen(self) -> str: ...

    @property
    def current_player(self) -> int: ...

    @property
    def is_terminal(self) -> bool: ...

    def legal_actions(self) -> list[int]: ...

    def action_uci(self, action_id: int) -> str: ...

    def action_id_from_uci(self, move_uci: str) -> int: ...

    def child(self, action_id: int) -> ChessPosition: ...

    def terminal_value(self) -> float: ...

    def approximate_result_score(self) -> float: ...

    def packed_encoding(self) -> bytes: ...


class ChessStateContract(GameStateContract[ChessPosition]):
    def __init__(self) -> None:
        self._representation = RepresentationDimensions(
            channels=29,
            rows=8,
            columns=8,
            binary_channels=tuple(range(22)),
            scalar_channels=tuple(range(22, 29)),
            packed_planes=PackedPlaneLayout(
                board_size=8,
                binary_plane_count=22,
                scalar_count=7,
            ),
        )

    @property
    def name(self) -> str:
        return 'chess'

    @property
    def action_size(self) -> int:
        return 4864

    @property
    def maximum_legal_action_count(self) -> int:
        return 218

    @property
    def representation(self) -> RepresentationDimensions:
        return self._representation

    def initial_position(self) -> ChessPosition:
        from AlphaZeroCpp import ChessPosition

        return ChessPosition()

    def legal_action_ids(self, position: ChessPosition) -> tuple[int, ...]:
        return tuple(sorted(position.legal_actions()))

    def child_position(self, position: ChessPosition, action_id: int) -> ChessPosition:
        return position.child(action_id)

    def is_irreversible_transition(
        self,
        position: ChessPosition,
        action_id: int,
        child: ChessPosition,
    ) -> bool:
        move = position.action_uci(action_id)
        piece = _piece_at(position.fen, move[:2])
        capture = _piece_at(position.fen, move[2:4]) is not None
        castling_changed = position.fen.split()[2] != child.fen.split()[2]
        return piece is not None and (piece.lower() == 'p' or capture or castling_changed)

    def current_player(self, position: ChessPosition) -> Player:
        return Player(position.current_player)

    def natural_terminal_wdl(self, position: ChessPosition) -> WdlTarget | None:
        if not position.is_terminal:
            return None
        return WdlTarget.from_scalar(position.terminal_value())

    def adjudicated_wdl(self, position: ChessPosition, reason: TerminationReason) -> WdlTarget:
        if reason not in {TerminationReason.MAXIMUM_PLIES, TerminationReason.ADJUDICATION}:
            raise ValueError(f'Chess cannot adjudicate termination reason {reason.value}.')
        score = position.approximate_result_score() * position.current_player
        return WdlTarget.from_scalar(score)

    def encode_network_input(self, position: ChessPosition) -> PackedPlanePayload:
        return self.packed_plane_layout.value(bytes(position.packed_encoding()))

    @property
    def augmentation_count(self) -> int:
        return 2

    def transform_action_id(self, action_id: int, augmentation_index: int) -> int:
        if not 0 <= augmentation_index < self.augmentation_count:
            raise ValueError('Chess augmentation index is outside the fixed layout.')
        if augmentation_index == 0:
            return action_id
        from AlphaZeroCpp import mirror_chess_action_id

        return mirror_chess_action_id(action_id)

    def transform_encoded_state(
        self,
        encoded_state: PackedPlanePayload,
        augmentation_index: int,
    ) -> PackedPlanePayload:
        if not 0 <= augmentation_index < self.augmentation_count:
            raise ValueError('Chess augmentation index is outside the fixed layout.')
        if augmentation_index == 0:
            return encoded_state
        representation = self.representation
        state = decode_packed_planes(
            encoded_state,
            representation.packed_planes,
            representation.binary_channels,
            representation.scalar_channels,
        )
        return encode_packed_planes(
            np.flip(state, axis=2),
            representation.packed_planes,
            representation.binary_channels,
            representation.scalar_channels,
        )

    def transform_replay_targets(self, sample: ReplaySample, augmentation_index: int) -> ReplaySample:
        def transform_policy(policy: SparsePolicyTarget) -> SparsePolicyTarget:
            return SparsePolicyTarget(
                visits=SearchVisitCounts(
                    action_ids=tuple(
                        self.transform_action_id(action_id, augmentation_index)
                        for action_id in policy.visits.action_ids
                    ),
                    visit_counts=policy.visits.visit_counts,
                ),
                legal_action_ids=tuple(
                    self.transform_action_id(action_id, augmentation_index) for action_id in policy.legal_action_ids
                ),
            )

        transformed_auxiliary = []
        for target in sample.auxiliary_targets:
            match target:
                case EligibleNextPolicyTarget(policy=policy):
                    transformed_auxiliary.append(EligibleNextPolicyTarget(policy=transform_policy(policy)))
                case EligibleRemainingGameLengthTarget() | EligibleScalarAuxiliaryTarget() | EligibleLegalMovesTarget():
                    transformed_auxiliary.append(target)
                case _:
                    transformed_auxiliary.append(target)
        return ReplaySample(
            encoded_state=self.transform_encoded_state(sample.encoded_state, augmentation_index),
            policy=transform_policy(sample.policy),
            wdl_target=sample.wdl_target,
            root_value=sample.root_value,
            auxiliary_targets=tuple(transformed_auxiliary),
            sample_weight=sample.sample_weight,
            source_model_generation=sample.source_model_generation,
            source_created_at_seconds=sample.source_created_at_seconds,
        )


CHESS_STATE_CONTRACT = ChessStateContract()
CHESS_NETWORK_DIMENSIONS = NetworkDimensions(
    channels=CHESS_STATE_CONTRACT.representation.channels,
    rows=CHESS_STATE_CONTRACT.representation.rows,
    columns=CHESS_STATE_CONTRACT.representation.columns,
    actions=CHESS_STATE_CONTRACT.action_size,
)


def _piece_at(fen: str, square: str) -> str | None:
    file_index = ord(square[0]) - ord('a')
    rank_index = 8 - int(square[1])
    file_cursor = 0
    for token in fen.split()[0].split('/')[rank_index]:
        if token.isdigit():
            file_cursor += int(token)
            continue
        if file_cursor == file_index:
            return token
        file_cursor += 1
    return None
