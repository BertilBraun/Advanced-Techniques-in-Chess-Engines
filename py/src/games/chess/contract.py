from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt
from src.games.contracts import GameStateContract, Player, WdlTarget
from src.games.representation import (
    NetworkDimensions,
    PackedPlaneLayout,
    PackedPlanePayload,
    RepresentationDimensions,
)
from src.self_play.completed_game import TerminationReason

CHESS_ACTION_SIZE = 1880

OWN_KING_SIDE_CASTLING_PLANE = 12
OWN_QUEEN_SIDE_CASTLING_PLANE = 13
OPPONENT_KING_SIDE_CASTLING_PLANE = 14
OPPONENT_QUEEN_SIDE_CASTLING_PLANE = 15


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
        return CHESS_ACTION_SIZE

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

    def transform_decoded_states(
        self,
        states: npt.NDArray[np.float32],
        augmentation_indices: npt.NDArray[np.int64],
    ) -> None:
        if states.ndim != 4 or states.shape[1:] != (29, 8, 8) or len(states) != len(augmentation_indices):
            raise ValueError('Chess decoded states and augmentation indices are not batch-aligned.')
        if np.any((augmentation_indices < 0) | (augmentation_indices >= self.augmentation_count)):
            raise ValueError('Chess augmentation index is outside the fixed layout.')
        mirrored_rows = augmentation_indices == 1
        if not np.any(mirrored_rows):
            return
        mirrored = np.flip(states[mirrored_rows], axis=3)
        mirrored[:, [OWN_KING_SIDE_CASTLING_PLANE, OWN_QUEEN_SIDE_CASTLING_PLANE]] = mirrored[
            :, [OWN_QUEEN_SIDE_CASTLING_PLANE, OWN_KING_SIDE_CASTLING_PLANE]
        ]
        mirrored[:, [OPPONENT_KING_SIDE_CASTLING_PLANE, OPPONENT_QUEEN_SIDE_CASTLING_PLANE]] = mirrored[
            :, [OPPONENT_QUEEN_SIDE_CASTLING_PLANE, OPPONENT_KING_SIDE_CASTLING_PLANE]
        ]
        states[mirrored_rows] = mirrored

    def transform_action_id(self, action_id: int, augmentation_index: int) -> int:
        if not 0 <= augmentation_index < self.augmentation_count:
            raise ValueError('Chess augmentation index is outside the fixed layout.')
        if augmentation_index == 0:
            return action_id
        from AlphaZeroCpp import mirror_chess_action_id

        return mirror_chess_action_id(action_id)


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
