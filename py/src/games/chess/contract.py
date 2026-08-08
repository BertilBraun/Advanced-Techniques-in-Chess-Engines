from __future__ import annotations

import numpy as np
import numpy.typing as npt

from src.games.chess.board import MAX_MATERIAL_VALUE, PIECE_VALUE, ChessBoard, ChessMove
from src.games.chess.game import BOARD_LENGTH, ChessGame, DictMove, index_to_square, square_to_index
from src.games.contracts import GameStateContract, Player, RepresentationDimensions, WdlTarget
from src.neural_network import NetworkDimensions
from src.packed_planes import PackedPlaneLayout, PackedPlanePayload, decode_packed_planes, encode_packed_planes


class ChessStateContract(GameStateContract[ChessBoard]):
    def __init__(self) -> None:
        self.game = ChessGame()
        shape = self.game.representation_shape
        self._representation = RepresentationDimensions(
            channels=shape[0],
            rows=shape[1],
            columns=shape[2],
            binary_channels=self.game.binary_channels,
            scalar_channels=self.game.scalar_channels,
            packed_planes=PackedPlaneLayout(
                board_size=shape[1],
                binary_plane_count=len(self.game.binary_channels),
                scalar_count=len(self.game.scalar_channels),
            ),
        )

    @property
    def name(self) -> str:
        return 'chess'

    @property
    def action_size(self) -> int:
        return self.game.action_size

    @property
    def representation(self) -> RepresentationDimensions:
        return self._representation

    def initial_position(self) -> ChessBoard:
        return self.game.get_initial_board()

    def legal_action_ids(self, position: ChessBoard) -> tuple[int, ...]:
        return tuple(sorted(self.game.encode_move(move, position) for move in position.get_valid_moves()))

    def child_position(self, position: ChessBoard, action_id: int) -> ChessBoard:
        child = position.copy()
        child.make_move(self.game.decode_move(action_id, position))
        return child

    def current_player(self, position: ChessBoard) -> Player:
        return Player(position.current_player)

    def terminal_wdl(self, position: ChessBoard) -> WdlTarget | None:
        if not position.is_game_over():
            return None
        winner = position.check_winner()
        if winner is None:
            return WdlTarget(win=0.0, draw=1.0, loss=0.0)
        if winner == position.current_player:
            return WdlTarget(win=1.0, draw=0.0, loss=0.0)
        return WdlTarget(win=0.0, draw=0.0, loss=1.0)

    def adjudicated_wdl(self, position: ChessBoard) -> WdlTarget:
        material_score = 0
        for piece_type, value in PIECE_VALUE.items():
            material_score += value * len(position.board.pieces(piece_type, position.board.turn))
            material_score -= value * len(position.board.pieces(piece_type, not position.board.turn))
        return WdlTarget.from_scalar(material_score / MAX_MATERIAL_VALUE)

    def encode_network_input(self, position: ChessBoard) -> PackedPlanePayload:
        state = self.canonical_board(position).astype(np.int8, copy=False)
        return encode_packed_planes(
            state,
            self.representation.packed_planes,
            self.representation.binary_channels,
            self.representation.scalar_channels,
        )

    @property
    def augmentation_count(self) -> int:
        return 2

    def transform_action_id(self, action_id: int, augmentation_index: int) -> int:
        if not 0 <= augmentation_index < self.augmentation_count:
            raise ValueError('Chess augmentation index is outside the fixed layout.')
        if augmentation_index == 0:
            return action_id
        move = self.game.index2move[action_id]
        from_row, from_column = square_to_index(move.from_square)
        to_row, to_column = square_to_index(move.to_square)
        mirrored = DictMove(
            from_square=index_to_square(from_row, BOARD_LENGTH - 1 - from_column),
            to_square=index_to_square(to_row, BOARD_LENGTH - 1 - to_column),
            promotion=move.promotion,
        )
        return self.game.move2index[mirrored]

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

    def canonical_board(self, board: ChessBoard) -> npt.NDArray[np.int8]:
        return self.game.get_canonical_board(board)

    def encode_move(self, move: ChessMove, board: ChessBoard) -> int:
        return self.game.encode_move(move, board)

    def decode_move(self, action: int, board: ChessBoard) -> ChessMove:
        return self.game.decode_move(action, board)

    def replay_piece_counts(self, state: npt.NDArray[np.int8]) -> tuple[int, int]:
        return self.game.replay_piece_counts(state)


CHESS_STATE_CONTRACT = ChessStateContract()
CHESS_NETWORK_DIMENSIONS = NetworkDimensions(
    channels=CHESS_STATE_CONTRACT.representation.channels,
    rows=CHESS_STATE_CONTRACT.representation.rows,
    columns=CHESS_STATE_CONTRACT.representation.columns,
    actions=CHESS_STATE_CONTRACT.action_size,
)
