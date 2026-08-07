from __future__ import annotations

import numpy as np
import numpy.typing as npt

from src.games.chess.ChessBoard import ChessBoard, ChessMove
from src.games.chess.ChessGame import ChessGame
from src.games.contracts import GameStateContract, RepresentationDimensions
from src.neural_network import NetworkDimensions
from src.packed_planes import PackedPlaneLayout


class ChessStateContract(GameStateContract):
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

    def create_initial_board(self) -> ChessBoard:
        return self.game.get_initial_board()

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
