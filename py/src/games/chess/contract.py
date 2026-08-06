from __future__ import annotations

from src.games.chess.ChessBoard import ChessBoard, ChessMove
from src.games.chess.ChessGame import ChessGame
from src.games.contracts import GameStateContract, RepresentationDimensions


_GAME = ChessGame()


CHESS_STATE_CONTRACT = GameStateContract[ChessBoard, ChessMove](
    name='chess',
    action_size=_GAME.action_size,
    representation=RepresentationDimensions(
        channels=_GAME.representation_shape[0],
        rows=_GAME.representation_shape[1],
        columns=_GAME.representation_shape[2],
        binary_channels=_GAME.binary_channels,
        scalar_channels=_GAME.scalar_channels,
    ),
    create_initial_board=_GAME.get_initial_board,
    canonical_board=_GAME.get_canonical_board,
    encode_move=_GAME.encode_move,
    decode_move=_GAME.decode_move,
    replay_piece_counts=_GAME.replay_piece_counts,
)
