from typing import List, Optional, Tuple

import chess
from src.games.GUI import BaseGridGameGUI
from src.games.GameVisuals import GameVisuals
from src.games.chess.ChessBoard import ChessBoard, ChessMove
from src.games.chess.ChessGame import BOARD_LENGTH


class ChessVisuals(GameVisuals[ChessMove]):
    # TODO specify input of more complex moves like castling, promotion, etc.
    def draw_pieces(self, board: ChessBoard, gui: BaseGridGameGUI) -> None:
        piece_map = board.board.piece_map()
        for square, piece in piece_map.items():
            row, col = divmod(square, BOARD_LENGTH)
            color = 'black' if piece.color == chess.BLACK else 'white'
            gui.draw_text(row, col, piece.unicode_symbol(), color)

    def is_two_click_game(self) -> bool:
        return False

    def get_moves_from_square(self, board: ChessBoard, row: int, col: int) -> List[Tuple[int, int]]:
        square = self._decode_square((row, col))
        moves = [move.to_square for move in board.get_valid_moves() if move.from_square == square]
        return [self._encode_square(to_square) for to_square in moves]

    def try_make_move(
        self,
        board: ChessBoard,
        from_cell: Optional[Tuple[int, int]],
        to_cell: Tuple[int, int],
    ) -> Optional[ChessMove]:
        assert from_cell is not None, 'from_cell should not be None'

        from_square = self._decode_square(from_cell)
        to_square = self._decode_square(to_cell)

        move = chess.Move(from_square, to_square)
        if move in board.get_valid_moves():
            return move
        assert False, f'Invalid move: {move}. Not in legal moves: {list(board.board.legal_moves)}'

    def _decode_square(self, cell: tuple[int, int]) -> int:
        return (BOARD_LENGTH - 1 - cell[0]) * BOARD_LENGTH + cell[1]

    def _encode_square(self, square: int) -> tuple[int, int]:
        return (BOARD_LENGTH - 1 - (square // BOARD_LENGTH), square % BOARD_LENGTH)
