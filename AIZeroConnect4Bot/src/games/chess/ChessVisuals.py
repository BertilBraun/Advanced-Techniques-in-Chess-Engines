from typing import List, Optional, Tuple

from src.eval.GUI import BaseGridGameGUI
from src.games.GameVisuals import GameVisuals
from src.games.chess.ChessBoard import ChessBoard, ChessMove
from src.games.chess.ChessGame import BOARD_LENGTH


class ChessVisuals(GameVisuals[ChessMove]):
    def draw_pieces(self, board: ChessBoard, gui: BaseGridGameGUI) -> None:
        piece_map = board.board.piece_map()
        for square, piece in piece_map.items():
            row, col = self._encode_square(square)
            gui.draw_text(row, col, piece.unicode_symbol())

    def is_two_click_game(self) -> bool:
        return True

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

        moves = [
            move for move in board.get_valid_moves() if move.from_square == from_square and move.to_square == to_square
        ]
        if len(moves) > 1:
            print(f'Multiple moves found for {from_cell} -> {to_cell}')
            for i, move in enumerate(moves):
                print(f'{i}: {move}')
            while True:
                try:
                    index = int(input('Enter the index of the move you want to make: '))
                    return moves[index]
                except ValueError | IndexError:
                    print('Invalid input. Please enter a valid index.')

        return moves[0] if moves else None

    def _decode_square(self, cell: tuple[int, int]) -> int:
        return cell[0] * BOARD_LENGTH + cell[1]

    def _encode_square(self, square: int) -> tuple[int, int]:
        return ((square // BOARD_LENGTH), square % BOARD_LENGTH)
