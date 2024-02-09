from os import environ

environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import pygame  # noqa: E402

from typing import Optional  # noqa: E402

from Framework import ChessBot, Board, Move, Square  # noqa: E402
from Framework.ChessGUI import ChessGUI  # noqa: E402


class HumanPlayer(ChessBot):
    def __init__(self) -> None:
        """Initializes the human player."""
        super().__init__('Human')
        self.gui = ChessGUI()

    def think(self, board: Board) -> Move:
        """Allows a human player to input a move using the GUI."""
        self.gui.draw_board(board)
        self.selected_square = None

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()  # Handle window close as quit event
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if self.selected_square is None:
                        self.first_square_selected(board)
                    else:
                        if possible_move := self.second_square_selected(board):
                            return possible_move

    def first_square_selected(self, board: Board) -> None:
        self.selected_square = self.gui.get_square_from_click()
        selected_piece = board.piece_at(self.selected_square)

        if not selected_piece or selected_piece.color != board.turn:
            self.selected_square = None
        else:
            self.highlight_selected_piece(board, self.selected_square)

    def second_square_selected(self, board: Board) -> Move | None:
        assert self.selected_square is not None, 'No first square selected'

        target_square = self.gui.get_square_from_click()
        move = self.create_move(board, self.selected_square, target_square)

        if not move:
            self.selected_square = None  # Reset selected square
            self.gui.draw_board(board)  # Redraw board if move not valid to clear highlights
            return None

        self.visualize_move(board, move)
        return move

    def create_move(self, board: Board, from_square: Square, to_square: Square) -> Optional[Move]:
        """Attempt to create a move based on selected squares, simplified for clarity."""
        try:
            move = board.find_move(from_square, to_square)
            if move in board.legal_moves:
                return move
        except ValueError:
            pass  # Handle invalid moves
        print('Invalid move. Please try again.')
        return None

    def highlight_selected_piece(self, board: Board, square: Square) -> None:
        self.gui.highlight_square(square, 'green')

        for move in board.legal_moves:
            if move.from_square == self.selected_square:
                self.gui.highlight_square(move.to_square, 'yellow')

    def visualize_move(self, board: Board, move: Move) -> None:
        """Visualize the move on the GUI."""
        board.push(move)
        self.gui.draw_board(board)
        board.pop()
