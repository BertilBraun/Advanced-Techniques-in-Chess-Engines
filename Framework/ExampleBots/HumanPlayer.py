import pygame

from typing import Optional

from Framework import ChessBot, Board, Move, Square, Piece
from Framework.ChessGUI import ChessGUI

class HumanPlayer(ChessBot):
    def __init__(self) -> None:
        """Initializes the human player."""
        super().__init__("Human")
        self.gui = ChessGUI()
    
    def think(self, board: Board) -> Move:
        """Allows a human player to input a move using the GUI."""
        self.gui.draw_board(board)
        selected_square = None

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()  # Handle window close as quit event
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if selected_square is None:
                        selected_square = self.gui.get_square_from_click()
                        selected_piece = board.piece_at(selected_square)
                        if not selected_piece or selected_piece.color != board.turn:
                            selected_square = None
                        else:
                            self.highlight_selected_piece(board, selected_square)
                    else:
                        target_square = self.gui.get_square_from_click()
                        move = self.create_move(board, selected_square, target_square)
                        if move:
                            return move
                        selected_square = None  # Reset if move not valid
                        self.gui.draw_board(board) # Redraw board if move not valid to clear highlights

    def create_move(self, board: Board, from_square: Square, to_square: Square) -> Optional[Move]:
        """Attempt to create a move based on selected squares, simplified for clarity."""
        try:
            move = board.find_move(from_square, to_square)
            if move in board.legal_moves:
                return move
        except ValueError:
            pass  # Handle invalid moves
        print("Invalid move. Please try again.")
        return None
    
    def highlight_selected_piece(self, board: Board, selected_square: Square) -> None:
        self.gui.highlight_square(selected_square, "green")
            
        for move in board.legal_moves:
            if move.from_square == selected_square:
                self.gui.highlight_square(move.to_square, "yellow")
        