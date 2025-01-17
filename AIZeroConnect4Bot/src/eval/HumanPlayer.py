import time
from typing import Optional, Tuple

from src.eval.Bot import Bot
from src.eval.GUI import BaseGridGameGUI
from src.games.GameVisuals import GameVisuals
from src.settings import CurrentGameMove, CurrentBoard


class HumanPlayer(Bot):
    def __init__(self, gui: BaseGridGameGUI, game_visuals: GameVisuals) -> None:
        """Initializes the human player."""
        super().__init__('HumanPlayer', max_time_to_think=0.0)
        self.gui = gui
        self.game_visuals = game_visuals
        self.selected_cell: Optional[Tuple[int, int]] = None

    async def think(self, board: CurrentBoard) -> CurrentGameMove:
        # Common input loop for all games
        self.gui.clear_highlights_and_redraw(lambda: self.game_visuals.draw_pieces(board, self.gui))

        while True:
            clicked, quit = self.gui.events_occurred()
            if quit:
                exit()

            # We check for mouse events:
            if clicked:
                cell = self.gui.get_cell_from_click()
                if cell is not None:
                    move = self.handle_click(board, cell)
                    if move is not None:
                        # Return the move if successfully formed
                        # Show the move on the GUI before returning
                        new_board = board.copy()
                        new_board.make_move(move)
                        self.gui.clear_highlights_and_redraw(lambda: self.game_visuals.draw_pieces(new_board, self.gui))
                        return move

            time.sleep(0.2)

    def handle_click(self, board: CurrentBoard, cell: Tuple[int, int]) -> Optional[CurrentGameMove]:
        # If game is one-click move (like Connect4), we try to form a move immediately
        if not self.game_visuals.is_two_click_game():
            # Single-click logic: just try to form a move
            return self.game_visuals.try_make_move(board, None, cell)

        # If two-click game (Chess/Checkers):
        if self.selected_cell is None:
            # First click: select a piece if valid
            moves_from_cell = self.game_visuals.get_moves_from_square(board, *cell)
            if moves_from_cell:
                # Highlight selection and possible moves
                self.selected_cell = cell
                self.gui.clear_highlights_and_redraw(lambda: self.game_visuals.draw_pieces(board, self.gui))
                self.gui.highlight_cell(cell[0], cell[1], 'green')
                for to_row, to_col in moves_from_cell:
                    self.gui.highlight_cell(to_row, to_col, 'yellow')
                self.gui.update_display()
            return None
        else:
            # Second click: try to form a move
            from_cell = self.selected_cell
            self.selected_cell = None
            to_cell = cell
            move = self.game_visuals.try_make_move(board, from_cell, to_cell)
            if move:
                return move
            else:
                # Invalid move, reset and redraw
                self.gui.clear_highlights_and_redraw(lambda: self.game_visuals.draw_pieces(board, self.gui))
                return None
