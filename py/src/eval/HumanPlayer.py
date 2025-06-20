import time
from typing import Optional, Tuple

from src.eval.Bot import Bot
from src.settings import CurrentGameMove, CurrentBoard, CurrentGame, CurrentGameVisuals, CURRENT_GAME

if CURRENT_GAME == 'hex':
    from src.eval.HexGUI import HexGridGameGUI as GUI
else:
    from src.eval.GridGUI import BaseGridGameGUI as GUI


class HumanPlayer(Bot):
    def __init__(self) -> None:
        """Initializes the human player."""
        super().__init__('HumanPlayer', max_time_to_think=0.0)
        _, rows, cols = CurrentGame.representation_shape
        if hasattr(CurrentBoard(), 'board_dimensions'):
            rows, cols = CurrentBoard().board_dimensions  # type: ignore
        self.gui = GUI(rows, cols)
        self.game_visuals = CurrentGameVisuals
        self.selected_cell: Optional[Tuple[int, int]] = None

        self.board_history: list[CurrentBoard] = []

    def think(self, board: CurrentBoard) -> CurrentGameMove:
        # Common input loop for all games
        print('Waiting for human input...')

        self.display_board(board)

        self.board_history.append(board.copy())
        current_displayed_board = len(self.board_history) - 1

        while True:
            events = self.gui.events_occurred()
            if events.quit:
                exit()

            # We check for mouse events:
            if events.clicked:
                if current_displayed_board < len(self.board_history) - 1:
                    # If we are not displaying the most recent board, we go back to it
                    current_displayed_board = len(self.board_history) - 1
                    self.display_board(self.board_history[current_displayed_board])
                    continue

                cell = self.gui.get_cell_from_click()
                if cell is not None:
                    move = self.handle_click(board, cell)
                    if move is not None:
                        # Return the move if successfully formed
                        # Show the move on the GUI before returning
                        new_board = board.copy()
                        new_board.make_move(move)
                        self.display_board(new_board)
                        self.board_history.append(new_board.copy())
                        return move

            if events.left:
                # Go back in history
                if current_displayed_board > 0:
                    current_displayed_board -= 1
                    self.display_board(self.board_history[current_displayed_board])

            if events.right:
                # Go forward in history
                if current_displayed_board < len(self.board_history) - 1:
                    current_displayed_board += 1
                    self.display_board(self.board_history[current_displayed_board])

            time.sleep(0.05)

    def display_board(self, board: CurrentBoard):
        self.gui.clear_highlights_and_redraw(lambda: self.game_visuals.draw_pieces(board, self.gui))  # type: ignore
        self.gui.update_display()

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
                self.display_board(board)
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
                self.display_board(board)
                return None
