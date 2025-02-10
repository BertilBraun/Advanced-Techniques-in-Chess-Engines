import time

from src.settings import CurrentGame, CurrentGameVisuals
from src.eval.GUI import BaseGridGameGUI

# TODO Copy this move list from tensorboard
move_list = '1:2,3,1,7,5,4,8'  # example move list

result, moves = move_list.split(':', maxsplit=1)
moves = moves.split(',')


def display_board(move_index: int, gui: BaseGridGameGUI):
    board = CurrentGame.get_initial_board()
    for move in moves[:move_index]:
        if move.startswith('FEN'):
            board = CurrentGame.get_initial_board()
            board.set_fen(move[4:-1])
        else:
            board.make_move(CurrentGame.decode_move(int(move)))

    gui.clear_highlights_and_redraw(lambda: CurrentGameVisuals.draw_pieces(board, gui))
    gui.update_display()


def main():
    _, rows, cols = CurrentGame.representation_shape
    if hasattr(CurrentGame.get_initial_board(), 'board_dimensions'):
        rows, cols = CurrentGame.get_initial_board().board_dimensions  # type: ignore
    gui = BaseGridGameGUI(
        rows, cols, title=f'Game Inspector: Press left and right to navigate through the game. Game Result: {result}'
    )

    current_displayed_board = 0

    display_board(current_displayed_board, gui)

    while True:
        events = gui.events_occurred()
        if events.quit:
            exit()

        if events.left:
            # Go back in history
            if current_displayed_board > 0:
                current_displayed_board -= 1
                display_board(current_displayed_board, gui)

        if events.right:
            # Go forward in history
            if current_displayed_board < len(move_list) - 1:
                current_displayed_board += 1
                display_board(current_displayed_board, gui)

        time.sleep(0.2)


if __name__ == '__main__':
    main()
