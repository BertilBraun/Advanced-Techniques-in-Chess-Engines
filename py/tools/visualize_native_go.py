from __future__ import annotations

import argparse
import time


from src.games.go.GoGridGUI import GoGridGUI
from src.games.go.GoVisuals import GoVisuals
from src.games.go.NativeGoBoard import NativeGoBoard


def window_title(board: NativeGoBoard, history_index: int, history_length: int) -> str:
    player = 'Black' if board.current_player == 1 else 'White'
    status = f'{player} to move | move {board.move_number} | position {history_index + 1}/{history_length}'
    if board.ko_point is not None:
        status += f' | ko: {board.ko_point}'
    if board.is_game_over():
        winner = board.check_winner()
        winner_name = 'draw' if winner is None else ('Black' if winner == 1 else 'White')
        status = f'Game over: {winner_name} | {status}'
    return f'Native Go Inspector | {status} | Up: pass | Left/Right: history'


def draw(board: NativeGoBoard, gui: GoGridGUI, visuals: GoVisuals, history_index: int, history_length: int) -> None:
    gui.clear_highlights_and_redraw(lambda: visuals.draw_pieces(board, gui))
    gui.update_window_title(window_title(board, history_index, history_length))


def run(board_size: int, komi_half_points: int, maximum_moves: int | None) -> None:
    board = NativeGoBoard(board_size, komi_half_points, maximum_moves)
    gui = GoGridGUI(board_size)
    visuals = GoVisuals()
    history = [board.copy()]
    history_index = 0
    draw(board, gui, visuals, history_index, len(history))

    while True:
        events = gui.events_occurred()
        if events.quit:
            return
        if events.left and history_index > 0:
            history_index -= 1
            board = history[history_index].copy()
            draw(board, gui, visuals, history_index, len(history))
        if events.right and history_index + 1 < len(history):
            history_index += 1
            board = history[history_index].copy()
            draw(board, gui, visuals, history_index, len(history))

        action: int | None = None
        if history_index == len(history) - 1 and not board.is_game_over():
            if events.clicked:
                cell = gui.get_cell_from_click()
                if cell is not None:
                    action = visuals.try_make_move(board, None, cell)
            elif events.up:
                action = board.pass_action
        if action is not None:
            board.make_move(action)
            history.append(board.copy())
            history_index += 1
            draw(board, gui, visuals, history_index, len(history))
        time.sleep(0.03)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Interactively inspect the standalone native Go rules.')
    parser.add_argument('--board-size', type=int, choices=(7, 9), default=7)
    parser.add_argument('--komi-half-points', type=int, default=15)
    parser.add_argument('--maximum-moves', type=int)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    run(arguments.board_size, arguments.komi_half_points, arguments.maximum_moves)


if __name__ == '__main__':
    main()
