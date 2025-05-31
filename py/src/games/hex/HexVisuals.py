from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from src.eval.HexGUI import HexGridGameGUI
from src.games.GameVisuals import GameVisuals
from src.games.hex.HexBoard import HexBoard, SIZE
from src.games.Game import Board
from src.games.hex.HexGame import HexGame
from src.mcts.MCTSNode import MCTSNode, ucb
from src.util import lerp


class HexVisuals(GameVisuals[int]):
    def draw_pieces(self, board: HexBoard, gui: HexGridGameGUI) -> None:
        # Draw X for player 1 and O for player -1
        for r in range(SIZE):
            for c in range(SIZE):
                cell = board.board[r, c]
                if cell == 1:
                    gui.draw_hex_cell(r, c, 'blue')
                elif cell == -1:
                    gui.draw_hex_cell(r, c, 'green')

    def is_two_click_game(self) -> bool:
        # Single click selects destination only
        return False

    def get_moves_from_square(self, board: Board[int], row: int, col: int) -> List[Tuple[int, int]]:
        # Hex is single-click game, not two-click
        assert False, 'Hex is not a two-click game'

    def try_make_move(
        self, board: Board[int], from_cell: Optional[Tuple[int, int]], to_cell: Tuple[int, int]
    ) -> Optional[int]:
        # from_cell ignored
        r, c = to_cell
        idx = r * SIZE + c
        if idx in board.get_valid_moves():
            return idx
        return None


def display_node(
    root: MCTSNode,
    inspect_or_search: bool,
    search_function: Callable[[list[tuple[HexBoard, None]]], Any],
) -> None:
    import time
    from src.games.hex.HexVisuals import HexGridGameGUI
    from src.games.hex.HexGame import SIZE as HEX_SIZE
    from src.settings import CurrentBoard

    assert CurrentBoard == HexBoard, 'Current board must be a HexBoard instance'

    gui = HexGridGameGUI(HEX_SIZE, HEX_SIZE)
    game = HexGame()

    def draw():
        for i in range(HEX_SIZE):
            for j in range(HEX_SIZE):
                gui.draw_hex_cell(i, j, 'lightgrey')

        for i in range(HEX_SIZE):
            for j in range(HEX_SIZE):
                if root.board.board[i, j] == 1:
                    gui.draw_hex_cell(i, j, 'blue')
                elif root.board.board[i, j] == -1:
                    gui.draw_hex_cell(i, j, 'green')
        for i in range(HEX_SIZE):
            for j in range(HEX_SIZE):
                if root.board.board[i, j] == 0:
                    max_policy = max(root.children_number_of_visits)
                    for k, child in enumerate(root.children):
                        if child.encoded_move_to_get_here == game.encode_move(i * HEX_SIZE + j, root.board):
                            # highest policy should be pure red, lowest should be pure white
                            color = lerp(
                                np.array([255, 255, 255]),
                                np.array([255, 0, 0]),
                                root.children_number_of_visits[k] / max_policy,
                            )
                            gui.draw_hex_cell(i, j, color)  # type: ignore
                            gui.draw_text(i, j, f'pol:{root.children_policies[k]:.2f}', offset=(0, -30))
                            total = sum(
                                vis for vis in root.children_number_of_visits if vis >= root.number_of_visits * 0.01
                            )
                            if root.children_number_of_visits[k] >= root.number_of_visits * 0.01:
                                gui.draw_text(
                                    i, j, f'res:{root.children_number_of_visits[k] / total:.2f}', offset=(0, -10)
                                )
                            gui.draw_text(
                                i,
                                j,
                                f'{root.children_result_scores[k]/root.children_number_of_visits[k]:.2f}@{root.children_number_of_visits[k]}',
                                offset=(0, 10),
                            )
                            gui.draw_text(
                                i,
                                j,
                                f'ucb:{ucb(root.children[k], 1.7, root.result_score if not root.parent else 1.0):.2f}',
                                offset=(0, 30),
                            )
                            txt = ''
                            if root.children[k].is_fully_expanded:
                                txt = 'F'
                            if root.children[k].board and root.children[k].is_terminal_node:
                                txt += 'T'
                            gui.draw_text(i, j, txt, offset=(0, 50))
                            break
        gui.update_window_title(
            f"Current Player: {'blue' if root.board.current_player == 1 else 'green'}@{root.result_score:.2f} MCTS result Score - {'Inspect' if inspect_or_search else 'Search'}"
        )
        gui.update_display()

    print('Sum of all policies:', sum(root.children_policies))
    print('Sum of all visit counts:', sum(root.children_number_of_visits))
    print('Visit counts of root:', root.number_of_visits)
    while True:
        time.sleep(0.2)
        draw()
        events = gui.events_occurred()
        if events.quit:
            exit()
        if events.left:
            return
        if events.right:
            inspect_or_search = not inspect_or_search
        if events.clicked:
            cell = gui.get_cell_from_click()
            if cell is not None:
                move = cell[0] * HEX_SIZE + cell[1]
                print(
                    cell,
                    move,
                    game.encode_move(move, root.board),
                    game.decode_move(game.encode_move(move, root.board), root.board),
                    [child.encoded_move_to_get_here for child in root.children],
                )
                for k, child in enumerate(root.children):
                    if child.encoded_move_to_get_here == game.encode_move(move, root.board):
                        print(
                            f'Child {k}: {child.encoded_move_to_get_here}, {child.number_of_visits}, {child.result_score}'
                        )
                        if inspect_or_search:
                            if child.is_fully_expanded and child.number_of_visits > 1:
                                display_node(child, inspect_or_search, search_function)
                            else:
                                print('Child is not fully expanded')
                        else:
                            search_function([(child.board, None)])
                        break
                else:
                    print('No child found')
