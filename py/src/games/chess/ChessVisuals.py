from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple, TYPE_CHECKING

from src.util import lerp

if TYPE_CHECKING:
    from src.mcts.MCTSNode import MCTSNode

from src.eval.GridGUI import BaseGridGameGUI
from src.games.GameVisuals import GameVisuals
from src.games.chess.ChessBoard import ChessBoard, ChessMove
from src.games.chess.ChessGame import BOARD_LENGTH


class ChessVisuals(GameVisuals[ChessMove]):
    def draw_pieces(self, board: ChessBoard, gui: BaseGridGameGUI) -> None:
        piece_map = board.board.piece_map()
        for square, piece in piece_map.items():
            row, col = self._encode_square(square)
            gui.draw_text(row, col, piece.unicode_symbol(), font_size=64)

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
        return (BOARD_LENGTH - 1 - cell[0]) * BOARD_LENGTH + cell[1]

    def _encode_square(self, square: int) -> tuple[int, int]:
        return (BOARD_LENGTH - 1 - (square // BOARD_LENGTH), square % BOARD_LENGTH)


def display_node(
    root: MCTSNode,
    inspect_or_search: bool,
    search_function: Callable[[list[tuple[ChessBoard, Any]]], Any],
) -> None:
    import time
    import numpy as np

    from src.eval.GridGUI import BaseGridGameGUI
    from src.games.chess.ChessVisuals import ChessVisuals
    from src.games.chess.ChessGame import BOARD_LENGTH, ChessGame
    from src.mcts.MCTSNode import ucb
    from src.settings import TRAINING_ARGS

    visuals = ChessVisuals()
    gui = BaseGridGameGUI(BOARD_LENGTH, BOARD_LENGTH)
    game = ChessGame()

    # Track whether a "from" square has been clicked; None means no selection yet.
    selected_from_square: int | None = None

    def draw() -> None:
        gui.draw_background()

        # 2) If no square is selected, we aggregate policies by origin-square
        if selected_from_square is None:
            # Build a dictionary: origin_square → sum_of_policies_over_children
            aggregated_policy: dict[int, float] = {}
            aggregated_visits: dict[int, int] = {}
            for k, child in enumerate(root.children):
                mv = game.decode_move(child.encoded_move_to_get_here, root.board)
                origin = mv.from_square
                aggregated_policy[origin] = aggregated_policy.get(origin, 0.0) + root.children_policies[k]
                aggregated_visits[origin] = aggregated_visits.get(origin, 0) + root.children_number_of_visits[k]

            assert aggregated_policy, 'No aggregated policy found; this should not happen'
            assert aggregated_visits, 'No aggregated visits found; this should not happen'

            max_visits = max(aggregated_visits.values())

            # For each origin-square that actually appears in children, draw that square
            for origin_sq, agg_vis in aggregated_visits.items():
                agg_pol = aggregated_policy[origin_sq]
                # Encode to (row, col)
                r0, c0 = visuals._encode_square(origin_sq)
                normalized = agg_vis / max_visits
                # Lerp white→red: white=(255,255,255), red=(255,0,0)
                color_rgb = lerp(np.array([255, 255, 255]), np.array([255, 0, 0]), normalized)
                color_tuple = tuple(color_rgb.astype(int).tolist())

                # Draw the cell with the aggregated policy color
                gui.draw_cell(r0, c0, color_tuple)
                gui.draw_text(r0, c0, f'Σpol:{agg_pol:.2f}', offset=(0, -40))
                gui.draw_text(r0, c0, f'visits:{agg_vis}', offset=(0, 40))

        else:
            # 3) If a "from" square is selected, highlight it in yellow
            r_sel, c_sel = visuals._encode_square(selected_from_square)
            gui.draw_cell(r_sel, c_sel, 'yellow')

            # Then, for each child whose move.from_square matches selected_from_square,
            # draw the child's destination square with detailed per-child info.
            matching_indices = [
                k
                for k, child in enumerate(root.children)
                if game.decode_move(child.encoded_move_to_get_here, root.board).from_square == selected_from_square
            ]
            min_visits = TRAINING_ARGS.self_play.mcts.min_visit_count
            total_vis_over_thresh = 0.0
            # Precompute the denominator for normalized "res:" if needed (only among matching children)
            for k in range(len(root.children)):
                vis_k = root.children_number_of_visits[k]
                if vis_k > min_visits:
                    total_vis_over_thresh += vis_k - min_visits

            for k in matching_indices:
                child = root.children[k]
                mv = game.decode_move(child.encoded_move_to_get_here, root.board)
                dest_sq = mv.to_square
                r_to, c_to = visuals._encode_square(dest_sq)

                # 3a) Color based on child's individual policy
                pol_k = root.children_policies[k]
                vis_k = max(root.children_number_of_visits[k] - min_visits, 0)

                if total_vis_over_thresh > 0:
                    normalized = vis_k / total_vis_over_thresh
                    color_rgb = lerp(np.array([255, 255, 255]), np.array([255, 0, 0]), normalized)
                    color_tuple = tuple(color_rgb.astype(int).tolist())

                    gui.draw_cell(r_to, c_to, color_tuple)

                # 3b) Overlay text in vertical stack
                #    - policy
                gui.draw_text(r_to, c_to, f'pol:{pol_k:.2f}', offset=(0, -30))

                #    - normalized result share (only if visits ≥ 1% of root)
                if total_vis_over_thresh > 0 and vis_k > 0:
                    res_share = vis_k / total_vis_over_thresh
                    gui.draw_text(r_to, c_to, f'res:{res_share:.2f}', offset=(0, -10))

                #    - average value @ raw visits
                avg_val = 0.0
                if vis_k > 0:
                    avg_val = root.children_result_scores[k] / (vis_k + min_visits)
                gui.draw_text(r_to, c_to, f'{avg_val:.2f}@{vis_k}', offset=(0, 10))

                #    - UCB score
                parent_score = root.result_score if not root.parent else 1.0
                ucb_score = ucb(child, 1.7, parent_score)
                gui.draw_text(r_to, c_to, f'ucb:{ucb_score:.2f}', offset=(0, 30))

                #    - Flags: 'F' if fully expanded, 'T' if terminal
                flags = ''
                if child.is_fully_expanded:
                    flags += 'F'
                if child.board and child.is_terminal_node:
                    flags += 'T'
                gui.draw_text(r_to, c_to, flags, offset=(0, 50))

        # 4) Draw all piece symbols on top of whatever highlights we did
        visuals.draw_pieces(root.board, gui)

        # 5) Update window title
        player_name = 'White' if root.board.current_player == 1 else 'Black'
        mode_name = 'Inspect' if inspect_or_search else 'Search'
        gui.update_window_title(f'Current Player: {player_name} @ {root.result_score:.2f}  MCTS Score  –  {mode_name}')

        gui.update_display()

    # Print debug info in console
    print('Sum of all policies:', sum(root.children_policies))
    print('Sum of all visit counts:', sum(root.children_number_of_visits))
    print('Visit counts of root:', root.number_of_visits)
    for k, child in enumerate(root.children):
        mv = game.decode_move(child.encoded_move_to_get_here, root.board)
        print(
            f'Child #{k} ({mv.from_square}→{mv.to_square}): '
            f'visits={root.children_number_of_visits[k]}, '
            f'policy={root.children_policies[k]:.2f}, '
            f'result_score={child.result_score:.2f}, '
            f'fully_expanded={child.is_fully_expanded}'
        )

    # Main event loop
    while True:
        time.sleep(0.05)
        draw()
        events = gui.events_occurred()

        if events.quit:
            exit()
        if events.left:
            # Return up to the parent display
            return
        if events.right:
            # Toggle between Inspect/Search modes
            inspect_or_search = not inspect_or_search

        if events.clicked:
            clicked_cell = gui.get_cell_from_click()  # (row, col) or None
            if clicked_cell is None:
                continue

            clicked_square = visuals._decode_square(clicked_cell)

            if selected_from_square is None:
                # First click: select the "from" square
                selected_from_square = clicked_square
            else:
                # Second click: treat as "to" square
                from_sq = selected_from_square
                to_sq = clicked_square
                selected_from_square = None  # reset selection

                # Find which child (if any) matches from_sq→to_sq
                for k, child in enumerate(root.children):
                    mv = game.decode_move(child.encoded_move_to_get_here, root.board)
                    if mv.from_square == from_sq and mv.to_square == to_sq:
                        print(
                            f'Clicked move: from {from_sq} to {to_sq}  '
                            f'(child #{k}/#{len(root.children)}): visits={child.number_of_visits}, score={child.result_score}'
                        )

                        if inspect_or_search:
                            # In Inspect mode, recuse only if fully expanded & >1 visit
                            if child.is_fully_expanded and child.number_of_visits > 1:
                                display_node(child, inspect_or_search, search_function)
                            else:
                                print('Child is not fully expanded')
                        else:
                            # In Search mode, perform a one‐step search from that child
                            search_function([(child.board, None)])
                        break
                else:
                    # No matching child found
                    print('No child found corresponding to that move')
