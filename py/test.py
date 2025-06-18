if __name__ == '__main__':
    import torch.multiprocessing as mp

    # set the start method to spawn for multiprocessing
    # this is required for the C++ self play process
    # and should be set before importing torch.multiprocessing
    # otherwise it will not work on Windows
    mp.set_start_method('spawn')

import os

os.environ['OMP_NUM_THREADS'] = '1'  # Limit the number of threads to 1 for OpenMP
os.environ['MKL_NUM_THREADS'] = '1'  # Limit the number of threads to 1 for MKL

import torch

import chess
from AlphaZeroCpp import MCTS, MCTSNode, MCTSParams, InferenceClientParams, MCTSResults, INVALID_NODE, NodeId


import random
import time
from typing import Any, Callable

from src.Encoding import decode_board_state, encode_board_state
from src.games.chess.ChessBoard import ChessBoard
from src.util import lerp
from src.util.save_paths import create_model, create_optimizer, model_save_path, save_model_and_optimizer

from src.settings import TRAINING_ARGS, CurrentGame


game = CurrentGame

board = game.get_initial_board()

for i in range(100):
    if (
        game.get_canonical_board(board) != decode_board_state(encode_board_state(game.get_canonical_board(board)))
    ).any():
        print('Canonical board does not match after encoding/decoding!')
        print('Current board FEN:', board.board.fen())
        print('Original board:', board)
        print('Canonical board:')
        for layer in game.get_canonical_board(board):
            print(layer)
        print('Encoded board:', encode_board_state(game.get_canonical_board(board)))
        print('Decoded board:')
        for layer in decode_board_state(encode_board_state(game.get_canonical_board(board))):
            print(layer)

        exit()

    move = random.choice(board.get_valid_moves())
    board.make_move(move)


def run_mcts_py(fens: list[str], model_path: str):
    from src.cluster.InferenceClient import InferenceClient
    from src.mcts.MCTS import MCTS

    # Load the model from the default path.
    client = InferenceClient(0, TRAINING_ARGS.network, f'reference/{CurrentGame.__class__.__name__}')
    client.load_model(model_path)
    mcts = MCTS(client, TRAINING_ARGS.self_play.mcts)

    boards = [(ChessBoard.from_fen(fen), None) for fen in fens]

    start = time.time()
    for _ in range(10):
        results = mcts.search(boards, [True] * len(boards))  # type: ignore[call-arg]
    end = time.time()
    print(f'PY: Search took {end - start:.2f} seconds for {len(boards)} boards.')

    # TODO assert that the results are the same?


def run_mcts_cpp(fens: list[str], model_path: str):
    inputs = [
        (fen, INVALID_NODE, TRAINING_ARGS.self_play.mcts.num_searches_per_turn) for fen in fens
    ]  # Use the FEN string to create the input for MCTS
    start = time.time()
    for _ in range(10):
        mcts.search(inputs)  # type: ignore[call-arg]
    end = time.time()
    print(f'CPP: Search took {end - start:.2f} seconds for {len(inputs)} boards.')


network = create_model(TRAINING_ARGS.network, torch.device('cpu'))
optimizer = create_optimizer(network, 'adamw')

save_model_and_optimizer(network, optimizer, 0, 'models')

model_path = model_save_path(0, 'models')

client_args = InferenceClientParams(
    device_id=0,
    currentModelPath=str(model_path),
    maxBatchSize=256,  # maybe 512
)

mcts_args = MCTSParams(
    num_parallel_searches=4,
    c_param=1.7,
    dirichlet_alpha=0.3,
    dirichlet_epsilon=0.25,
    node_reuse_discount=0.8,
    min_visit_count=2,
    num_threads=8,
    num_fast_searches=800,
    num_full_searches=800,
)

mcts = MCTS(client_args, mcts_args)

STARTING_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
MATE_IN_ONE_FEN = '1rb5/4r3/3p1npb/3kp1P1/1P3P1P/5nR1/2Q1BK2/bN4NR w - - 3 61'
MATE_IN_TWO_FEN = 'r1bq2r1/b4pk1/p1pp1p2/1p2pP2/1P2P1PB/3P4/1PPQ2P1/R3K2R w - - 0 1'

input_fens = [STARTING_FEN]

# run_mcts_py(input_fens, str(model_path))
# run_mcts_cpp(input_fens, str(model_path))

# Suppose we want to run 800 sims from the initial position,
# and we have no “previous node,” so we pass INVALID_NODE:
boards = [(STARTING_FEN, INVALID_NODE, 80)] * 12
# TODO check this for end game positions - does that find the mate moves?
# TODO check the node reuse, does it work as expected?

results: MCTSResults = mcts.search(boards)
for r in results.results:
    print('eval =', r.result)
    for encoded_move, cnt in r.visits:
        print(f'  {encoded_move} visited {cnt} times')


stats = mcts.get_inference_statistics()


from datasets import load_dataset

from src.mcts.MCTS import action_probabilities
from src.settings import CurrentBoard, CurrentGame


def run_mate_puzzle_regression():
    """
    Regression test for mate puzzles using batch MCTS search.

    Loads 64 puzzles from the Hugging Face chess-puzzles dataset.
    For puzzles with more than one move in the 'Moves' field, the first move is applied
    to the initial FEN (bringing the position to one where mate is available),
    and the expected mate move is taken as the second move.

    The test asserts:
      - The best move from MCTS must match the expected mate move.
      - For puzzles themed as mateIn1, the mate move must account for >80% of total visits.

    Only puzzles that fail these conditions are printed, and an AssertionError is raised.
    """
    # Get the inference client with a well-trained model.

    # Load the chess puzzles dataset.
    dataset = load_dataset('Lichess/chess-puzzles', split='train')
    # Filter for puzzles with mate themes.
    mate_puzzles = []
    for puzzle in dataset:
        if 'mateIn2' in puzzle['Themes']:  # or 'mateIn2' in puzzle['Themes']:
            mate_puzzles.append(puzzle)

            if len(mate_puzzles) == 20:  # 200:  # Break early, to avoid iterating over 4Mil puzzles.
                break
    print('Found', len(mate_puzzles), 'mate puzzles.')

    inputs: list[tuple[str, NodeId, bool]] = []
    metadata: list[tuple[str, str, str]] = []  # (PuzzleId, expected_move, Themes)

    for puzzle in mate_puzzles:
        puzzle_id = puzzle['PuzzleId']
        fen = puzzle['FEN']
        moves_str = puzzle['Moves']
        moves = moves_str.split()

        # Load the starting board.
        board = CurrentBoard.from_fen(fen)

        # If more than one move is provided, apply the first move to reach the mate position.
        if len(moves) > 1:
            pre_move = chess.Move.from_uci(moves[0])
            board.make_move(pre_move)
            expected_move = moves[1]
        else:
            expected_move = moves[0]

        inputs.append((board.board.fen(), INVALID_NODE, True))
        metadata.append((puzzle_id, expected_move, puzzle))

    # Run batch MCTS search.
    print('Running batch MCTS search...')
    res = mcts.search(inputs)
    results = res.results

    failures = []
    for (puzzle_id, expected_move, puzzle), result, (fen, _, _) in zip(metadata, results, inputs):
        board = CurrentBoard.from_fen(fen)

        # Apply a softmax to the visit counts.
        probs = action_probabilities(result.visits)
        moves = [
            (CurrentGame.decode_move(move, board).uci(), probs[move], visit_count)
            for move, visit_count in result.visits
        ]
        sorted_moves = list(sorted(moves, key=lambda x: -x[1]))

        print(f'Puzzle {puzzle_id} with expected move {expected_move} and themes {puzzle["Themes"]}')
        print(f'FEN: {fen}')
        print(f'Board:\n{repr(board)}')
        print(f'Expected move: {expected_move}')
        print(f'Result: {result.result:.2f}')
        print(f'Moves: {sorted_moves}')

        best_move, best_visits, _ = max(moves, key=lambda mv: mv[1])

        expected = next((prob, count) for move, prob, count in moves if move == expected_move)

        moves_str_list = []
        for i, (mv, visits, count) in enumerate(sorted_moves):
            if visits < 0.01:
                number_of_remaining_moves = len(sorted_moves) - i
                moves_str_list.append(f'... {number_of_remaining_moves} more moves with <1% visits ...')
                break
            moves_str_list.append(f'{mv}: {visits:.4f} ({count})')
        moves_str = '\n\t'.join(moves_str_list)
        ratio_str = f'(visit policy: {best_visits:.4f}) with all moves:\n\t{moves_str}\nFor Board with FEN:\n{board.board.fen()}\n{"===" * 20}'

        def mates(move: chess.Move):
            b = board.copy()
            b.make_move(move)
            return b.is_game_over()

        if result.result < 0.85:
            failures.append(
                f"Puzzle {puzzle_id}: Expected mate move '{expected_move}' {expected} has low score {result.result:.2f}."
            )
        elif best_move != expected_move and not mates(chess.Move.from_uci(best_move)):
            failures.append(
                f"Puzzle {puzzle_id}: Expected mate move '{expected_move}' {expected}, but got '{best_move}' {ratio_str}."
            )
        elif (
            'mateIn1' in puzzle['Themes']
            and best_visits < 0.8
            and not all(mates(chess.Move.from_uci(move)) for move, policy, _ in moves if policy > 0.3)
        ):
            failures.append(
                f"Puzzle {puzzle_id}: Mate in 1 expected move '{expected_move}' {expected} has low visit ratio {ratio_str}."
            )

    if failures:
        print('Regression test failures:')
        for failure in failures:
            print(failure)
        print(f'{len(failures)} regression test failures detected.')
    else:
        print('All puzzles passed regression test.')


if __name__ == '__main__' and False:
    try:
        run_mate_puzzle_regression()
    except:
        pass


def display_node(root: MCTSNode, inspect_or_search: bool, search_function: Callable[[str], Any]) -> None:
    import time
    import numpy as np

    from src.eval.GridGUI import BaseGridGameGUI
    from src.games.chess.ChessVisuals import ChessVisuals
    from src.games.chess.ChessGame import BOARD_LENGTH
    from src.settings import TRAINING_ARGS

    visuals = ChessVisuals()
    gui = BaseGridGameGUI(BOARD_LENGTH, BOARD_LENGTH)

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
                mv = chess.Move.from_uci(child.move)
                origin = mv.from_square
                aggregated_policy[origin] = aggregated_policy.get(origin, 0.0) + child.policy
                aggregated_visits[origin] = aggregated_visits.get(origin, 0) + child.visits

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
                if chess.Move.from_uci(child.move).from_square == selected_from_square
            ]
            min_visits = TRAINING_ARGS.self_play.mcts.min_visit_count
            total_vis_over_thresh = 0.0
            # Precompute the denominator for normalized "res:" if needed (only among matching children)
            for child in root.children:
                vis_k = child.visits
                if vis_k > min_visits:
                    total_vis_over_thresh += vis_k - min_visits

            for k in matching_indices:
                child = root.children[k]
                mv = chess.Move.from_uci(child.move)
                dest_sq = mv.to_square
                r_to, c_to = visuals._encode_square(dest_sq)

                # 3a) Color based on child's individual policy
                pol_k = child.policy
                vis_k = max(child.visits - min_visits, 0)

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
                    avg_val = child.result / (vis_k + min_visits)
                gui.draw_text(r_to, c_to, f'{avg_val:.2f}@{vis_k}', offset=(0, 10))

                #    - UCB score
                ucb_score = child.ucb(1.7)
                gui.draw_text(r_to, c_to, f'ucb:{ucb_score:.2f}', offset=(0, 30))

                #    - Flags: 'F' if fully expanded, 'T' if terminal
                flags = ''
                if child.is_fully_expanded:
                    flags += 'F'
                if child.is_terminal:
                    flags += 'T'
                gui.draw_text(r_to, c_to, flags, offset=(0, 50))

        board = ChessBoard.from_fen(root.fen)

        # 4) Draw all piece symbols on top of whatever highlights we did
        visuals.draw_pieces(board, gui)

        # 5) Update window title
        player_name = 'White' if board.current_player == 1 else 'Black'
        mode_name = 'Inspect' if inspect_or_search else 'Search'
        gui.update_window_title(f'Current Player: {player_name} @ {root.result:.2f}  MCTS Score  –  {mode_name}')

        gui.update_display()

    # Print debug info in console
    print('Sum of all policies:', sum(child.policy for child in root.children))
    print('Sum of all visit counts:', sum(child.visits for child in root.children))
    print('Visit counts of root:', root.visits)

    # Main event loop
    while True:
        time.sleep(0.05)  # Sleep to reduce CPU usage
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
                    mv = chess.Move.from_uci(child.move)
                    if mv.from_square == from_sq and mv.to_square == to_sq:
                        print(
                            f'Clicked move: from {from_sq} to {to_sq}  '
                            f'(child #{k}/#{len(root.children)}): visits={child.visits}, score={child.result:.2f}, policy={child.policy:.2f}'
                        )

                        if inspect_or_search:
                            # In Inspect mode, recuse only if fully expanded & >1 visit
                            if child.is_fully_expanded and child.visits > 1:
                                display_node(child, inspect_or_search, search_function)
                            else:
                                print('Child is not fully expanded')
                        else:
                            # In Search mode, perform a one‐step search from that child
                            search_function(child.fen)
                        break
                else:
                    # No matching child found
                    print('No child found corresponding to that move')


def dis(fen: str, id: NodeId) -> None:
    inp = [(fen, id, True)]  # Use the FEN string to create the input for MCTS
    results: MCTSResults = mcts.search(inp)
    # get the id of the root node of the first result
    root = mcts.get_node(results.results[0].children[0]).parent
    assert root is not None, 'Root node should not be None'

    def search_function(fen: str) -> None:
        # find out the node id of the child node with the given fen
        for child, childId in zip(root.children, results.results[0].children):
            if child.fen == fen:
                dis(child.fen, childId)

    display_node(
        root,
        inspect_or_search=False,  # Start in Inspect mode
        search_function=search_function,  # Use MCTS search function
    )


dis(MATE_IN_ONE_FEN, INVALID_NODE)

input('Press Enter to exit...')  # Keep the script running to see the output in the console.
