import chess
from datasets import load_dataset

from src.mcts.MCTS import MCTS, action_probabilities
from src.mcts.MCTSNode import MCTSNode
from src.settings import TRAINING_ARGS, CurrentBoard, CurrentGame
from src.cluster.InferenceClient import InferenceClient


def get_testing_inference_client() -> InferenceClient:
    """
    Get an inference client for testing purposes.

    Returns:
        InferenceClient: A client with a well-trained model.
    """
    # Load the model from the default path.
    client = InferenceClient(0, TRAINING_ARGS.network, f'reference/{CurrentGame.__class__.__name__}')
    # TODO client.load_model('reference/sft_chess.pt')
    return client


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
    client = get_testing_inference_client()
    mcts = MCTS(client, TRAINING_ARGS.self_play.mcts)

    # Load the chess puzzles dataset.
    dataset = load_dataset('Lichess/chess-puzzles', split='train')
    # Filter for puzzles with mate themes.
    mate_puzzles = []
    for puzzle in dataset:
        if 'mateIn2' in puzzle['Themes']:  # or 'mateIn2' in puzzle['Themes']:
            mate_puzzles.append(puzzle)

            if len(mate_puzzles) == 200:  # Break early, to avoid iterating over 4Mil puzzles.
                break
    print('Found', len(mate_puzzles), 'mate puzzles.')

    inputs: list[tuple[CurrentBoard, MCTSNode | None]] = []
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

        inputs.append((board, None))
        metadata.append((puzzle_id, expected_move, puzzle))

    # Run batch MCTS search.
    print('Running batch MCTS search...')
    results = mcts.search(inputs)

    failures = []
    for (puzzle_id, expected_move, puzzle), result, (board, _) in zip(metadata, results, inputs):
        # Apply a softmax to the visit counts.
        probs = action_probabilities(result.visit_counts)
        moves = [(move, probs[move], visit_count) for move, visit_count in result.visit_counts]

        best_move_encoded, best_visits, _ = max(moves, key=lambda mv: mv[1])
        best_move_uci = CurrentGame.decode_move(best_move_encoded, board).uci()

        expected = next(
            (prob, count) for move, prob, count in moves if CurrentGame.decode_move(move, board).uci() == expected_move
        )

        sorted_moves = list(sorted(moves, key=lambda x: -x[1]))
        moves_str_list = []
        for i, (mv, visits, count) in enumerate(sorted_moves):
            if visits < 0.01:
                number_of_remaining_moves = len(sorted_moves) - i
                moves_str_list.append(f'... {number_of_remaining_moves} more moves with <1% visits ...')
                break
            moves_str_list.append(f'{CurrentGame.decode_move(mv, board).uci()}: {visits:.4f} ({count})')
        moves_str = '\n\t'.join(moves_str_list)
        ratio_str = f'(visit policy: {best_visits:.4f}) with all moves:\n\t{moves_str}\nFor Board with FEN:\n{board.board.fen()}\n{"==="*20}'

        def mates(move: chess.Move):
            b = board.copy()
            b.make_move(move)
            return b.is_game_over()

        if result.result_score < 0.85:
            failures.append(
                f"Puzzle {puzzle_id}: Expected mate move '{expected_move}' {expected} has low score {result.result_score:.2f}."
            )
        elif best_move_uci != expected_move and not mates(chess.Move.from_uci(best_move_uci)):
            failures.append(
                f"Puzzle {puzzle_id}: Expected mate move '{expected_move}' {expected}, but got '{best_move_uci}' {ratio_str}."
            )
        elif (
            'mateIn1' in puzzle['Themes']
            and best_visits < 0.8
            and not all(mates(CurrentGame.decode_move(move, board)) for move, policy, _ in moves if policy > 0.3)
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


if __name__ == '__main__':
    run_mate_puzzle_regression()
