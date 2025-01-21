from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np


from src.games.tictactoe.TicTacToeBoard import TicTacToeBoard
from src.alpha_zero.SelfPlayDataset import SelfPlayDataset
from src.games.tictactoe.TicTacToeGame import TicTacToeGame


# Type aliases for memoization
MemoKey = Tuple[int, ...]  # board_state
MemoValue = int  # outcome

# Memoization dictionary
memo: Dict[MemoKey, MemoValue] = {}


def board_to_key(board: TicTacToeBoard) -> Tuple[int, ...]:
    return tuple(board.board.tolist())


def minimax(board: TicTacToeBoard) -> int:
    """
    Evaluate the board and return the list of optimal moves and outcome for the current player.
    Returns a optimal outcome
    """

    key = board_to_key(board)
    if key in memo:
        return memo[key]

    winner = board.check_winner()

    if winner == 1:
        # Current player has already won
        memo[key] = 1
        return memo[key]
    elif winner == -1:
        # Opponent has won
        memo[key] = -1
        return memo[key]
    elif board.is_full():
        # Draw
        memo[key] = 0
        return memo[key]

    if board.current_player == 1:
        # Maximizing player
        best_outcome = -1
        for move in board.get_valid_moves():
            new_board = board.copy()
            new_board.make_move(move)
            best_outcome = max(best_outcome, minimax(new_board))
    else:
        # Minimizing player
        best_outcome = 1
        for move in board.get_valid_moves():
            new_board = board.copy()
            new_board.make_move(move)
            best_outcome = min(best_outcome, minimax(new_board))

    memo[key] = best_outcome
    return memo[key]


def get_best_moves(board: TicTacToeBoard) -> Tuple[List[int], int]:
    """
    Get the best moves for the current player on the board.
    """
    counter = defaultdict(list)

    for move in board.get_valid_moves():
        new_board = board.copy()
        new_board.make_move(move)
        outcome = minimax(new_board)
        counter[outcome].append(move)

    if board.current_player == 1:
        outcome = max(counter.keys())
    else:
        outcome = min(counter.keys())

    return counter[outcome], outcome


def generate_database() -> Path:
    game = TicTacToeGame()

    # To keep track of all states to process
    states_to_process: List[TicTacToeBoard] = [TicTacToeBoard()]
    generated_states: set[MemoKey] = set()  # Initialize as empty

    dataset = SelfPlayDataset()

    while states_to_process:
        current_board = states_to_process.pop()
        key = board_to_key(current_board)

        if key in generated_states or current_board.is_game_over():
            # Already processed and written to file
            continue

        move_list, outcome = get_best_moves(current_board)

        if move_list:
            # Store the memory
            policy = np.array([1 if move in move_list else 0 for move in range(9)], dtype=np.float32)
            policy /= np.sum(policy)

            dataset.add_sample(
                game.get_canonical_board(current_board),
                policy,
                outcome if current_board.current_player == 1 else -outcome,
            )
            generated_states.add(key)

            # Recurse on all valid moves to ensure all states are processed
            for move in current_board.get_valid_moves():
                new_board = current_board.copy()
                new_board.make_move(move)

                if board_to_key(new_board) not in generated_states:
                    states_to_process.append(new_board)
        else:
            assert False

    return dataset.save('reference', 0, 'tictactoe_database')


if __name__ == '__main__':
    print('Generating TicTacToe database...')
    output_file = generate_database()
    print(f'Database saved to {output_file}')
