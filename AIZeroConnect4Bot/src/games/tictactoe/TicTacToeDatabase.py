from __future__ import annotations

from collections import defaultdict
from typing import Dict, Tuple, List

import numpy as np

from AIZeroConnect4Bot.src.games.tictactoe.TicTacToeBoard import TicTacToeBoard
from AIZeroConnect4Bot.src.games.tictactoe.TicTacToeGame import TicTacToeGame

# Define the outcome constants
OUTCOME_WIN = 1
OUTCOME_DRAW = 0
OUTCOME_LOSS = -1

# Type aliases for memoization
MemoKey = Tuple[int, ...]  # board_state
MemoValue = int  # outcome

# Memoization dictionary
memo: Dict[MemoKey, MemoValue] = {}


def board_to_key(board: TicTacToeBoard) -> Tuple[int, ...]:
    return tuple(board.board.tolist())


def board_from_str(s: str) -> TicTacToeBoard:
    """Example:
     OX
     OX
    OXO"""
    board = TicTacToeBoard()
    for j, line in enumerate(s.split('\n')):
        for i in range(3):
            if line[i] == 'X':
                board.board[j * 3 + i] = 1
            elif line[i] == 'O':
                board.board[j * 3 + i] = -1
    if s.count('O') > s.count('X'):
        board.board *= -1
        board.current_player = 1
    else:
        board.current_player = 1 if s.count('X') == s.count('O') else -1
    return board


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


def generate_memory_line(board: TicTacToeBoard, moves: List[int], outcome: int) -> str:
    """
    Generates a line for the database file in the format:
    (board_state);(move1, move2, ...);outcome
    """

    board_state = tuple(board.board.tolist())
    moves_tuple = tuple(moves)
    return f'{board_state};{moves_tuple};{outcome}\n'


def generate_database(output_file: str):
    """
    Generate the TicTacToe database and save it to the specified output file.
    Each line in the file has the format:
    (board_state);(move1, move2, ...);outcome
    """
    initial_board = TicTacToeBoard()

    # To keep track of all states to process
    states_to_process: List[TicTacToeBoard] = [initial_board]
    generated_states: set[MemoKey] = set()  # Initialize as empty

    with open(output_file, 'w') as f:
        while states_to_process:
            current_board = states_to_process.pop()
            key = board_to_key(current_board)

            if key in generated_states or current_board.is_game_over():
                # Already processed and written to file
                continue

            move_list, outcome = get_best_moves(current_board)

            if move_list:
                # Store the memory
                if current_board.current_player == 1:
                    f.write(generate_memory_line(current_board, move_list, outcome))
                else:
                    current_board.board *= -1  # Switch perspective
                    f.write(generate_memory_line(current_board, move_list, -outcome))
                    current_board.board *= -1  # Switch back
                generated_states.add(key)

                # Recurse on all valid moves to ensure all states are processed
                for move in current_board.get_valid_moves():
                    new_board = current_board.copy()
                    new_board.make_move(move)

                    if board_to_key(new_board) not in generated_states:
                        states_to_process.append(new_board)
            else:
                assert False


if __name__ == '__main__':
    output_file = 'tictactoe_database.txt'
    board = TicTacToeBoard()
    board.board = np.array([-1, 0, 0, 0, 0, 0, 1, -1, 1])
    board.current_player = 1

    print(board.board.reshape(3, 3))
    print(TicTacToeGame().get_canonical_board(board).flatten().tolist())
    print(minimax(board))
    for move in board.get_valid_moves():
        new_board = board.copy()
        new_board.make_move(move)
        print(new_board.board.reshape(3, 3))
        print(minimax(new_board))
    print(get_best_moves(board))
    # exit()

    print('Generating TicTacToe database...')
    generate_database(output_file)
    print(f'Database saved to {output_file}')
