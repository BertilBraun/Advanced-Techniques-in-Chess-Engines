import numpy as np
from numpy.typing import NDArray

from Framework import *


def filter_policy_then_get_moves_and_probabilities(
    policy: NDArray[np.float32], board: Board
) -> list[tuple[Move, float]]:
    """
    Gets a list of moves with their corresponding probabilities from a policy.

    The policy is a 1D numpy array representing the probabilities of each move
    in the board. The list of moves is a list of tuples, where each tuple contains
    a move and its corresponding probability.

    :param policy: The policy to get the moves and probabilities from.
    :param board: The chess board to filter the policy with.
    :return: The list of moves with their corresponding probabilities.
    """
    filtered_policy = __filter_policy_with_legal_moves(policy, board)
    moves_with_probabilities = __map_policy_to_moves(filtered_policy)
    return moves_with_probabilities


def encode_move(move: Move) -> int:
    """
    Encodes a chess move into a move index.

    :param move: The move to encode.
    :return: The encoded move index.
    """
    if move.promotion not in __MOVE_MAPPINGS[move.from_square][move.to_square]:
        raise ValueError(f'Error: move.promotion not in MOVE_MAPPINGS[move.from_square][move.to_square]: {move}')

    return __MOVE_MAPPINGS[move.from_square][move.to_square][move.promotion]


def decode_move(move_index: int) -> Move:
    """
    Decodes a move index into a chess move.

    :param move_index: The index of the move to decode.
    :return: The decoded chess move.
    """
    from_square, to_square, promotion_type = __REVERSE_MOVE_MAPPINGS[move_index]
    return Move(from_square, to_square, promotion=promotion_type)


def print_move_mappings(move_mappings: list[list[dict[PieceType | None, int]]]) -> None:
    """
    Prints the move mappings to the console.

    :param move_mappings: The move mappings to print.
    """
    total_moves = 0

    for from_square, moves in enumerate(move_mappings):
        for to_square, promotional_mapping in enumerate(moves):
            for promotion_type, index in promotional_mapping.items():
                print(f'{square_name(from_square)} -> {square_name(to_square)}: {promotion_type} -> {index}')
                total_moves += 1

    print(f'Total moves: {total_moves}')


def __precalculate_move_mappings() -> tuple[list[list[dict[PieceType | None, int]]], int]:
    KNIGHT_MOVES = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
    ROOK_MOVES = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    BISHOP_MOVES = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

    move_mappings: list[list[dict[PieceType | None, int]]] = [[{} for _ in range(64)] for _ in range(64)]
    index = 0

    def add_move(from_square: Square, to_square: Square, promotion_type: PieceType | None) -> None:
        nonlocal index
        move_mappings[from_square][to_square][promotion_type] = index
        index += 1

    def add_promotion_moves(from_square: Square, col: int, to_row: int) -> None:
        for offset in (-1, 0, 1):
            if 0 <= col + offset < 8:
                to_square = square(col + offset, to_row)
                add_move(from_square, to_square, QUEEN)
                add_move(from_square, to_square, ROOK)
                add_move(from_square, to_square, BISHOP)
                add_move(from_square, to_square, KNIGHT)

    for row in range(8):
        for col in range(8):
            from_square = square(col, row)

            # Calculate knight moves from this square
            for dx, dy in KNIGHT_MOVES:
                if 0 <= row + dx < 8 and 0 <= col + dy < 8:  # Check if move is within bounds
                    to_square = square(col + dy, row + dx)
                    add_move(from_square, to_square, None)

            # Calculate rook moves from this square
            for dx, dy in ROOK_MOVES:
                for i in range(1, 8):
                    if 0 <= row + i * dx < 8 and 0 <= col + i * dy < 8:
                        to_square = square(col + i * dy, row + i * dx)
                        add_move(from_square, to_square, None)

            # Calculate bishop moves from this square
            for dx, dy in BISHOP_MOVES:
                for i in range(1, 8):
                    if 0 <= row + i * dx < 8 and 0 <= col + i * dy < 8:
                        to_square = square(col + i * dy, row + i * dx)
                        add_move(from_square, to_square, None)

            # Calculate pawn promotion moves from this square
            if row == 1:
                add_promotion_moves(from_square, col, row - 1)
            elif row == 6:
                add_promotion_moves(from_square, col, row + 1)

    return move_mappings, index


def __precalculate_reverse_move_mappings(
    move_mappings: list[list[dict[PieceType | None, int]]],
) -> list[tuple[Square, Square, PieceType | None]]:
    reverse_move_mappings: list[tuple[Square, Square, PieceType | None]] = [None] * ACTION_SIZE  # type: ignore

    for from_square, moves in enumerate(move_mappings):
        for to_square, promotional_mapping in enumerate(moves):
            for promotion_type, index in promotional_mapping.items():
                reverse_move_mappings[index] = (from_square, to_square, promotion_type)

    return reverse_move_mappings


__MOVE_MAPPINGS, ACTION_SIZE = __precalculate_move_mappings()
__REVERSE_MOVE_MAPPINGS = __precalculate_reverse_move_mappings(__MOVE_MAPPINGS)


def __encode_legal_moves(board: Board) -> NDArray[np.int8]:
    """
    Encodes the legal moves of a chess board into a 1D numpy array.

    Each entry in the array represents a possible move on the board. If the
    corresponding move is legal, the entry is 1, and 0 otherwise. The array
    has a length of TOTAL_MOVES, representing all possible moves from all squares
    to all reachable squares.

    :param board: The chess board to encode.
    :return: A 1D numpy array representing the encoded legal moves.
    """
    # Initialize a 1D array filled with zeros
    # There are TOTAL_MOVES possible moves
    legal_moves_encoded = np.zeros(ACTION_SIZE, dtype=np.int8)

    # Iterate over all legal moves available in the position
    for move in board.legal_moves:
        legal_moves_encoded[encode_move(move)] = 1

    return legal_moves_encoded


def __filter_policy_with_legal_moves(policy: NDArray[np.float32], board: Board) -> NDArray[np.float32]:
    """
    Filters a policy with the legal moves of a chess board.

    The policy is a 1D numpy array representing the probabilities of each move
    in the board. The legal moves are encoded in a 1D numpy array, where each
    entry is 1 if the corresponding move is legal, and 0 otherwise. The policy
    is then filtered to only include the probabilities of the legal moves.

    :param policy: The policy to filter.
    :param board: The chess board to filter the policy with.
    :return: The filtered policy.
    """
    legal_moves_encoded = __encode_legal_moves(board)
    policy *= legal_moves_encoded
    policy /= np.sum(policy)
    return policy


def __map_policy_to_moves(policy: NDArray[np.float32]) -> list[tuple[Move, float]]:
    """
    Maps a filtered policy to a list of moves with their corresponding probabilities.

    The policy is a 1D numpy array representing the probabilities of each move
    in the board. The list of moves is a list of tuples, where each tuple contains
    a move and its corresponding probability.

    :param policy: The policy to map.
    :return: The list of moves with their corresponding probabilities.
    """
    # Find indices where probability > 0
    nonzero_indices = np.nonzero(policy > 0)[0]
    moves_with_probabilities = []

    # Assuming decode_move can only process one index at a time
    for move_index in nonzero_indices:
        move = decode_move(move_index)
        probability = policy[move_index]
        moves_with_probabilities.append((move, probability))

    return moves_with_probabilities
