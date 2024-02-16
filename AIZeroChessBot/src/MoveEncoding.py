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
    :param current_player: The current player to encode the move for.
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


def decode_moves(move_indices: NDArray[np.int32]) -> list[Move]:
    """
    Decodes an array of move indices into a list of chess moves.

    :param move_indices: The array of move indices to decode.
    :return: The list of decoded chess moves.
    """
    moves = [__REVERSE_MOVE_MAPPINGS[index] for index in move_indices]
    return [Move(from_square, to_square, promotion=promotion_type) for from_square, to_square, promotion_type in moves]


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


def flip_action_probabilities(
    action_probabilities: NDArray[np.float32], flip_move_index: Callable[[int | None], int | None]
) -> NDArray[np.float32]:
    flipped_probabilities = np.zeros_like(action_probabilities)

    for idx, prob in enumerate(action_probabilities):
        flipped_idx = flip_move_index(idx)
        if flipped_idx is not None:  # Ensure the move is valid after flipping
            flipped_probabilities[flipped_idx] = prob

    return flipped_probabilities


def flip_move_index_horizontal(move_index: int | None) -> int | None:
    return __FLIPPED_INDICES_HORIZONTAL[move_index] if move_index is not None else None


def flip_move_index_vertical(move_index: int | None) -> int | None:
    return __FLIPPED_INDICES_VERTICAL[move_index] if move_index is not None else None


def flip_square_horizontal(square: Square) -> Square:
    # Flip the file of the square, keeping the rank constant
    rank, file = divmod(square, 8)
    flipped_file = 7 - file  # 0 becomes 7, 1 becomes 6, ..., 7 becomes 0
    return rank * 8 + flipped_file


def flip_square_vertical(square: Square) -> Square:
    # Flip the rank of the square, keeping the file constant
    rank, file = divmod(square, 8)
    flipped_rank = 7 - rank  # 0 becomes 7, 1 becomes 6, ..., 7 becomes 0
    return flipped_rank * 8 + file


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


def __precalculate_flipped_indices(
    move_mappings: list[list[dict[PieceType | None, int]]],
) -> tuple[list[int | None], list[int | None]]:
    flipped_indices_horizontal: list[int | None] = [None] * ACTION_SIZE
    flipped_indices_vertical: list[int | None] = [None] * ACTION_SIZE

    for from_square in range(64):
        for to_square in range(64):
            for promotion, index in move_mappings[from_square][to_square].items():
                # Calculate the flipped squares
                flipped_from_horizontal = flip_square_horizontal(from_square)
                flipped_to_horizontal = flip_square_horizontal(to_square)
                flipped_from_vertical = flip_square_vertical(from_square)
                flipped_to_vertical = flip_square_vertical(to_square)

                # Get the corresponding flipped indices
                flipped_index_horizontal = move_mappings[flipped_from_horizontal][flipped_to_horizontal].get(promotion)
                flipped_index_vertical = move_mappings[flipped_from_vertical][flipped_to_vertical].get(promotion)

                # Store in the precalculated lists
                flipped_indices_horizontal[index] = flipped_index_horizontal
                flipped_indices_vertical[index] = flipped_index_vertical

    return flipped_indices_horizontal, flipped_indices_vertical


__MOVE_MAPPINGS, ACTION_SIZE = __precalculate_move_mappings()
__REVERSE_MOVE_MAPPINGS = __precalculate_reverse_move_mappings(__MOVE_MAPPINGS)
__FLIPPED_INDICES_HORIZONTAL, __FLIPPED_INDICES_VERTICAL = __precalculate_flipped_indices(__MOVE_MAPPINGS)


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

    # Decode all moves at once
    moves = decode_moves(nonzero_indices)

    # Pair up moves with their probabilities
    moves_with_probabilities = list(zip(moves, policy[nonzero_indices]))

    return moves_with_probabilities
