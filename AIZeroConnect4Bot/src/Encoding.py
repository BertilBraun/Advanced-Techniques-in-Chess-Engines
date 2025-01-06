import numpy as np

from src.games.Game import Board
from src.settings import CurrentGame, CurrentGameMove, CurrentBoard


_N_BITS = CurrentGame.representation_shape[1] * CurrentGame.representation_shape[2]
assert _N_BITS <= 64, 'The state is too large to encode'
# Prepare the bit masks: 1, 2, 4, ..., 2^(n_bits-1)
_BIT_MASK = 1 << np.arange(_N_BITS, dtype=np.uint64)  # use uint64 to prevent overflow


def encode_board_state(state: np.ndarray) -> np.ndarray:
    """Encode the state into a tuple of integers. Each integer represents a channel of the state. This assumes that the state is a binary state.

    The encoding is done by setting the i-th bit of the integer to the i-th bit of the flattened state.
    For example, if the state is:
    [[1, 0],
     [0, 1]]
    The encoding would be:
    >>> 1001 = 9

    For a state with multiple channels, the encoding is done for each channel separately.
    [[[1, 0]],
     [[0, 1]]]
    The encoding would be:
    >>> [10, 01] = [2, 1]
    """
    # Shape: (channels, height * width)
    flattened = state.reshape(state.shape[0], -1).astype(np.uint64)

    # Perform vectorized dot product to encode each channel
    encoded = (flattened * _BIT_MASK).sum(axis=1)

    return encoded


def decode_board_state(state: np.ndarray) -> np.ndarray:
    # Convert to uint64 to prevent overflow
    state = state.astype(np.uint64)

    encoded_array = state.reshape(-1, 1)  # shape: (channels, 1)

    # Extract bits for each channel
    bits = ((encoded_array & _BIT_MASK) > 0).astype(np.int8)

    return bits.reshape(CurrentGame.representation_shape)


def get_board_result_score(board: Board) -> float | None:
    """
    Returns the result score for the given board.

    :param board: The board to get the result score for.
    :return: The result score for the given board. 1 if the current player won, -1 if the current player lost, and 0 if the game is a draw.
    """
    if winner := board.check_winner():
        return winner * -board.current_player  # 1 if current player won, -1 if current player lost

    if board.is_game_over():
        return 0.0

    return None


def filter_policy_then_get_moves_and_probabilities(
    policy: np.ndarray, board: CurrentBoard
) -> list[tuple[CurrentGameMove, float]]:
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


def __filter_policy_with_legal_moves(policy: np.ndarray, board: CurrentBoard) -> np.ndarray:
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
    legal_moves_encoded = CurrentGame.encode_moves(board.get_valid_moves())
    filtered_policy = policy * legal_moves_encoded
    policy_sum = np.sum(filtered_policy)
    if policy_sum == 0:
        filtered_policy = legal_moves_encoded / np.sum(legal_moves_encoded)
    else:
        filtered_policy /= policy_sum
    return filtered_policy


def __map_policy_to_moves(policy: np.ndarray) -> list[tuple[CurrentGameMove, float]]:
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

    # Decode the indices to moves
    moves = CurrentGame.decode_moves(nonzero_indices)

    # Pair up moves with their probabilities
    moves_with_probabilities = list(zip(moves, policy[nonzero_indices]))

    return moves_with_probabilities
