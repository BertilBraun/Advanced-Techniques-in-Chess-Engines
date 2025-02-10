from chess import (
    Move,
    A5,
    B6,
    B5,
    A6,
    C6,
    C5,
    D6,
    D5,
    E6,
    E5,
    F6,
    F5,
    G6,
    G5,
    H6,
    H5,
    A4,
    B3,
    B4,
    A3,
    C3,
    C4,
    D3,
    D4,
    E3,
    E4,
    F3,
    F4,
    G3,
    G4,
    H3,
    H4,
)
import numpy as np
import numpy.typing as npt

from numba import njit

from src.games.Game import Board
from src.settings import CurrentGame, CurrentBoard
from src.util.timing import timeit


_BOARD_SHAPE = CurrentGame.representation_shape
_N_BITS = _BOARD_SHAPE[1] * _BOARD_SHAPE[2]
assert _N_BITS <= 64, 'The state is too large to encode'
# Prepare the bit masks: 1, 2, 4, ..., 2^(n_bits-1)
_BIT_MASK = 1 << np.arange(_N_BITS, dtype=np.uint64)  # use uint64 to prevent overflow


def encode_board_state(state: npt.NDArray[np.int8]) -> npt.NDArray[np.uint64]:
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
    # check if the state is not continuous, and if so, make it continuous
    if not state.flags['C_CONTIGUOUS']:
        state = np.ascontiguousarray(state)

    return _encode_board_state(state)


@njit
def _encode_board_state(state: npt.NDArray[np.int8]) -> npt.NDArray[np.uint64]:
    # Shape: (channels, height * width)
    flattened = state.reshape(state.shape[0], -1).astype(np.uint64)

    # Perform vectorized dot product to encode each channel
    encoded = (flattened * _BIT_MASK).sum(axis=1)

    return encoded


def encode_board_states(states: npt.NDArray[np.int8]) -> npt.NDArray[np.uint64]:
    """Encode multiple states into a tuple of integers. Each integer represents a channel of the state. This assumes that the state is a binary state."""
    # check if the state is not continuous, and if so, make it continuous
    if not states.flags['C_CONTIGUOUS']:
        states = np.ascontiguousarray(states)

    return _encode_board_states(states)


@njit
def _encode_board_states(states: npt.NDArray[np.int8]) -> npt.NDArray[np.uint64]:
    # Shape: (batch, channels, height * width)
    flattened = states.reshape(states.shape[0], states.shape[1], -1).astype(np.uint64)

    # Perform vectorized dot product to encode each channel
    encoded = (flattened * _BIT_MASK).sum(axis=2)

    return encoded


def decode_board_state(state: npt.NDArray[np.uint64]) -> npt.NDArray[np.int8]:
    """Convert a tuple of integers into a binary state. Each integer represents a channel of the state. This assumes that the state is a binary state."""
    assert state.dtype == np.uint64, 'The state must be encoded as uint64 to prevent overflow'

    return _decode_board_state(state)


@njit
def _decode_board_state(state: npt.NDArray[np.uint64]) -> npt.NDArray[np.int8]:
    encoded_array = state.reshape(-1, 1)  # shape: (channels, 1)

    # Extract bits for each channel
    bits = ((encoded_array & _BIT_MASK) > 0).astype(np.int8)

    return bits.reshape(_BOARD_SHAPE)


def get_board_result_score(board: Board) -> float | None:
    """
    Returns the result score for the given board.

    :param board: The board to get the result score for.
    :return: The result score for the given board. -1 if the current player has lost, -0.5 to 0.5 for a draw. None if the game is not over.
    """
    if not board.is_game_over():
        return None

    if winner := board.check_winner():
        assert winner != board.current_player, 'The winner must be the opponent, sine he just played a checkmate move'
        return -1.0

    return 0.5 * board.current_player * board.get_approximate_result_score()


MoveScore = tuple[int, float]


def filter_policy_with_en_passant_moves_then_get_moves_and_probabilities(
    policy: np.ndarray, board: CurrentBoard
) -> list[MoveScore]:
    """
    Gets a list of moves with their corresponding probabilities from a policy.

    The policy is a 1D numpy array representing the probabilities of each move
    in the board. The list of moves is a list of tuples, where each tuple contains
    a move and its corresponding probability.

    :param policy: The policy to get the moves and probabilities from.
    :param board: The chess board to filter the policy with.
    :return: The list of moves with their corresponding probabilities.
    """
    filtered_policy = __filter_policy_with_legal_moves_and_en_passant_moves(policy, board)
    moves_with_probabilities = __map_policy_to_moves(filtered_policy)
    return moves_with_probabilities


def __filter_policy_with_legal_moves_and_en_passant_moves(policy: np.ndarray, board: CurrentBoard) -> np.ndarray:
    """
    Filters a policy with the legal moves of a chess board but also allows all en passant moves.

    The policy is a 1D numpy array representing the probabilities of each move
    in the board. The legal moves are encoded in a 1D numpy array, where each
    entry is 1 if the corresponding move is legal, and 0 otherwise. The policy
    is then filtered to only include the probabilities of the legal moves.

    :param policy: The policy to filter.
    :param board: The chess board to filter the policy with.
    :return: The filtered policy.
    """
    en_passant_moves = [
        # White en passant moves
        Move(A5, B6),
        Move(B5, A6),
        Move(B5, C6),
        Move(C5, B6),
        Move(C5, D6),
        Move(D5, C6),
        Move(D5, E6),
        Move(E5, D6),
        Move(E5, F6),
        Move(F5, E6),
        Move(F5, G6),
        Move(G5, F6),
        Move(G5, H6),
        Move(H5, G6),
        # Black en passant moves
        Move(A4, B3),
        Move(B4, A3),
        Move(B4, C3),
        Move(C4, B3),
        Move(C4, D3),
        Move(D4, C3),
        Move(D4, E3),
        Move(E4, D3),
        Move(E4, F3),
        Move(F4, E3),
        Move(F4, G3),
        Move(G4, F3),
        Move(G4, H3),
        Move(H4, G3),
    ]
    legal_moves_encoded = CurrentGame.encode_moves(board.get_valid_moves() + en_passant_moves)
    filtered_policy = policy * legal_moves_encoded
    policy_sum = np.sum(filtered_policy)
    if policy_sum == 0:
        filtered_policy = legal_moves_encoded / np.sum(legal_moves_encoded)
    else:
        filtered_policy /= policy_sum
    return filtered_policy


@timeit
def filter_policy_then_get_moves_and_probabilities(policy: np.ndarray, board: CurrentBoard) -> list[MoveScore]:
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


def filter_moves(moves: list[MoveScore], board: CurrentBoard) -> list[MoveScore]:
    """
    Filters a list of moves with the legal moves of a chess board.

    The list of moves is a list of tuples, where each tuple contains
    a move and its corresponding probability. The legal moves are encoded in a 1D numpy array, where each
    entry is 1 if the corresponding move is legal, and 0 otherwise. The policy
    is then filtered to only include the probabilities of the legal moves.

    :param moves: The moves to filter.
    :param board: The chess board to filter the moves with.
    :return: The filtered moves.
    """
    legal_moves_encoded = CurrentGame.encode_moves(board.get_valid_moves())
    filtered_moves = [(move, prob) for move, prob in moves if legal_moves_encoded[move] > 0 and prob > 0]

    prob_sum = sum(prob for _, prob in filtered_moves)

    if prob_sum == 0:
        return __map_policy_to_moves(legal_moves_encoded / np.sum(legal_moves_encoded))

    for i, (move, prob) in enumerate(filtered_moves):
        filtered_moves[i] = (move, prob / prob_sum)

    return filtered_moves


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


def __map_policy_to_moves(policy: np.ndarray) -> list[MoveScore]:
    """
    Maps a filtered policy to a list of moves with their corresponding probabilities.

    The policy is a 1D numpy array representing the probabilities of each move
    in the board. The list of moves is a list of tuples, where each tuple contains
    a move and its corresponding probability.

    :param policy: The policy to map.
    :return: The list of encoded moves with their corresponding probabilities.
    """
    # Find indices where probability > 0
    nonzero_indices = np.nonzero(policy > 0)[0]

    # Pair up moves with their probabilities
    moves_with_probabilities = list(zip(nonzero_indices, policy[nonzero_indices]))

    return moves_with_probabilities
