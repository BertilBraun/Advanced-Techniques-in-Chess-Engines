import numpy as np


from src.games.Game import Board
from src.settings import CurrentGame, CurrentBoard
from src.util.timing import timeit


C, H, W = CurrentGame.representation_shape
NUM_BITS = C * H * W


def encode_board_state(state: np.ndarray) -> bytes:
    """
    state: np.ndarray of shape (C, H, W), values are 0 or 1 (any integer dtype).
    Returns a bytes object of length ceil(C*H*W/8).
    """
    flat = (state.reshape(-1) & 1).astype(np.uint8)
    packed = np.packbits(flat)
    return packed.tobytes()


def decode_board_state(b: bytes) -> np.ndarray:
    """
    b: bytes previously returned by encode_board_bytes
    shape: the original (C, H, W)
    Returns an np.int8 array of 0/1 with that shape.
    """
    expected_n_bytes = (NUM_BITS + 7) // 8

    packed = np.frombuffer(b, dtype=np.uint8)
    if packed.size < expected_n_bytes:
        packed = np.concatenate([packed, np.zeros(expected_n_bytes - packed.size, dtype=np.uint8)])
    else:
        packed = packed[:expected_n_bytes]
    flat = np.unpackbits(packed)[:NUM_BITS]
    return flat.reshape(C, H, W).astype(np.int8)


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
    legal_moves_encoded = CurrentGame.encode_moves(board.get_valid_moves(), board)
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
