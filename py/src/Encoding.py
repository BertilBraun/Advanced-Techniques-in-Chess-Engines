import numpy as np
import numpy.typing as npt


from src.games.Game import Board
from src.settings import CurrentGame, CurrentBoard
from src.util.timing import timeit


C, H, W = CurrentGame.representation_shape
NUM_BITS = C * H * W

BINARY_CHANNELS = CurrentGame.binary_channels
SCALAR_CHANNELS = CurrentGame.scalar_channels

N_BB = len(BINARY_CHANNELS)  # 19
N_SCALAR = len(SCALAR_CHANNELS)  # 6
ENCODED_BYTES = N_BB * 8 + N_SCALAR  # 158

assert set(BINARY_CHANNELS).isdisjoint(SCALAR_CHANNELS)
assert N_BB * 8 + N_SCALAR == ENCODED_BYTES, 'Encoded bytes must match the sum of bit-board and scalar channels'
assert N_BB + N_SCALAR == C, 'Total number of channels must match the representation shape C'


def _plane_to_u64(plane: npt.NDArray[np.int8]) -> np.uint64:
    """Pack an (8,8) binary plane into a uint64 (little-endian bit order)."""
    bits = np.packbits(plane.reshape(-1).astype(np.uint8), bitorder='little')
    # packbits gave us 8 bytes → interpret them as one little-endian uint64
    return np.frombuffer(bits, dtype='<u8', count=1)[0]


def _u64_to_plane(bb: np.uint64) -> npt.NDArray[np.int8]:
    """Unpack uint64 back into an (8,8) binary plane."""
    bytes_ = np.asarray([bb], dtype='<u8').view(np.uint8)
    bits = np.unpackbits(bytes_, bitorder='little')[:64]
    return bits.astype(np.int8).reshape(8, 8)


# ---------- public API ---------------------------------------------------------
def encode_board_state(state: npt.NDArray[np.int8]) -> bytes:
    """
    Compress a canonical board tensor (C,W,H) into bytes.
    Bit-board channels → one uint64 each, scalar channels → one int8 each.
    """
    assert state.shape == (C, H, W)

    # 1. bit-board part --------------------------------------------------------
    bb_arr = np.empty(N_BB, dtype='<u8')
    for out_i, ch in enumerate(BINARY_CHANNELS):
        bb_arr[out_i] = _plane_to_u64(state[ch])

    # 2. scalar part -----------------------------------------------------------
    scalars = state[np.array(SCALAR_CHANNELS), 0, 0].astype(np.int8)

    # 3. concatenate and return -----------------------------------------------
    return bb_arr.tobytes() + scalars.tobytes()


def decode_board_state(buf: bytes) -> npt.NDArray[np.int8]:
    """
    Decompress bytes produced by `encode_board_state` back to (C,W,H) int8 tensor.
    Works even if trailing zero-bytes were stripped.
    """
    if len(buf) < ENCODED_BYTES:  # ← ❶ auto-pad if necessary
        buf = buf + b'\x00' * (ENCODED_BYTES - len(buf))
    else:  # ← ❷ or clip a too-long slice
        buf = buf[:ENCODED_BYTES]

    # --- split the buffer ----------------------------------------------------
    bb_arr = np.frombuffer(buf, dtype='<u8', count=N_BB)
    scalars = np.frombuffer(buf[N_BB * 8 :], dtype=np.int8, count=N_SCALAR)

    # --- rebuild the (C,H,W) tensor -----------------------------------------
    out = np.zeros((C, H, W), dtype=np.int8)

    for i, ch in enumerate(BINARY_CHANNELS):
        out[ch] = _u64_to_plane(bb_arr[i])  # unpack the bit-boards

    for i, ch in enumerate(SCALAR_CHANNELS):
        out[ch, :, :] = scalars[i]  # broadcast the scalars

    return out


def get_board_result_score(board: Board) -> float | None:
    """
    Returns the result score for the given board.

    :param board: The board to get the result score for.
    :return: The result score for the given board. -1 if the current player has lost, -0.5 to 0.5 for a draw. None if the game is not over.
    """
    if not board.is_game_over():
        return None

    if (winner := board.check_winner()) is not None:
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
