from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from src.games.chess.ChessBoard import ChessBoard
from src.games.chess.contract import CHESS_STATE_CONTRACT
from src.packed_planes import (
    PackedPlanePayload,
    decode_packed_planes,
    decode_packed_planes_batch,
    decode_packed_planes_into,
    encode_packed_planes,
)


C = CHESS_STATE_CONTRACT.representation.channels
H = CHESS_STATE_CONTRACT.representation.rows
W = CHESS_STATE_CONTRACT.representation.columns
NUM_BITS = C * H * W

BINARY_CHANNELS = CHESS_STATE_CONTRACT.representation.binary_channels
SCALAR_CHANNELS = CHESS_STATE_CONTRACT.representation.scalar_channels
PACKED_PLANES = CHESS_STATE_CONTRACT.representation.packed_planes

N_BB = len(BINARY_CHANNELS)
N_SCALAR = len(SCALAR_CHANNELS)
ENCODED_BYTES = PACKED_PLANES.payload_bytes

assert set(BINARY_CHANNELS).isdisjoint(SCALAR_CHANNELS)
assert N_BB == PACKED_PLANES.binary_plane_count
assert N_SCALAR == PACKED_PLANES.scalar_count
assert N_BB + N_SCALAR == C, 'Total number of channels must match the representation shape C'


def encode_board_state(state: npt.NDArray[np.int8]) -> PackedPlanePayload:
    """
    Compress a canonical chess tensor into the packed plane-major payload.

    Binary channels are serialized as little-endian 64-bit words and scalar
    channels are stored as one signed byte each after the binary section.
    """
    return encode_packed_planes(
        state,
        PACKED_PLANES,
        BINARY_CHANNELS,
        SCALAR_CHANNELS,
    )


def decode_board_state(encoded_state: PackedPlanePayload) -> npt.NDArray[np.int8]:
    """Expand one packed chess payload back into the canonical `(C, H, W)` tensor."""
    decoded = decode_packed_planes(
        encoded_state,
        PACKED_PLANES,
        BINARY_CHANNELS,
        SCALAR_CHANNELS,
    )
    return decoded


def decode_board_states(encoded_states: Sequence[PackedPlanePayload]) -> npt.NDArray[np.int8]:
    """Decode a batch of packed chess payloads with vectorized NumPy operations."""
    return decode_packed_planes_batch(
        encoded_states,
        PACKED_PLANES,
        BINARY_CHANNELS,
        SCALAR_CHANNELS,
    )


def decode_board_states_into(
    encoded_states: Sequence[PackedPlanePayload],
    output: npt.NDArray[np.float32],
) -> None:
    """Decode packed chess payloads directly into a preallocated float32 batch tensor."""
    decode_packed_planes_into(
        encoded_states,
        PACKED_PLANES,
        BINARY_CHANNELS,
        SCALAR_CHANNELS,
        output,
    )


def get_board_result_score(board: ChessBoard) -> float | None:
    """
    Return the terminal score from the current player's perspective.

    The score is `-1.0` for checkmate, `0.0` for a draw, and `None` while the
    game is still in progress.
    """
    if not board.is_game_over():
        return None

    if (winner := board.check_winner()) is not None:
        assert winner != board.current_player, 'The winner must be the opponent, sine he just played a checkmate move'
        return -1.0

    return 0.0
