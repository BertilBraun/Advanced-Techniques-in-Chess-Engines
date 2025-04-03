import numpy as np
import numpy.typing as npt

from numba import njit

from src.settings import ACTION_SIZE


# Prepare the bit masks: 1, 2, 4, ..., 2^(n_bits-1)
_BIT_MASK = 1 << np.arange(64, dtype=np.uint64)  # use uint64 to prevent overflow


def decode_board_state(state: npt.NDArray[np.uint64]) -> npt.NDArray[np.int8]:
    """Convert a tuple of integers into a binary state. Each integer represents a channel of the state. This assumes that the state is a binary state."""
    assert state.dtype == np.uint64, 'The state must be encoded as uint64 to prevent overflow'

    return _decode_board_state(state)


@njit
def _decode_board_state(state: npt.NDArray[np.uint64]) -> npt.NDArray[np.int8]:
    encoded_array = state.reshape(-1, 1)  # shape: (channels, 1)

    # Extract bits for each channel
    bits = ((encoded_array & _BIT_MASK) > 0).astype(np.int8)

    return bits.reshape(-1, 8, 8)  # shape: (channels, height, width)


def action_probabilities(visit_counts: npt.NDArray[np.uint16]) -> npt.NDArray[np.float32]:
    """Convert a list of visit counts into a probability distribution. This assumes that the visit counts are in the format (move, count)."""
    assert visit_counts.dtype == np.uint16, 'The visit counts must be encoded as uint16 to prevent overflow'

    return _action_probabilities(visit_counts)


@njit
def _action_probabilities(visit_counts: npt.NDArray[np.uint16]) -> npt.NDArray[np.float32]:
    probabilities = np.zeros(ACTION_SIZE, dtype=np.float32)

    for move, count in visit_counts:
        probabilities[move] = count

    total_count = np.sum(probabilities)
    assert total_count > 0, 'The visit counts must be non-zero to prevent division by zero'

    return probabilities / total_count
