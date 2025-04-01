import numpy as np
import numpy.typing as npt

from numba import njit


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
