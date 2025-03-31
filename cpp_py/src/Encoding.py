import numpy as np
import numpy.typing as npt

from numba import njit


# Prepare the bit masks: 1, 2, 4, ..., 2^(n_bits-1)
_BIT_MASK = 1 << np.arange(64, dtype=np.uint64)  # use uint64 to prevent overflow


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

    return bits.reshape(-1, 8, 8)  # shape: (channels, height, width)
