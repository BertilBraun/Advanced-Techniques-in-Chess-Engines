from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class NetworkDimensions:
    channels: int
    rows: int
    columns: int
    actions: int
    outcomes: int = 3

    def __post_init__(self) -> None:
        if min(self.channels, self.rows, self.columns, self.actions, self.outcomes) <= 0:
            raise ValueError('Network dimensions must be positive.')


@dataclass(frozen=True)
class PackedPlaneLayout:
    """Game-owned dimensions for one canonical packed binary-plus-scalar layout."""

    board_size: int
    binary_plane_count: int
    scalar_count: int

    def __post_init__(self) -> None:
        if self.board_size <= 0:
            raise ValueError('Packed-plane board size must be positive.')
        if self.binary_plane_count < 0 or self.scalar_count < 0:
            raise ValueError('Packed-plane channel counts must be nonnegative.')

    @property
    def bit_count(self) -> int:
        return self.board_size * self.board_size

    @property
    def word_count(self) -> int:
        return (self.bit_count + 63) // 64

    @property
    def binary_bytes(self) -> int:
        return self.binary_plane_count * self.word_count * 8

    @property
    def payload_bytes(self) -> int:
        return self.binary_bytes + self.scalar_count

    @property
    def padding_bits(self) -> int:
        return self.word_count * 64 - self.bit_count

    def validate_payload_length(self, payload: bytes) -> bytes:
        if len(payload) != self.payload_bytes:
            raise ValueError(f'Packed-plane payload must be exactly {self.payload_bytes} bytes, got {len(payload)}.')
        return payload

    def empty_value(self) -> PackedPlanePayload:
        return PackedPlanePayload(self.validate_payload_length(bytes(self.payload_bytes)))

    def value(self, payload: bytes) -> PackedPlanePayload:
        return PackedPlanePayload(self.validate_payload_length(payload))


@dataclass(frozen=True)
class PackedPlanePayload:
    """Immutable contiguous packed state value used by replay and batch code."""

    payload: bytes

    def __bytes__(self) -> bytes:
        return self.payload

    def __len__(self) -> int:
        return len(self.payload)

    def memory_bytes(self) -> int:
        return sys.getsizeof(self) + sys.getsizeof(self.payload)


@dataclass(frozen=True)
class RepresentationDimensions:
    channels: int
    rows: int
    columns: int
    binary_channels: tuple[int, ...]
    scalar_channels: tuple[int, ...]
    packed_planes: PackedPlaneLayout


def _validate_state_shape(
    state: npt.NDArray[np.int8],
    layout: PackedPlaneLayout,
    binary_channels: tuple[int, ...],
    scalar_channels: tuple[int, ...],
) -> None:
    expected_shape = (
        layout.binary_plane_count + layout.scalar_count,
        layout.board_size,
        layout.board_size,
    )
    if state.shape != expected_shape:
        raise ValueError(f'Packed-plane state must have shape {expected_shape}, got {state.shape}.')
    if len(binary_channels) != layout.binary_plane_count or len(scalar_channels) != layout.scalar_count:
        raise ValueError('Packed-plane channels disagree with the declared layout.')


def encode_packed_planes(
    state: npt.NDArray[np.int8],
    layout: PackedPlaneLayout,
    binary_channels: tuple[int, ...],
    scalar_channels: tuple[int, ...],
) -> PackedPlanePayload:
    """Pack binary planes first and append scalar bytes after the binary section."""
    _validate_state_shape(state, layout, binary_channels, scalar_channels)
    plane_byte_count = layout.word_count * 8
    binary_payload = bytearray(layout.binary_bytes)
    for plane_index, channel_index in enumerate(binary_channels):
        plane = state[channel_index].astype(np.uint8, copy=False).reshape(-1)
        packed_bits = np.packbits(plane, bitorder='little')
        plane_offset = plane_index * plane_byte_count
        binary_payload[plane_offset : plane_offset + len(packed_bits)] = packed_bits.tobytes()
    scalar_payload = state[np.array(scalar_channels), 0, 0].astype(np.int8, copy=False).tobytes()
    return layout.value(bytes(binary_payload) + scalar_payload)


def decode_packed_planes(
    encoded_state: PackedPlanePayload,
    layout: PackedPlaneLayout,
    binary_channels: tuple[int, ...],
    scalar_channels: tuple[int, ...],
) -> npt.NDArray[np.int8]:
    """Decode one packed payload into an `int8` tensor."""
    decoded_float = np.zeros(
        (
            layout.binary_plane_count + layout.scalar_count,
            layout.board_size,
            layout.board_size,
        ),
        dtype=np.float32,
    )
    decode_packed_planes_into(
        (encoded_state,), layout, binary_channels, scalar_channels, decoded_float[np.newaxis, ...]
    )
    return decoded_float.astype(np.int8, copy=False)


def _normalized_payload_bytes(
    encoded_states: Sequence[PackedPlanePayload],
    layout: PackedPlaneLayout,
) -> npt.NDArray[np.uint8]:
    if not encoded_states:
        return np.empty((0, layout.payload_bytes), dtype=np.uint8)
    # Replay stores one contiguous payload per sample, so batching is just concatenation.
    payload = b''.join(layout.validate_payload_length(state.payload) for state in encoded_states)
    return np.frombuffer(payload, dtype=np.uint8).reshape(len(encoded_states), layout.payload_bytes)


def decode_packed_planes_batch(
    encoded_states: Sequence[PackedPlanePayload],
    layout: PackedPlaneLayout,
    binary_channels: tuple[int, ...],
    scalar_channels: tuple[int, ...],
) -> npt.NDArray[np.int8]:
    """Decode a batch of packed payloads into an `int8` tensor."""
    decoded_float = np.zeros(
        (
            len(encoded_states),
            layout.binary_plane_count + layout.scalar_count,
            layout.board_size,
            layout.board_size,
        ),
        dtype=np.float32,
    )
    decode_packed_planes_into(encoded_states, layout, binary_channels, scalar_channels, decoded_float)
    return decoded_float.astype(np.int8, copy=False)


def decode_packed_planes_into(
    encoded_states: Sequence[PackedPlanePayload],
    layout: PackedPlaneLayout,
    binary_channels: tuple[int, ...],
    scalar_channels: tuple[int, ...],
    output: npt.NDArray[np.float32],
) -> None:
    """Decode packed payloads directly into a caller-owned float32 output tensor."""
    expected_shape = (
        len(encoded_states),
        layout.binary_plane_count + layout.scalar_count,
        layout.board_size,
        layout.board_size,
    )
    if output.shape != expected_shape or output.dtype != np.float32:
        raise ValueError(f'Packed-plane decode output must have shape {expected_shape} and float32 dtype.')
    output.fill(0.0)
    if not encoded_states:
        return

    encoded = _normalized_payload_bytes(encoded_states, layout)
    plane_byte_count = layout.word_count * 8
    binary_bytes = encoded[:, : layout.binary_bytes].reshape(
        len(encoded_states),
        layout.binary_plane_count,
        plane_byte_count,
    )
    binary_bits = np.unpackbits(binary_bytes, axis=2, bitorder='little')
    # Padding bits must stay zero so C++ and Python have one canonical byte layout.
    if layout.padding_bits and np.any(binary_bits[:, :, layout.bit_count :]):
        raise ValueError('Packed-plane payload contains nonzero padding bits.')
    trimmed_bits = binary_bits[:, :, : layout.bit_count]
    scalar_values = encoded[:, layout.binary_bytes :].view(np.int8)
    output[:, binary_channels] = trimmed_bits.reshape(
        len(encoded_states),
        layout.binary_plane_count,
        layout.board_size,
        layout.board_size,
    )
    output[:, scalar_channels] = scalar_values[:, :, np.newaxis, np.newaxis]
