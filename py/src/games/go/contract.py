from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import numpy.typing as npt

from AlphaZeroCpp import GoPlayer, GoPosition7, GoPosition9, GoRules

from src.neural_network import NetworkDimensions
from src.games.contracts import GameStateContract, RepresentationDimensions
from src.packed_planes import (
    PackedPlaneLayout,
    PackedPlanePayload,
    decode_packed_planes,
    decode_packed_planes_into,
    encode_packed_planes,
)


NativeGoPosition = GoPosition7 | GoPosition9


class GoSymmetryIndex(IntEnum):
    IDENTITY = 0
    ROTATE_90 = 1
    ROTATE_180 = 2
    ROTATE_270 = 3
    REFLECT = 4
    REFLECT_ROTATE_90 = 5
    REFLECT_ROTATE_180 = 6
    REFLECT_ROTATE_270 = 7


@dataclass(frozen=True)
class GoStateContract(GameStateContract):
    board_size: int
    history_length: int = 8

    def __post_init__(self) -> None:
        if self.board_size not in (7, 9):
            raise ValueError('Go state contract supports only 7x7 and 9x9 boards.')
        if self.history_length != 8:
            raise ValueError('Go state contract supports exactly eight history positions.')

    @property
    def binary_channels(self) -> tuple[int, ...]:
        return tuple(range(self.history_length * 2))

    @property
    def scalar_channels(self) -> tuple[int, ...]:
        return (self.history_length * 2,)

    @property
    def channels(self) -> int:
        return len(self.binary_channels) + len(self.scalar_channels)

    @property
    def action_size(self) -> int:
        return self.board_size**2 + 1

    @property
    def pass_action(self) -> int:
        return self.board_size**2

    @property
    def packed_planes(self) -> PackedPlaneLayout:
        return PackedPlaneLayout(self.board_size, len(self.binary_channels), len(self.scalar_channels))

    @property
    def network_dimensions(self) -> NetworkDimensions:
        return NetworkDimensions(self.channels, self.board_size, self.board_size, self.action_size)

    @property
    def name(self) -> str:
        return 'go'

    @property
    def representation(self) -> RepresentationDimensions:
        return RepresentationDimensions(
            channels=self.channels,
            rows=self.board_size,
            columns=self.board_size,
            binary_channels=self.binary_channels,
            scalar_channels=self.scalar_channels,
            packed_planes=self.packed_planes,
        )

    def initial_position(self, rules: GoRules) -> NativeGoPosition:
        return GoPosition7(rules) if self.board_size == 7 else GoPosition9(rules)

    def packed_position(self, position: NativeGoPosition) -> PackedPlanePayload:
        if position.board_size != self.board_size:
            raise ValueError('Native Go position board size disagrees with its Python contract.')
        return self.packed_planes.value(bytes(position.packed_encoding()))

    def transform_action(self, action_id: int, symmetry: GoSymmetryIndex) -> int:
        if not 0 <= action_id < self.action_size:
            raise ValueError('Go action lies outside the action space.')
        if action_id == self.pass_action:
            return action_id
        x = action_id % self.board_size
        y = action_id // self.board_size
        maximum = self.board_size - 1
        match symmetry:
            case GoSymmetryIndex.IDENTITY:
                transformed_x, transformed_y = x, y
            case GoSymmetryIndex.ROTATE_90:
                transformed_x, transformed_y = maximum - y, x
            case GoSymmetryIndex.ROTATE_180:
                transformed_x, transformed_y = maximum - x, maximum - y
            case GoSymmetryIndex.ROTATE_270:
                transformed_x, transformed_y = y, maximum - x
            case GoSymmetryIndex.REFLECT:
                transformed_x, transformed_y = maximum - x, y
            case GoSymmetryIndex.REFLECT_ROTATE_90:
                transformed_x, transformed_y = maximum - y, maximum - x
            case GoSymmetryIndex.REFLECT_ROTATE_180:
                transformed_x, transformed_y = x, maximum - y
            case GoSymmetryIndex.REFLECT_ROTATE_270:
                transformed_x, transformed_y = y, x
        return transformed_y * self.board_size + transformed_x

    def transform_state(self, encoded_state: PackedPlanePayload, symmetry: GoSymmetryIndex) -> PackedPlanePayload:
        state = decode_packed_planes(
            encoded_state,
            self.packed_planes,
            self.binary_channels,
            self.scalar_channels,
        )
        binary = state[: len(self.binary_channels)]
        rotation = int(symmetry) % 4
        if int(symmetry) >= int(GoSymmetryIndex.REFLECT):
            binary = np.flip(binary, axis=2)
        if rotation:
            binary = np.rot90(binary, k=-rotation, axes=(1, 2))
        transformed = state.copy()
        transformed[: len(self.binary_channels)] = binary
        return encode_packed_planes(
            transformed,
            self.packed_planes,
            self.binary_channels,
            self.scalar_channels,
        )

    def decode_batch_into(
        self,
        states: tuple[PackedPlanePayload, ...],
        output: npt.NDArray[np.float32],
    ) -> None:
        decode_packed_planes_into(
            states,
            self.packed_planes,
            self.binary_channels,
            self.scalar_channels,
            output,
        )

    @staticmethod
    def player_sign(player: GoPlayer) -> int:
        return 1 if player == GoPlayer.BLACK else -1
