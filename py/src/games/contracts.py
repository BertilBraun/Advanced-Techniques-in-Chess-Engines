from __future__ import annotations

from abc import ABC, abstractmethod
from enum import IntEnum
from functools import cached_property
from math import isfinite
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np
import numpy.typing as npt
from src.games.representation import PackedPlaneLayout, PackedPlanePayload, RepresentationDimensions
from src.util.frozen_model import FrozenModel

if TYPE_CHECKING:
    from src.self_play.completed_game import TerminationReason


PositionT = TypeVar('PositionT')
TerminalOraclePositionT = TypeVar('TerminalOraclePositionT', contravariant=True)


class Player(IntEnum):
    FIRST = 1
    SECOND = -1


class WdlTarget(FrozenModel):
    win: float
    draw: float
    loss: float

    def model_post_init(self, __context: object) -> None:
        values = (self.win, self.draw, self.loss)
        if any(not isfinite(value) or value < 0.0 for value in values):
            raise ValueError('WDL values must be finite and nonnegative.')
        if abs(sum(values) - 1.0) > 1e-6:
            raise ValueError('WDL values must sum to one.')

    def reversed(self) -> WdlTarget:
        return WdlTarget(win=self.loss, draw=self.draw, loss=self.win)

    @classmethod
    def from_scalar(cls, value: float) -> WdlTarget:
        if not isfinite(value) or not -1.0 <= value <= 1.0:
            raise ValueError('WDL scalar must be finite and lie in [-1, 1].')
        remainder = 1.0 - abs(value)
        return cls(
            win=max(value, 0.0) + remainder / 3.0,
            draw=remainder / 3.0,
            loss=max(-value, 0.0) + remainder / 3.0,
        )


class TerminalOracle(ABC, Generic[TerminalOraclePositionT]):
    @abstractmethod
    def probe_wdl(self, position: TerminalOraclePositionT) -> WdlTarget | None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class GameStateContract(ABC, Generic[PositionT]):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def action_size(self) -> int:
        raise NotImplementedError

    @property
    def maximum_legal_action_count(self) -> int:
        return self.action_size

    @property
    @abstractmethod
    def representation(self) -> RepresentationDimensions:
        raise NotImplementedError

    @property
    def packed_plane_layout(self) -> PackedPlaneLayout:
        return self.representation.packed_planes

    @abstractmethod
    def initial_position(self) -> PositionT:
        raise NotImplementedError

    @abstractmethod
    def legal_action_ids(self, position: PositionT) -> tuple[int, ...]:
        raise NotImplementedError

    @abstractmethod
    def child_position(self, position: PositionT, action_id: int) -> PositionT:
        raise NotImplementedError

    @abstractmethod
    def is_irreversible_transition(
        self,
        position: PositionT,
        action_id: int,
        child: PositionT,
    ) -> bool:
        raise NotImplementedError

    @abstractmethod
    def current_player(self, position: PositionT) -> Player:
        raise NotImplementedError

    @abstractmethod
    def natural_terminal_wdl(self, position: PositionT) -> WdlTarget | None:
        raise NotImplementedError

    @abstractmethod
    def adjudicated_wdl(self, position: PositionT, reason: TerminationReason) -> WdlTarget:
        raise NotImplementedError

    @abstractmethod
    def encode_network_input(self, position: PositionT) -> PackedPlanePayload:
        raise NotImplementedError

    @property
    @abstractmethod
    def augmentation_count(self) -> int:
        raise NotImplementedError

    @cached_property
    def action_permutations(self) -> npt.NDArray[np.uint16]:
        permutations = np.empty((self.augmentation_count, self.action_size), dtype=np.uint16)
        for augmentation_index in range(self.augmentation_count):
            permutations[augmentation_index] = tuple(
                self.transform_action_id(action_id, augmentation_index) for action_id in range(self.action_size)
            )
        if permutations.shape != (self.augmentation_count, self.action_size) or permutations.dtype != np.uint16:
            raise ValueError('Action permutations must use the fixed augmentation/action shape and uint16 dtype.')
        expected = np.arange(self.action_size, dtype=np.uint16)
        if not np.array_equal(permutations[0], expected):
            raise ValueError('Identity augmentation must preserve every action ID.')
        if any(not np.array_equal(np.sort(permutation), expected) for permutation in permutations):
            raise ValueError('Every augmentation action mapping must be a bijection.')
        permutations.flags.writeable = False
        return permutations

    @abstractmethod
    def transform_decoded_states(
        self,
        states: npt.NDArray[np.float32],
        augmentation_indices: npt.NDArray[np.int64],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def transform_action_id(self, action_id: int, augmentation_index: int) -> int:
        raise NotImplementedError
