from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from math import isfinite
from typing import TYPE_CHECKING, Generic, TypeVar

from src.packed_planes import PackedPlaneLayout, PackedPlanePayload
from src.util.frozen_model import FrozenModel

if TYPE_CHECKING:
    from src.replay.contracts import ReplaySample
    from src.self_play.completed_game import TerminationReason


PositionT = TypeVar('PositionT')


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


@dataclass(frozen=True)
class RepresentationDimensions:
    channels: int
    rows: int
    columns: int
    binary_channels: tuple[int, ...]
    scalar_channels: tuple[int, ...]
    packed_planes: PackedPlaneLayout


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
    def current_player(self, position: PositionT) -> Player:
        raise NotImplementedError

    @abstractmethod
    def terminal_wdl(self, position: PositionT) -> WdlTarget | None:
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

    @abstractmethod
    def transform_action_id(self, action_id: int, augmentation_index: int) -> int:
        raise NotImplementedError

    @abstractmethod
    def transform_encoded_state(
        self,
        encoded_state: PackedPlanePayload,
        augmentation_index: int,
    ) -> PackedPlanePayload:
        raise NotImplementedError

    @abstractmethod
    def transform_replay_targets(
        self,
        sample: ReplaySample,
        augmentation_index: int,
    ) -> ReplaySample:
        raise NotImplementedError
