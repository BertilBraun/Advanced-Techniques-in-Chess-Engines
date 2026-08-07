from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.packed_planes import PackedPlaneLayout


@dataclass(frozen=True)
class RepresentationDimensions:
    channels: int
    rows: int
    columns: int
    binary_channels: tuple[int, ...]
    scalar_channels: tuple[int, ...]
    packed_planes: PackedPlaneLayout


class GameStateContract(ABC):
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
