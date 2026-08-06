from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
import numpy.typing as npt

from src.games.Board import Board
from src.packed_planes import PackedPlaneLayout


MoveT = TypeVar('MoveT')
BoardT = TypeVar('BoardT', bound=Board[object])


@dataclass(frozen=True)
class RepresentationDimensions:
    channels: int
    rows: int
    columns: int
    binary_channels: tuple[int, ...]
    scalar_channels: tuple[int, ...]
    packed_planes: PackedPlaneLayout


@dataclass(frozen=True)
class GameStateContract(Generic[BoardT, MoveT]):
    name: str
    action_size: int
    representation: RepresentationDimensions
    create_initial_board: Callable[[], BoardT]
    canonical_board: Callable[[BoardT], npt.NDArray[np.int8]]
    encode_move: Callable[[MoveT, BoardT], int]
    decode_move: Callable[[int, BoardT], MoveT]
    replay_piece_counts: Callable[[npt.NDArray[np.int8]], tuple[int, int]]
