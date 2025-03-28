from __future__ import annotations

import chess
import numpy as np
from typing import NamedTuple

from src.games.ChessBoard import ChessBoard, ChessMove

BOARD_LENGTH = 8
BOARD_SIZE = BOARD_LENGTH * BOARD_LENGTH
ENCODING_CHANNELS = 13 + 1  # 12 for pieces + 1 for castling rights + 1 for color


class ChessGame:
    @staticmethod
    def representation_shape() -> tuple[int, int, int]:
        return (ENCODING_CHANNELS, BOARD_LENGTH, BOARD_LENGTH)
