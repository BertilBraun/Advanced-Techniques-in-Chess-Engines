from dataclasses import dataclass
from enum import Enum, auto

from chess import Move


class TranspositionFlag(Enum):
    EXACT = auto()
    LOWER_BOUND = auto()
    UPPER_BOUND = auto()


@dataclass
class TranspositionEntry:
    hash: int = 0  # Hash of the board position
    move: Move = Move.null()  # Best move found from this position
    score: int = 0  # Evaluation score of the position
    depth: int = 0  # Depth at which this entry was generated
    flag: TranspositionFlag = TranspositionFlag.EXACT  # Flag indicating the type of score
