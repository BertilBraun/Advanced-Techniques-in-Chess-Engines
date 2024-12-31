import numpy as np
from dataclasses import dataclass

from src.mcts.MCTSNode import MCTSNode
from src.Encoding import get_board_result_score
from src.settings import CURRENT_BOARD, CURRENT_GAME


@dataclass
class SelfPlayGameMemory:
    board: CURRENT_BOARD
    action_probabilities: np.ndarray


class SelfPlayGame:
    def __init__(self) -> None:
        self.board = CURRENT_GAME.get_initial_board()
        self.memory: list[SelfPlayGameMemory] = []
        self.root: MCTSNode = None  # type: ignore
        self.node: MCTSNode | None = None
