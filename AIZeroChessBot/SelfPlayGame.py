from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from AIZeroChessBot.AlphaMCTSNode import AlphaMCTSNode

from Framework import *
from AIZeroChessBot.BoardEncoding import board_result_to_score


@dataclass
class SelfPlayGameMemory:
    board: Board
    action_probabilities: NDArray[np.float32]
    turn: Color


class SelfPlayGame:
    def __init__(self) -> None:
        self.board = Board()
        self.memory: list[SelfPlayGameMemory] = []
        self.root: AlphaMCTSNode = None  # type: ignore
        self.node: AlphaMCTSNode | None = None

    def get_best_child_or_back_propagate(self, c_param: float) -> AlphaMCTSNode | None:
        node = self.root

        while node.is_fully_expanded:
            node = node.best_child(c_param)

        if node.is_terminal_node:
            node.back_propagate(-board_result_to_score(node.board))
            return None

        return node
