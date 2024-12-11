import numpy as np
from dataclasses import dataclass

from AIZeroConnect4Bot.src.Board import Board, Color
from AIZeroConnect4Bot.src.AlphaMCTSNode import AlphaMCTSNode
from AIZeroConnect4Bot.src.Encoding import get_board_result_score


@dataclass
class SelfPlayGameMemory:
    board: Board
    action_probabilities: np.ndarray
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
            result = get_board_result_score(node.board)
            assert result is not None
            node.back_propagate(result)
            return None

        return node
