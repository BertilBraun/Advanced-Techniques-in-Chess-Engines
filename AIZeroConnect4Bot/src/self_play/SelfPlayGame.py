import numpy as np
from dataclasses import dataclass

from AIZeroConnect4Bot.src.AlphaMCTSNode import AlphaMCTSNode
from AIZeroConnect4Bot.src.Encoding import get_board_result_score
from AIZeroConnect4Bot.src.settings import CURRENT_BOARD, CURRENT_GAME


@dataclass
class SelfPlayGameMemory:
    board: CURRENT_BOARD
    action_probabilities: np.ndarray


class SelfPlayGame:
    def __init__(self) -> None:
        self.board = CURRENT_GAME.get_initial_board()
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
            node.back_propagate(-result)
            return None

        return node
