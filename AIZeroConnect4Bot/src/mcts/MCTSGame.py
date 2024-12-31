import numpy as np
from dataclasses import dataclass

from src.mcts.MCTSNode import MCTSNode
from src.Encoding import get_board_result_score
from src.settings import CURRENT_BOARD, CURRENT_GAME


@dataclass
class MCTSGameMemory:
    board: CURRENT_BOARD
    action_probabilities: np.ndarray


class MCTSGame:
    def __init__(self) -> None:
        self.board = CURRENT_GAME.get_initial_board()
        self.memory: list[MCTSGameMemory] = []
        self.root: MCTSNode = None  # type: ignore
        self.node: MCTSNode | None = None

    def get_best_child_or_back_propagate(self, c_param: float) -> MCTSNode | None:
        node = self.root

        while node.is_fully_expanded:
            node = node.best_child(c_param)

        if node.is_terminal_node:
            result = get_board_result_score(node.board)
            assert result is not None
            node.back_propagate(-result)
            return None

        return node
