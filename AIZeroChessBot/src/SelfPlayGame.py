from Framework import *

from AIZeroChessBot.src.AlphaMCTSNode import AlphaMCTSNode
from AIZeroChessBot.src.BoardEncoding import get_board_result_score


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
            node.back_propagate(get_board_result_score(node.board, self.root.board.turn))
            return None

        return node
