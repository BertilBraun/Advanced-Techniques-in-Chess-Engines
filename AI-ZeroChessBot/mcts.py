from __future__ import annotations
from math import log, sqrt

from tqdm import tqdm

from Framework import *


def evaluation(board: Board) -> tuple[Move, float]:
    possible_moves = list(board.legal_moves)

    # TODO eval board state here to get evaluation over all possible moves and evaluation of the board state
    # TODO get the best move from the evaluation that is also a legal move
    # TODO the evaluation should be a float between -1 and 1, where -1 is a loss, 0 is a draw, and 1 is a win

    return possible_moves[0], 0.0


class MCTSState:
    def __init__(self, board: Board, turn: Color, remaining_depth: float) -> None:
        self.remaining_depth = remaining_depth
        self.board = board
        self.turn = turn

    def get_legal_actions(self) -> list[Move]:
        return list(self.board.legal_moves)

    def is_game_over(self) -> bool:
        return self.board.is_game_over()

    def game_result(self) -> float | None:
        result = self.board.result()
        if result == '1-0':
            return 1
        elif result == '0-1':
            return -1
        elif result == '1/2-1/2':
            return 0

        return None

    def create_next_state(self, move: Move) -> MCTSState:
        new_state = self.board.copy(stack=False)
        new_state.push(move)
        return MCTSState(new_state, not self.board.turn, self.remaining_depth - 1)


class MCTSNode:
    def __init__(
        self,
        state: MCTSState,
        move_to_get_here: Move,
        parent: MCTSNode | None = None,
    ) -> None:
        self.state = state
        self.parent = parent
        self.children: list[MCTSNode] = []
        self.move_to_get_here = move_to_get_here
        self._number_of_visits = 0
        self._result = 0
        self.untried_moves = self.state.get_legal_actions()

    @property
    def q(self) -> float:
        return self._result

    @property
    def n(self) -> float:
        return self._number_of_visits

    @property
    def is_terminal_node(self) -> bool:
        return self.state.is_game_over()

    @property
    def is_fully_expanded(self) -> bool:
        return len(self.untried_moves) == 0

    def expand(self) -> MCTSNode:
        move = self.untried_moves.pop()
        next_state = self.state.create_next_state(move)
        child_node = MCTSNode(next_state, parent=self, move_to_get_here=move)

        self.children.append(child_node)
        return child_node

    def rollout(self) -> float:
        current_rollout_state = self.state
        state_score = 0.0

        while not current_rollout_state.is_game_over() and current_rollout_state.remaining_depth > 0:
            move, state_score = evaluation(current_rollout_state.board)
            current_rollout_state = current_rollout_state.create_next_state(move)

        result = current_rollout_state.game_result()
        return result if result is not None else state_score

    def back_propagate(self, result: float) -> None:
        self._number_of_visits += 1.0
        self._result += result
        if self.parent:
            self.parent.back_propagate(result)

    def best_child(self, c_param: float = 0.1) -> MCTSNode:
        def weighted_score(node: MCTSNode) -> float:
            return node.q / node.n + c_param * sqrt((2 * log(self.n) / node.n))

        return max(self.children, key=weighted_score)

    def _tree_policy(self) -> MCTSNode:
        current_node = self

        while not current_node.is_terminal_node:
            if not current_node.is_fully_expanded:
                return current_node.expand()
            else:
                current_node = current_node.best_child()

        return current_node


def UCT(root_state: MCTSState, max_iter: int) -> Move:
    root = MCTSNode(root_state, Move.null(), None)
    for _ in tqdm(range(max_iter), desc='MCTS Iterations'):
        v = root._tree_policy()
        reward = v.rollout()
        v.back_propagate(reward)

    return root.best_child(c_param=0.0).move_to_get_here


class MCTSBot(ChessBot):
    def __init__(self) -> None:
        super().__init__('MCTS Bot')
        self.max_depth = 30
        self.max_iter = 100

    def think(self, board: Board) -> Move:
        root_state = MCTSState(board, board.turn, self.max_depth)
        return UCT(root_state, self.max_iter)
