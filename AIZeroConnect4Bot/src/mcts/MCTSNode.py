from __future__ import annotations
import math

import numpy as np

from src.settings import CurrentBoard, CurrentGame, CurrentGameMove


class MCTSNode:
    @classmethod
    def root(cls, board: CurrentBoard) -> MCTSNode:
        instance = cls(policy=1.0, move_to_get_here=CurrentGame.null_move, parent=None)
        instance.board = board
        instance.number_of_visits = 1
        return instance

    def __init__(self, policy: float, move_to_get_here: CurrentGameMove, parent: MCTSNode | None) -> None:
        self.board: CurrentBoard = None  # type: ignore
        self.parent = parent
        self.children: list[MCTSNode] = []
        self.move_to_get_here = move_to_get_here
        self.number_of_visits = 0
        self.result_score = 0
        self.policy = policy

    def init(self) -> None:
        """Initializes the node by creating a board if it doesn't have one."""
        if not self.board:
            if not self.parent or not self.parent.board:
                raise ValueError('Parent node must have a board')

            self.board = self.parent.board.copy()
            self.board.make_move(self.move_to_get_here)

    @property
    def is_fully_expanded(self) -> bool:
        return len(self.children) > 0

    @property
    def is_terminal_node(self) -> bool:
        assert self.board, 'Node must have a board'
        return self.board.is_game_over()

    def ucb(self, c_param: float) -> float:
        assert self.parent, 'Node must have a parent'

        policy_score = c_param * math.sqrt(self.parent.number_of_visits) / (1 + self.number_of_visits)

        if self.number_of_visits > 0:
            # Q(s, a) - the average reward of the node's children from the perspective of the node's parent
            q_score = 1 - ((self.result_score / self.number_of_visits) + 1) / 2
        else:
            q_score = 0

        return policy_score * self.policy + q_score

    def expand(self, moves_with_scores: list[tuple[CurrentGameMove, float]]) -> None:
        self.children = [
            MCTSNode(
                policy=score,
                move_to_get_here=move,
                parent=self,
            )
            for move, score in moves_with_scores
            if score > 0.0
        ]

        # Store precomputed values for the children to make the best_child method faster because it's called a lot
        self.children_number_of_visits = np.zeros(len(self.children), dtype=np.uint16)
        self.children_q_scores = np.zeros(len(self.children), dtype=np.float32)
        self.children_policies = np.array([child.policy for child in self.children], dtype=np.float32)

    def back_propagate(self, result: float) -> None:
        self.number_of_visits += 1
        self.result_score += result
        if self.parent:
            child_index = self.parent.children.index(self)
            self.parent.children_number_of_visits[child_index] += 1
            self.parent.children_q_scores[child_index] = 1 - ((self.result_score / self.number_of_visits) + 1) / 2
            self.parent.back_propagate(-result)

    def best_child(self, c_param: float) -> MCTSNode:
        """Selects the best child node using the UCB1 formula and initializes the best child before returning it."""
        policy_score = c_param * np.sqrt(self.number_of_visits) / (1 + self.children_number_of_visits)

        ucb_scores = self.children_q_scores + self.children_policies * policy_score

        # Select the best child
        best_child = self.children[np.argmax(ucb_scores)]
        best_child.init()
        return best_child

    def __repr__(self) -> str:
        return f"""AlphaMCTSNode(
{self.board}
visits: {self.number_of_visits}
score: {self.result_score:.2f}
policy: {self.policy:.2f}
move: {self.move_to_get_here}
children: {len(self.children)}
)"""
