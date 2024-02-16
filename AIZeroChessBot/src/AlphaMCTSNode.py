from __future__ import annotations

import numpy as np

from Framework import *


class AlphaMCTSNode:
    def __init__(
        self,
        policy: float = 1.0,
        move_to_get_here: Move = Move.null(),
        parent: AlphaMCTSNode | None = None,
    ) -> None:
        self.board: Board = None  # type: ignore
        self.parent = parent
        self.children: list[AlphaMCTSNode] = []
        self.move_to_get_here = move_to_get_here
        self.number_of_visits = 1.0 if parent is None else 0.001
        self.result_score = 0.0
        self.policy = policy

    def init(self) -> None:
        """Initializes the node by creating a board if it doesn't have one."""
        if not self.board:
            if not self.parent or not self.parent.board:
                raise ValueError('Parent node must have a board')

            self.board = self.parent.board.copy(stack=False)
            self.board.push(self.move_to_get_here)

    @property
    def is_terminal_node(self) -> bool:
        return self.board is not None and self.board.is_game_over()

    @property
    def is_fully_expanded(self) -> bool:
        return len(self.children) > 0

    def ucb(self, c_param: float = 0.1) -> float:
        assert self.parent, 'Node must have a parent'

        ucb_score = self.policy * c_param * np.sqrt(self.parent.number_of_visits) / (1 + self.number_of_visits)

        if self.number_of_visits > 0:
            # Q(s, a) - the average reward of the node's children from the perspective of the node's parent
            ucb_score += 1 - ((self.result_score / self.number_of_visits) + 1) / 2

        return ucb_score

    def expand(self, moves_with_scores: list[tuple[Move, float]]) -> None:
        self.children = [AlphaMCTSNode(score, move, parent=self) for move, score in moves_with_scores]

        # Convert to NumPy arrays
        self.children_number_of_visits = np.array([child.number_of_visits for child in self.children], dtype=np.float32)
        self.children_result_scores = np.array([child.result_score for child in self.children], dtype=np.float32)
        self.children_policies = np.array([child.policy for child in self.children], dtype=np.float32)

    def back_propagate(self, result: float) -> None:
        self.number_of_visits += 1.0
        self.result_score += result
        if self.parent:
            child_index = self.parent.children.index(self)
            self.parent.children_number_of_visits[child_index] += 1
            self.parent.children_result_scores[child_index] += result
            self.parent.back_propagate(result)

    def best_child(self, c_param: float = 0.1) -> AlphaMCTSNode:
        """Selects the best child node using the UCB1 formula and initializes the best child before returning it."""

        q_score = 1 - ((self.children_result_scores / self.children_number_of_visits) + 1) / 2
        policy_score = c_param * np.sqrt(self.number_of_visits) / (1 + self.children_number_of_visits)

        ucb_scores = q_score + self.children_policies * policy_score

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
