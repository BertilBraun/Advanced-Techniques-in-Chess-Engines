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
        self.number_of_visits = 1 if parent is None else 0
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

    def expand(self, moves_with_scores: list[tuple[Move, float]]) -> None:
        for move, score in moves_with_scores:
            child_node = AlphaMCTSNode(score, move, parent=self)
            self.children.append(child_node)

    def back_propagate(self, result: float) -> None:
        self.number_of_visits += 1
        self.result_score += result
        if self.parent:
            self.parent.back_propagate(result)

    def best_child(self, c_param: float = 0.1) -> AlphaMCTSNode:
        # Prepare arrays for vectorized calculations
        number_of_visits = np.array([child.number_of_visits for child in self.children])
        result_scores = np.array([child.result_score for child in self.children])
        policies = np.array([child.policy for child in self.children])
        parent_number_of_visits = self.number_of_visits  # Assuming this is the total visits of the parent node

        # Vectorized UCB calculations
        with np.errstate(divide='ignore', invalid='ignore'):  # Handle division by zero
            q_score = 1 - ((result_scores / number_of_visits) + 1) / 2
            q_score[np.isnan(q_score)] = 0  # Handle NaN resulting from 0/0

            ucb_scores = q_score + c_param * policies * np.sqrt(parent_number_of_visits) / (1 + number_of_visits)

        # Find the index of the child with the highest UCB score
        best_child_index = np.argmax(ucb_scores)
        self.children[best_child_index].init()
        return self.children[best_child_index]

    def ucb(self, c_param: float = 0.1) -> float:
        # Note: This is called very frequently, so we want to keep it as fast as possible
        # assert self.parent, 'Node must have a parent'

        ucb_score = self.policy * c_param * np.sqrt(self.parent.number_of_visits) / (1 + self.number_of_visits)  # type: ignore assuming self.parent is not None

        if self.number_of_visits > 0:
            # Q(s, a) - the average reward of the node's children from the perspective of the node's parent
            ucb_score += 1 - ((self.result_score / self.number_of_visits) + 1) / 2

        return ucb_score

    def __repr__(self) -> str:
        return f"""AlphaMCTSNode(
{self.board}
visits: {self.number_of_visits}
score: {self.result_score:.2f}
policy: {self.policy:.2f}
move: {self.move_to_get_here}
children: {len(self.children)}
)"""
