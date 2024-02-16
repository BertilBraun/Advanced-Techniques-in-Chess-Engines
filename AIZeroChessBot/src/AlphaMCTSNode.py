from __future__ import annotations

import numpy as np

from Framework import *


class AlphaMCTSNode:
    def __init__(
        self,
        state: Board,
        policy: float = 1.0,
        move_to_get_here: Move = Move.null(),
        parent: AlphaMCTSNode | None = None,
    ) -> None:
        self.board = state
        self.parent = parent
        self.children: list[AlphaMCTSNode] = []
        self.move_to_get_here = move_to_get_here
        self.number_of_visits = 1 if parent is None else 0
        self.result_score = 0.0
        self.policy = policy

    @property
    def is_terminal_node(self) -> bool:
        return self.board.is_game_over()

    @property
    def is_fully_expanded(self) -> bool:
        return len(self.children) > 0

    def ucb(self, c_param: float = 0.1) -> float:
        # Note: This is called very frequently, so we want to keep it as fast as possible
        # assert self.parent, 'Node must have a parent'

        ucb_score = self.policy * c_param * np.sqrt(self.parent.number_of_visits) / (1 + self.number_of_visits)  # type: ignore assuming self.parent is not None

        if self.number_of_visits > 0:
            # Q(s, a) - the average reward of the node's children from the perspective of the node's parent
            ucb_score += 1 - ((self.result_score / self.number_of_visits) + 1) / 2

        return ucb_score

    def expand(self, moves_with_scores: list[tuple[Move, float]]) -> None:
        for move, score in moves_with_scores:
            new_board = self.board.copy(stack=False)
            new_board.push(move)

            child_node = AlphaMCTSNode(new_board, score, move, parent=self)

            self.children.append(child_node)

    def back_propagate(self, result: float) -> None:
        self.number_of_visits += 1
        self.result_score += result
        if self.parent:
            self.parent.back_propagate(result)

    def best_child(self, c_param: float = 0.1) -> AlphaMCTSNode:
        best = self.children[0]
        best_score = best.ucb(c_param)

        for child in self.children[1:]:
            score = child.ucb(c_param)
            if score > best_score:
                best = child
                best_score = score

        return best

        return max(self.children, key=lambda node: node.ucb(c_param))

    def __repr__(self) -> str:
        return f"""AlphaMCTSNode(
{self.board}
visits: {self.number_of_visits}
score: {self.result_score:.2f}
policy: {self.policy:.2f}
move: {self.move_to_get_here}
children: {len(self.children)}
)"""
