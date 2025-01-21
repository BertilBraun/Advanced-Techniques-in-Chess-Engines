from __future__ import annotations

import numpy as np

from src.settings import CurrentBoard, CurrentGame, CurrentGameMove


class MCTSNode:
    @classmethod
    def root(cls, board: CurrentBoard) -> MCTSNode:
        instance = cls(move_to_get_here=CurrentGame.null_move, parent=None, my_child_index=-1)
        instance.board = board
        return instance

    def __init__(self, move_to_get_here: CurrentGameMove, parent: MCTSNode | None, my_child_index: int) -> None:
        self.board: CurrentBoard = None  # type: ignore
        self.parent = parent
        self.move_to_get_here = move_to_get_here
        self.my_child_index = my_child_index
        self.children: list[MCTSNode] = []
        self.children_number_of_visits: np.ndarray  # Initialized in expand
        self.children_result_scores: np.ndarray  # Initialized in expand
        self.children_virtual_losses: np.ndarray  # Initialized in expand
        self.children_policies: np.ndarray  # Initialized in expand

    def _maybe_init_board(self) -> None:
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

    @property
    def number_of_visits(self) -> int:
        return (
            self.parent.children_number_of_visits[self.my_child_index]
            if self.parent
            else self.children_number_of_visits.sum() + 1
        )

    @property
    def result_score(self) -> float:
        if self.number_of_visits == 0:
            return -float('inf')
        return (
            self.parent.children_result_scores[self.my_child_index]
            if self.parent
            else np.sum(self.children_result_scores)
        ) / self.number_of_visits

    def expand(self, moves_with_scores: list[tuple[CurrentGameMove, float]]) -> None:
        if self.is_fully_expanded:
            return  # Already expanded by another thread

        for move, score in moves_with_scores:
            if score > 0.0:
                self.children.append(MCTSNode(move_to_get_here=move, parent=self, my_child_index=len(self.children)))

        # Store precomputed values for the children to make the best_child method faster because it's called a lot
        self.children_number_of_visits = np.zeros(len(self.children), dtype=np.int32)
        self.children_result_scores = np.zeros(len(self.children), dtype=np.float32)
        self.children_virtual_losses = np.zeros(len(self.children), dtype=np.int32)
        self.children_policies = np.array([score for _, score in moves_with_scores if score > 0.0], dtype=np.float32)

    def update_virtual_losses(self, delta: int) -> None:
        node = self

        while node.parent:
            node.parent.children_virtual_losses[node.my_child_index] += delta
            node.parent.children_number_of_visits[node.my_child_index] += delta
            node = node.parent

    def back_propagate(self, result: float) -> None:
        node = self

        while node.parent:
            node.parent.children_number_of_visits[node.my_child_index] += 1
            node.parent.children_result_scores[node.my_child_index] += result
            result = -result
            node = node.parent

    def best_child(self, c_param: float) -> MCTSNode:
        """Selects the best child node using the UCB1 formula and initializes the best child before returning it."""
        # NOTE moving the calculations into seperate functions slowed this down by 2x, so it's all in here
        positive_visits_mask = self.children_number_of_visits > 0

        result_scores = self.children_result_scores[positive_visits_mask]
        virtual_losses = self.children_virtual_losses[positive_visits_mask]
        number_of_visits = self.children_number_of_visits[positive_visits_mask]

        q_score = np.zeros(len(self.children), dtype=np.float32)
        q_score[positive_visits_mask] = 1 - ((result_scores + virtual_losses) / number_of_visits + 1) / 2

        visits_quotient = np.sqrt(self.number_of_visits) / (1 + self.children_number_of_visits)

        u_score = c_param * self.children_policies * visits_quotient

        ucb_scores = q_score + u_score

        best_child_index = np.argmax(ucb_scores)
        best_child = self.children[best_child_index]
        best_child._maybe_init_board()
        return best_child

    def __repr__(self) -> str:
        if not self.is_fully_expanded:
            return 'MCTSNode(not expanded)'
        return f"""AlphaMCTSNode(
{repr(self.board) if self.board else None}
visits: {self.number_of_visits}
score: {self.result_score:.2f}
child visits: {self.children_number_of_visits}
child policy: {np.round(self.children_policies, 2)}
child moves: {[child.move_to_get_here for child in self.children]}
child scores: {np.round(self.children_result_scores, 2)}
best_move: {self.children[np.argmax(self.children_number_of_visits)].move_to_get_here if self.children else None}
)"""
