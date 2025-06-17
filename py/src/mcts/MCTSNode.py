from __future__ import annotations

import numpy as np
from numba import njit

from src.Encoding import MoveScore
from src.settings import CurrentBoard, CurrentGame


# NOTE: more virtual loss, to avoid the same node being selected multiple times (i.e. muliply delta by 2-5?)
VIRTUAL_LOSS_DELTA = 1  # How much to increase the virtual loss by when selecting a node


class MCTSNode:
    @classmethod
    def root(cls, board: CurrentBoard) -> MCTSNode:
        instance = cls(encoded_move_to_get_here=-1, parent=None, my_child_index=-1)
        instance.board = board
        return instance

    def __init__(self, encoded_move_to_get_here: int, parent: MCTSNode | None, my_child_index: int) -> None:
        self.board: CurrentBoard = None  # type: ignore
        self.parent = parent
        self.encoded_move_to_get_here = encoded_move_to_get_here
        self.my_child_index = my_child_index
        self.children: list[MCTSNode] = []
        self.children_number_of_visits: np.ndarray  # Initialized in expand
        self.children_result_scores: np.ndarray  # Initialized in expand
        self.children_virtual_losses: np.ndarray  # Initialized in expand
        self.children_policies: np.ndarray  # Initialized in expand

    def copy(self, parent: MCTSNode | None) -> MCTSNode:
        node = MCTSNode(
            encoded_move_to_get_here=self.encoded_move_to_get_here,
            parent=parent,
            my_child_index=self.my_child_index,
        )

        if self.board is not None:
            node.board = self.board.copy()

        if self.is_fully_expanded:
            node.children = [child.copy(node) for child in self.children]
            node.children_number_of_visits = self.children_number_of_visits.copy()
            node.children_result_scores = self.children_result_scores.copy()
            node.children_virtual_losses = self.children_virtual_losses.copy()
            node.children_policies = self.children_policies.copy()

        return node

    def _maybe_init_board(self) -> None:
        """Initializes the node by creating a board if it doesn't have one."""
        if not self.board:
            if not self.parent or not self.parent.board:
                raise ValueError('Parent node must have a board')

            self.board = self.parent.board.copy()
            self.board.make_move(CurrentGame.decode_move(self.encoded_move_to_get_here, self.board))

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
            if self.parent is not None
            else self.children_number_of_visits.sum() + 1
        )

    @property
    def result_score(self) -> float:
        if self.number_of_visits == 0:
            return -float('inf')

        if self.parent is not None:
            score = self.parent.children_result_scores[self.my_child_index]
        else:
            score = np.sum(self.children_result_scores)

        return -score / self.number_of_visits

    def expand(self, encoded_moves_with_scores: list[MoveScore]) -> None:
        if self.is_fully_expanded:
            return  # Already expanded by another thread

        assert all(score > 0.0 for _, score in encoded_moves_with_scores), 'Scores must be positive'

        for encoded_move, _ in encoded_moves_with_scores:
            self.children.append(
                MCTSNode(
                    encoded_move_to_get_here=encoded_move,
                    parent=self,
                    my_child_index=len(self.children),
                )
            )

        # Store precomputed values for the children to make the best_child method faster because it's called a lot
        self.children_number_of_visits = np.zeros(len(self.children), dtype=np.int32)
        self.children_result_scores = np.zeros(len(self.children), dtype=np.float32)
        self.children_virtual_losses = np.zeros(len(self.children), dtype=np.int32)
        self.children_policies = np.array([score for _, score in encoded_moves_with_scores], dtype=np.float32)

    def add_virtual_loss(self) -> None:
        node = self

        while node.parent:
            node.parent.children_virtual_losses[node.my_child_index] += VIRTUAL_LOSS_DELTA
            node.parent.children_number_of_visits[node.my_child_index] += 1
            node = node.parent

    def back_propagate(self, result: float) -> None:
        node = self

        while node.parent:
            node.parent.children_number_of_visits[node.my_child_index] += 1
            node.parent.children_result_scores[node.my_child_index] += result
            result = -1.0 * result * 0.99  # Discount the result for the parent node
            node = node.parent

    def back_propagate_and_remove_virtual_loss(self, result: float) -> None:
        node = self

        while node.parent:
            node.parent.children_result_scores[node.my_child_index] += result
            node.parent.children_virtual_losses[node.my_child_index] -= VIRTUAL_LOSS_DELTA
            # NOTE: We do not increment the number of visits here, as this is already done in add_virtual_loss
            result = -1.0 * result * 0.99  # Discount the result for the parent node
            node = node.parent

    def best_child(self, c_param: float) -> MCTSNode:
        """Selects the best child node using the UCB1 formula and initializes the best child before returning it."""
        # NOTE moving the calculations into seperate functions slowed this down by 2x, so it's all in here
        best_child_index = _best_child_index(
            c_param=c_param,
            children_number_of_visits=self.children_number_of_visits,
            children_result_scores=self.children_result_scores,
            children_virtual_losses=self.children_virtual_losses,
            children_policies=self.children_policies,
            own_number_of_visits=self.number_of_visits,
            # Fix: use parent's result score if this is the root node, else init to loss (-1.0 (the own result score is inverted in the UCB1 formula))
            own_result_score=0.0,  # TODO -1.0,  # TODO?? Apparently not, but once: self.result_score if self.parent is None else -1.0,
        )
        best_child = self.children[best_child_index]
        best_child._maybe_init_board()
        return best_child

    def __repr__(self) -> str:
        if not self.is_fully_expanded:
            return 'MCTSNode(not expanded)'

        # sort children by policy
        children = list(
            sorted(self.children, key=lambda child: self.children_policies[child.my_child_index], reverse=True)
        )

        def entropy(node: MCTSNode) -> float:
            # calculate the entropy of the nodes visit counts
            node_number_of_visits = node.number_of_visits
            return -sum(
                visit_count / node_number_of_visits * np.log2(visit_count / node_number_of_visits)
                for visit_count in node.children_number_of_visits
                if visit_count > 0
            )

        children_str = ''

        for child in children:
            if child.number_of_visits < 5:
                continue
            children_str += f'\n\tNum visits: {child.number_of_visits} (Score: {self.children_result_scores[child.my_child_index]:.2f}), Policy: {self.children_policies[child.my_child_index]:.2f} -> {child.number_of_visits / self.number_of_visits:.2f} (Move: {str(CurrentGame.decode_move(child.encoded_move_to_get_here, self.board))})'

        return f"""MCTSNode(
{repr(self.board) if self.board else None}
visits: {self.number_of_visits}
score: {self.result_score:.2f}
entropy: {entropy(self):.2f}
children:{children_str}
)"""

        return f"""AlphaMCTSNode(
{repr(self.board) if self.board else None}
visits: {self.number_of_visits}
score: {self.result_score:.2f}
child visits: {self.children_number_of_visits.tolist()}
child policy: {np.round(self.children_policies, 2).tolist()}
child moves: {[CurrentGame.decode_move(child.encoded_move_to_get_here) for child in self.children]}
child scores: {np.round(self.children_result_scores, 2).tolist()}
best_move: {CurrentGame.decode_move(self.children[np.argmax(self.children_number_of_visits)].encoded_move_to_get_here) if self.children else None}
)"""


@njit
def _best_child_index(
    c_param: float,
    children_number_of_visits: np.ndarray,
    children_result_scores: np.ndarray,
    children_virtual_losses: np.ndarray,
    children_policies: np.ndarray,
    own_number_of_visits: int,
    own_result_score: float,
) -> int:
    visited_mask = children_number_of_visits > 0
    unvisited_mask = ~visited_mask

    result_scores = children_result_scores[visited_mask]
    virtual_losses = children_virtual_losses[visited_mask]
    number_of_visits = children_number_of_visits[visited_mask]

    # Q score is a value [-1, 1] where -1 is a loss and 1 is a win in the current position
    # We want to maximize the Q score, which means minimizing the result score of the children, as the scores in the children are from the perspective of the opponent
    q_score = np.zeros_like(children_number_of_visits, dtype=np.float32)
    # TODO q_score[visited_mask] = -1 * (result_scores + virtual_losses) / number_of_visits
    # TODO q_score[unvisited_mask] = -own_result_score  # Init to parent  # Init to loss (-1.0) for unvisited moves

    # Q score based on foersterrobert
    # Q store should now be between 0 and 1, where 0 is a loss and 1 is a win for the current player
    q_score[visited_mask] = 1 - ((((result_scores + virtual_losses) / number_of_visits) + 1) / 2)
    q_score[unvisited_mask] = -own_result_score  # Init to loss (0.0) for unvisited moves

    visits_quotient = np.sqrt(own_number_of_visits) / (1 + children_number_of_visits)

    u_score = c_param * children_policies * visits_quotient

    ucb_scores = q_score + u_score

    return np.argmax(ucb_scores).item()


def ucb(node: MCTSNode, c_param: float, own_result_score: float) -> float:
    """
    Calculate the UCB1 score for a node.

    :param node: The node to calculate the UCB1 score for.
    :param c_param: The exploration parameter.
    :return: The UCB1 score for the node.
    """
    result_score = node.result_score
    virtual_loss = node.parent.children_virtual_losses[node.my_child_index]  # type: ignore
    number_of_visits = node.parent.children_number_of_visits[node.my_child_index]  # type: ignore
    policy = node.parent.children_policies[node.my_child_index]  # type: ignore

    if number_of_visits == 0:
        q_score = -own_result_score  # Init to loss (-1.0) for unvisited moves
    else:
        q_score = 1 - ((((result_score + virtual_loss) / number_of_visits) + 1) / 2)

    visits_quotient = np.sqrt(number_of_visits) / (1 + node.parent.number_of_visits)  # type: ignore

    u_score = c_param * policy * visits_quotient

    return q_score + u_score
