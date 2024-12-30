from __future__ import annotations
import os
import time

import numpy as np

from src.settings import CURRENT_BOARD, CURRENT_GAME, CURRENT_GAME_MOVE
from src.train.TrainingArgs import TrainingArgs


class AlphaMCTSNode:
    @classmethod
    def root(cls, board: CURRENT_BOARD) -> AlphaMCTSNode:
        instance = cls(policy=1.0, move_to_get_here=CURRENT_GAME.null_move, parent=None, num_played_moves=0)
        instance.board = board
        instance.number_of_visits = 1
        return instance

    def __init__(
        self, policy: float, move_to_get_here: CURRENT_GAME_MOVE, parent: AlphaMCTSNode | None, num_played_moves: int
    ) -> None:
        self.board: CURRENT_BOARD = None  # type: ignore
        self.parent = parent
        self.children: list[AlphaMCTSNode] = []
        self.move_to_get_here = move_to_get_here
        self.num_played_moves = num_played_moves  # This is the number of moves played to get to this node
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
        return not not self.children

    @property
    def is_terminal_node(self) -> bool:
        return self.board is not None and self.board.is_game_over()

    def ucb(self, c_param: float) -> float:
        assert self.parent, 'Node must have a parent'

        policy_score = c_param * np.sqrt(self.parent.number_of_visits) / (1 + self.number_of_visits)

        if self.number_of_visits > 0:
            # Q(s, a) - the average reward of the node's children from the perspective of the node's parent
            q_score = 1 - ((self.result_score / self.number_of_visits) + 1) / 2
        else:
            q_score = 0

        return policy_score * self.policy + q_score

    def expand(self, moves_with_scores: list[tuple[CURRENT_GAME_MOVE, float]]) -> None:
        self.children = [
            AlphaMCTSNode(
                policy=score,
                move_to_get_here=move,
                parent=self,
                num_played_moves=self.num_played_moves + 1,
            )
            for move, score in moves_with_scores
            if score > 0.0
        ]

        # Convert to NumPy arrays
        self.children_number_of_visits = np.array([child.number_of_visits for child in self.children], dtype=np.int16)
        self.children_result_scores = np.array([child.result_score for child in self.children], dtype=np.int16)
        self.children_policies = np.array([child.policy for child in self.children], dtype=np.float32)

    def back_propagate(self, result: float) -> None:
        self.number_of_visits += 1
        self.result_score += result
        if self.parent:
            child_index = self.parent.children.index(self)
            self.parent.children_number_of_visits[child_index] += 1
            self.parent.children_result_scores[child_index] += result
            self.parent.back_propagate(-result)

    def best_child(self, c_param: float) -> AlphaMCTSNode:
        """Selects the best child node using the UCB1 formula and initializes the best child before returning it."""

        q_score = np.zeros(len(self.children), dtype=np.float32)
        visited_children = self.children_number_of_visits > 0
        q_score[visited_children] = (
            1
            - ((self.children_result_scores[visited_children] / self.children_number_of_visits[visited_children]) + 1)
            / 2
        )

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
depth: {self.num_played_moves}
score: {self.result_score:.2f}
policy: {self.policy:.2f}
move: {self.move_to_get_here}
children: {len(self.children)}
)"""

    def show_graph(self, args: TrainingArgs):
        nodes = []
        edges = []
        self._show_graph(nodes, edges, args)
        print('Max depth, num terminal nodes, num nodes')
        print(self._collect_stats(self))

        print('Actual num unique nodes:', len(set(nodes)))

        # write in graphviz format
        with open('graph.dot', 'w') as f:
            f.write('digraph G {\n')
            for node in nodes:
                f.write(f'"{node}" [shape=box];\n')
            for edge in edges:
                f.write(f'"{edge[0]}" -> "{edge[1]}";\n')
            f.write('}\n')

        # convert to png
        current_time = time.strftime('%Y%m%d-%H%M%S')
        os.system(f'dot -Tpng graph.dot -o graph_{current_time}.png')
        # os.system(f'graph_{current_time}.png')
        # exit()

    def _collect_stats(self, node, depth=0):
        if not node.board:
            # max depth, num terminal nodes, num nodes
            return 0, 0, 0
        max_depth = depth
        num_terminal_nodes = 0
        num_nodes = 1
        for child in node.children:
            child_max_depth, child_num_terminal_nodes, child_num_nodes = self._collect_stats(child, depth + 1)
            max_depth = max(max_depth, child_max_depth)
            num_terminal_nodes += child_num_terminal_nodes
            num_nodes += child_num_nodes
        if node.board.is_game_over():
            num_terminal_nodes += 1
        return max_depth, num_terminal_nodes, num_nodes

    def graph_id(self, args: TrainingArgs):
        b = 'None'
        if self.board:
            b = CURRENT_GAME.get_canonical_board(self.board)[0] - CURRENT_GAME.get_canonical_board(self.board)[1]
        return f"""{b}
visits: {self.number_of_visits}
score: {self.result_score:.2f}
policy: {self.policy:.2f}
calc_policy: {round(self.number_of_visits / self.parent.number_of_visits, 2) if self.parent else 1}
ucb: {(round(self.ucb(args.c_param), 2)) if self.parent else "None"}"""

    def _show_graph(self, nodes, edges, args: TrainingArgs):
        if self.board is None:
            return
        nodes.append(self.graph_id(args))
        if self.parent:
            edges.append((self.parent.graph_id(args), self.graph_id(args)))
        for child in self.children:
            child._show_graph(nodes, edges, args)
