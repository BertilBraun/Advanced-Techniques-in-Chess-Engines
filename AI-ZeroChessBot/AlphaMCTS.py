from __future__ import annotations
from math import sqrt
import torch

from tqdm import tqdm

from Framework import *


# TODO exploration vs exploitation - how does that work here? How do we balance the two?
# TODO how to do the self-play part of the training?
# TODO why is there a State and a Node class? What is the difference between the two?
# TODO base the architecture on the AlphaZero paper but with less layers

"""
https://en.wikipedia.org/wiki/AlphaZero

AlphaZero only searches ~80.000 positions per second, while Stockfish searches ~70.000.000 positions per second.

Training
AlphaZero was trained solely via self-play, using 5,000 first-generation TPUs to generate the games and 64 second-generation TPUs to train the neural networks. In parallel, the in-training AlphaZero was periodically matched against its benchmark (Stockfish, Elmo, or AlphaGo Zero) in brief one-second-per-move games to determine how well the training was progressing. DeepMind judged that AlphaZero's performance exceeded the benchmark after around four hours of training for Stockfish, two hours for Elmo, and eight hours for AlphaGo Zero.
"""


class AlphaMCTSState:
    def __init__(self, board: Board) -> None:
        self.board = board

    def is_terminal(self) -> bool:
        return self.board.is_game_over()

    def make_move(self, move: Move) -> AlphaMCTSState:
        new_board = self.board.copy(stack=False)
        new_board.push(move)
        return AlphaMCTSState(new_board)

    @torch.no_grad()
    def evaluation(self) -> tuple[list[tuple[Move, float]], float]:
        possible_moves = list(self.board.legal_moves)

        # TODO eval board state here to get evaluation over all possible moves and evaluation of the board state
        # TODO filter the moves from the evaluation that are also legal moves
        # TODO the evaluation should be a float between -1 and 1, where -1 is a loss, 0 is a draw, and 1 is a win

        return [(possible_moves[0], 1.0)], 0.0


class AlphaMCTSNode:
    def __init__(
        self,
        state: AlphaMCTSState,
        policy: float,
        move_to_get_here: Move,
        parent: AlphaMCTSNode | None,
    ) -> None:
        self.state = state
        self.parent = parent
        self.children: list[AlphaMCTSNode] = []
        self.move_to_get_here = move_to_get_here
        self.number_of_visits = 0
        self.result_score = 0.0
        self.policy = policy

    @property
    def is_terminal_node(self) -> bool:
        return self.state.is_terminal()

    @property
    def is_fully_expanded(self) -> bool:
        return self.children == [] and not self.is_terminal_node

    def ucb(self, c_param: float = 0.1) -> float:
        assert self.parent, 'Node must have a parent'

        ucb_score = self.policy * c_param * sqrt(self.parent.number_of_visits) / (1 + self.number_of_visits)

        if self.number_of_visits > 0:
            # Q(s, a) - the average reward of the node's children from the perspective of the node's parent
            ucb_score += 1 - ((self.result_score / self.number_of_visits) + 1) / 2

        return ucb_score

    def expand(self, moves_with_scores: list[tuple[Move, float]]) -> None:
        for move, score in moves_with_scores:
            next_state = self.state.make_move(move)
            child_node = AlphaMCTSNode(next_state, score, move, parent=self)

            self.children.append(child_node)

    def back_propagate(self, result: float) -> None:
        self.number_of_visits += 1
        self.result_score += result
        if self.parent:
            self.parent.back_propagate(result)

    def best_child(self, c_param: float = 0.1) -> AlphaMCTSNode:
        return max(self.children, key=lambda node: node.ucb(c_param))

    def iterate(self) -> None:
        current_node = self

        while not current_node.is_terminal_node:
            if not current_node.is_fully_expanded:
                moves_with_scores, result = self.state.evaluation()
                current_node.expand(moves_with_scores)
                self.back_propagate(result)
                break
            else:
                current_node = current_node.best_child()


def UCT(root_state: AlphaMCTSState, max_iter: int) -> Move:
    root = AlphaMCTSNode(root_state, 0.0, Move.null(), None)

    for _ in tqdm(range(max_iter), desc='Alpha MCTS Iterations'):
        root.iterate()

    return root.best_child(c_param=0.0).move_to_get_here


class AlphaMCTSBot(ChessBot):
    def __init__(self) -> None:
        super().__init__('Alpha MCTS Bot')
        self.max_iter = 100

    def think(self, board: Board) -> Move:
        root_state = AlphaMCTSState(board)
        return UCT(root_state, self.max_iter)
