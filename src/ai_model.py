from math import sqrt, log
import random
import chess

import numpy as np

from util import board_to_nums
from keras.models import load_model

model = load_model('../dataset/model3.h5')


class Node:
    def __init__(self, state: chess.Board, move: chess.Move = None, parent=None):
        self.move = move
        self.state = state
        self.parent = parent
        self.unexplored_moves = list(self.state.legal_moves)
        self.children = []
        self.visits = 0
        self.wins = 0

    def add_child(self, state, move):
        child_node = Node(state, move, self)
        self.children.append(child_node)
        self.unexplored_moves.remove(move)
        return child_node

    def UCT_select_child(self):
        s = sorted(
            self.children,
            key=lambda c:
                c.wins / c.visits + sqrt(2 * log(self.visits) / c.visits)
        )
        return s[-1]

    def Update(self, result: float):
        self.visits += 1
        self.wins += result


def UCT(rootstate: chess.Board, itermax: int, depthmax: int) -> chess.Move:
    rootnode = Node(state=rootstate)
    for i in range(itermax):
        node = rootnode
        depth = 0
        state = rootstate.copy()

        # Select
        while node.unexplored_moves == [] and node.children != []:  # node is fully expanded and non-terminal
            node = node.UCT_select_child()
            state.push(node.move)

        # Expand
        # if we can expand (i.e. state/node is non-terminal)
        if node.unexplored_moves != []:
            m = random.choice(node.unexplored_moves)
            state.push(m)
            node = node.add_child(state, m)  # add child and descend tree
            depth += 1

        # Rollout - this can often be made orders of magnitude quicker using a state.GetRandomMove() function
        while list(state.legal_moves) != [] and depth < depthmax:  # while state is non-terminal
            state.push(random.choice(list(state.legal_moves)))
            depth += 1

        # Backpropagate
        while node != None:  # backpropagate from the expanded node and work back to the root node
            result = evaluate_position(state)

            # state is terminal. Update node with result from POV of node.playerJustMoved
            node.Update(result)
            node = node.parent

    return sorted(rootnode.children, key=lambda c: c.visits)[-1].move


def evaluate_position(board: chess.Board) -> float:
    board_array = board_to_nums(board)
    return model.predict(np.asarray([board_array]))
