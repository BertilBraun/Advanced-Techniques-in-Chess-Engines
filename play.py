#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import time
from math import log, sqrt

import chess
import chess.pgn
import numpy as np
from chessboard import display
from tensorflow.keras.models import load_model

# In[2]:


# In[12]:


def board_to_bitfields(board: chess.Board, turn: chess.Color) -> np.ndarray:

    pieces_array = []
    colors = [chess.WHITE, chess.BLACK]
    for c in colors if turn == chess.WHITE else colors[::-1]:
        for p in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING):
            pieces_array.append(board.pieces_mask(p, c))

    return np.array(pieces_array).astype(np.int64)


def bitfield_to_nums(bitfield: np.int64, white: bool) -> np.ndarray:

    board_array = np.zeros(64).astype(np.float32)

    for i in np.arange(64).astype(np.int64):
        if bitfield & (1 << i):
            board_array[i] = 1. if white else -1.

    return board_array


def bitfields_to_nums(bitfields: np.ndarray) -> np.ndarray:
    bitfields = bitfields.astype(np.int64)

    boards = []

    for i, bitfield in enumerate(bitfields):
        boards.append(bitfield_to_nums(bitfield, i < 6))

    return np.array(boards).astype(np.float32)


def board_to_nums(board: chess.Board, turn: chess.Color) -> np.ndarray:

    return bitfields_to_nums(board_to_bitfields(board, turn))


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
            result = evaluate_position(state, state.turn)

            # state is terminal. Update node with result from POV of node.playerJustMoved
            node.Update(result)
            node = node.parent

    return sorted(rootnode.children, key=lambda c: c.visits)[-1].move


total_evaluations = 0


def evaluate_position(board: chess.Board, turn: chess.Color) -> float:
    board_array = board_to_nums(board, turn)
    global total_evaluations
    total_evaluations += 1
    return model.predict(np.asarray([board_array.flatten()]))


model = load_model('training/001model025.h5')


def mcts_player(board: chess.Board):
    for move_choice in board.legal_moves:
        copy = board.copy()
        copy.push(move_choice)
        if copy.is_game_over():
            board.push(move_choice)
            return

    move = UCT(board, itermax=100, depthmax=30)
    board.push(move)
    return move


def mcts_player_with_stats(board: chess.Board):
    global total_evaluations
    total_evaluations = 0

    start = time.time()
    print("MCTS Player:", mcts_player(board))
    print("Total Evaluations:", total_evaluations)
    print("Time:", time.time() - start)


def human_player(board: chess.Board):
    while True:
        move = input("Input Your Move:")
        if move == "q":
            raise KeyboardInterrupt
        try:
            board.push_san(move)
            break
        except Exception as e:
            print(e)


def play_game(player1, player2):
    board = chess.Board()
    display.start(board.fen())

    while not board.is_game_over():
        if board.turn == chess.WHITE:
            player1(board)
        else:
            player2(board)

        # clear_output(wait=True)
        display.update(board.fen())
        # display(board)
        time.sleep(0.1)

    game = chess.pgn.Game.from_board(board)
    print(game)


play_game(mcts_player_with_stats, human_player)
