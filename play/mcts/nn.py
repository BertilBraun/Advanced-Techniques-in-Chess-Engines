
import time
from math import tan
from typing import List

import chess
import numpy as np
from tensorflow.keras.models import load_model
from util import board_to_nums

from mcts.mcts import ChessState

model = load_model(r'D:\Projects\ChessBot\play\mcts\001model184.h5')

total_time = 0
total_evaluations = 0


def evaluate_position_with_NN(state: ChessState, moves: List[chess.Move]) -> List[float]:
    global total_evaluations, total_time, model

    boards = [state.state.copy(stack=False) for _ in range(len(moves))]
    for i, move in enumerate(moves):
        boards[i].push(move)

    boards_array = [
        board_to_nums(board, state.turn).flatten()
        for board in boards
    ]

    start = time.time()

    predictions = model(np.asarray(boards_array), training=False).numpy()

    total_time += time.time() - start
    total_evaluations += len(boards)

    return [
        tan(prediction[0] if state.turn == chess.WHITE else -prediction[0])
        for prediction in predictions
    ]


def print_and_reset_stats():
    global total_evaluations, total_time
    print("Total Evaluations:", total_evaluations)
    print("Time:", total_time)
    total_evaluations = 0
    total_time = 0
