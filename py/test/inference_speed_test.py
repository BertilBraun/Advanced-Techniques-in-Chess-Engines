import random
import time
from src.cluster.InferenceClient import InferenceClient
import src.environ_setup  # noqa # isort:skip # This import is necessary for setting up the environment variables

import os

from src.games.chess.ChessSettings import TRAINING_ARGS, CurrentBoard
from src.util.save_paths import create_model, create_optimizer, save_model_and_optimizer

os.environ['OMP_NUM_THREADS'] = '1'  # Limit the number of threads to 1 for OpenMP
os.environ['MKL_NUM_THREADS'] = '1'  # Limit the number of threads to 1 for MKL

# This ensures, that the seperate processes spawned by torch.multiprocessing do not interfere with each other by using more than one core. Since we are using as many processes as cores for workers, we need to limit the number of threads to 1 for each process. Otherwise, we would use more than one core per process, which would lead to a lot of context switching and slow down the training.

import torch  # noqa

from AlphaZeroCpp import test_inference_speed_cpp


def test_inference_speed_py(num_boards: int, num_iterations: int) -> None:
    client = InferenceClient(0, TRAINING_ARGS.network, TRAINING_ARGS.save_path)
    client.update_iteration(0)
    total_time = 0.0
    for i in range(num_iterations):
        boards = []
        while len(boards) < num_boards:
            board = CurrentBoard()
            for _ in range(30):
                moves = board.get_valid_moves()
                if not moves:
                    break
                board.make_move(random.choice(moves))

            if board.is_game_over():
                continue
            boards.append(board)

        start = time.time()
        _ = client.inference_batch(boards)
        end = time.time()

        duration = end - start
        total_time += duration

        print(f'Iteration {i + 1}: Inference time: {duration:.6f} seconds')

    print(f'Total time: {total_time:.6f} seconds')
    print(f'Average time per iteration: {total_time / num_iterations:.6f} seconds')
    print(f'Average time per board: {total_time / (num_iterations * num_boards):.6f} seconds')


if __name__ == '__main__':
    print('Starting inference speed test...')
    num_boards = 256  # Number of boards to test in each iteration
    num_iterations = 20  # Number of iterations to run the test
    print(f'Number of boards: {num_boards}')
    print(f'Number of iterations: {num_iterations}')

    network = create_model(TRAINING_ARGS.network, torch.device('cpu'))
    optimizer = create_optimizer(network, TRAINING_ARGS.training.optimizer)
    save_model_and_optimizer(network, optimizer, 0, TRAINING_ARGS.save_path)

    print('Python:', '=' * 20)
    test_inference_speed_py(num_boards, num_iterations)
    print('Finished Python inference speed test.')

    print('C++:', '=' * 20)
    test_inference_speed_cpp(num_boards, num_iterations)
    print('Finished C++ inference speed test.')
