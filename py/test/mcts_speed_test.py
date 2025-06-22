import multiprocessing
import src.environ_setup  # noqa # isort:skip # This import is necessary for setting up the environment variables

import time
import random
from src.mcts.MCTS import MCTS
from src.train.TrainingArgs import MCTSParams
from src.cluster.InferenceClient import InferenceClient
from src.util.save_paths import create_model, create_optimizer, save_model_and_optimizer

import os

from src.games.chess.ChessSettings import TRAINING_ARGS, CurrentBoard

os.environ['OMP_NUM_THREADS'] = '1'  # Limit the number of threads to 1 for OpenMP
os.environ['MKL_NUM_THREADS'] = '1'  # Limit the number of threads to 1 for MKL

import torch  # noqa

from AlphaZeroCpp import test_mcts_speed_cpp, test_eval_mcts_speed_cpp

mcts_params = MCTSParams(
    num_searches_per_turn=600,
    fast_searches_proportion_of_full_searches=1.0,
    playout_cap_randomization=1.0,
    num_parallel_searches=4,
    dirichlet_alpha=0.3,
    dirichlet_epsilon=0.0,
    c_param=1.0,
    num_threads=multiprocessing.cpu_count(),
    percentage_of_node_visits_to_keep=0.0,
)


def test_mcts_speed_py(num_boards: int, num_iterations: int) -> None:
    client = InferenceClient(0, TRAINING_ARGS.network, TRAINING_ARGS.save_path)
    client.update_iteration(0)
    mcts = MCTS(client, mcts_params)

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
        _ = mcts.search(boards)
        end = time.time()

        duration = end - start
        total_time += duration

        print(f'Iteration {i + 1}: Inference time: {duration:.6f} seconds')

    print(f'Total time: {total_time:.6f} seconds')
    print(f'Average time per iteration: {total_time / num_iterations:.6f} seconds')
    print(f'Average time per board: {total_time / (num_iterations * num_boards):.6f} seconds')


if __name__ == '__main__':
    print('Starting mcts speed test...')
    num_iterations = 10  # Number of iterations to run the test
    print(f'Number of iterations: {num_iterations}')

    network = create_model(TRAINING_ARGS.network, torch.device('cpu'))
    optimizer = create_optimizer(network, TRAINING_ARGS.training.optimizer)
    save_model_and_optimizer(network, optimizer, 0, TRAINING_ARGS.save_path)

    for num_boards in [64, 1]:
        print(f'Number of boards: {num_boards}')
        for num_threads in [multiprocessing.cpu_count(), 16, 1]:
            mcts_params.num_threads = num_threads
            print(f'Number of threads: {num_threads}')

            print('Python:', '=' * 20)
            test_mcts_speed_py(num_boards, num_iterations)
            print('Finished Python mcts speed test.')

            print('C++:', '=' * 20)
            test_mcts_speed_cpp(
                num_boards,
                num_iterations,
                mcts_params.num_searches_per_turn,
                mcts_params.num_parallel_searches,
                mcts_params.num_threads,
            )
            print('Finished C++ mcts speed test.')

            print('C++ eval:', '=' * 20)
            test_eval_mcts_speed_cpp(
                num_boards,
                num_iterations,
                mcts_params.num_searches_per_turn,
                mcts_params.num_parallel_searches,
                mcts_params.num_threads,
            )
            print('Finished C++ eval mcts speed test.')
