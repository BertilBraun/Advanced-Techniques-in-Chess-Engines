if __name__ == '__main__':
    import torch.multiprocessing as mp

    # set the start method to spawn for multiprocessing
    # this is required for the C++ self play process
    # and should be set before importing torch.multiprocessing
    # otherwise it will not work on Windows
    mp.set_start_method('spawn')

import os

from src.train.TrainingArgs import NetworkParams
from src.util.save_paths import create_model, create_optimizer, model_save_path, save_model_and_optimizer

os.environ['OMP_NUM_THREADS'] = '1'  # Limit the number of threads to 1 for OpenMP
os.environ['MKL_NUM_THREADS'] = '1'  # Limit the number of threads to 1 for MKL

import torch

torch.manual_seed(42)  # Set the random seed for PyTorch
torch.set_num_threads(1)  # Limit the number of threads to 1 for PyTorch
torch.set_num_interop_threads(1)  # Limit the number of inter-op threads to 1 for PyTorch

torch.autograd.set_detect_anomaly(True)


import chess
from AlphaZeroCpp import (
    MCTS,
    MCTSParams,
    InferenceClientParams,
    MCTSResults,
    INVALID_NODE,
)


network = create_model(NetworkParams(2, 64), torch.device('cpu'))
optimizer = create_optimizer(network)

save_model_and_optimizer(network, optimizer, 0, 'models')

model_path = model_save_path(0, 'models')
model_path = model_path.with_suffix('.jit.pt')


client_args = InferenceClientParams(
    device_id=0,
    currentModelPath=str(model_path),
    maxBatchSize=256,  # maybe 512
)

mcts_args = MCTSParams(
    num_parallel_searches=4,
    c_param=1.4,
    dirichlet_alpha=0.03,
    dirichlet_epsilon=0.25,
    node_reuse_discount=0.8,
    min_visit_count=5,
    num_threads=8,
)

mcts = MCTS(client_args, mcts_args)

# Suppose we want to run 800 sims from the initial position,
# and we have no “previous node,” so we pass INVALID_NODE:
boards = [
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
     INVALID_NODE,
     800)
]

results: MCTSResults = mcts.search(boards)
for r in results.results:
    print("eval =", r.result)
    for (uci_move, cnt) in r.visits:
        move_obj = chess.Move.from_uci(uci_move)
        print(f"  {uci_move} visited {cnt} times → chess.Move = {move_obj}")
