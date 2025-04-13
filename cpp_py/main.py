import torch
import AlphaZeroCpp
from src.settings import TRAINING_ARGS

AlphaZeroCpp.self_play_main(
    0,  # run
    TRAINING_ARGS.save_path,
    8,  # num_self_play_nodes_on_cluster
    1,  # num_self_play_gpus
)
