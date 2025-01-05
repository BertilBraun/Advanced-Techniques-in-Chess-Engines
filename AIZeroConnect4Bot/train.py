import src.environ_setup  # isort:skip # noqa import first to setup environment variables and other configurations

import torch
from src.alpha_zero.train.TrainingArgs import ClusterParams
from src.cluster.ClusterManager import ClusterManager

from torch.optim import AdamW

from src.util.compile import try_compile
from src.util.log import log
from src.settings import TRAINING_ARGS, USE_GPU

from src.Network import Network
from src.cluster.ClusterAlphaZero import ClusterAlphaZero
from src.util.profiler import start_usage_logger


def get_device(cluster_manager: ClusterManager, trainers: int) -> torch.device:
    if not USE_GPU:
        return torch.device('cpu')

    if cluster_manager.rank == 0:
        # reserve the device 0 for training
        return torch.device('cuda', 0)

    # all other devices except the first one (except if we mix self play and train) are for self-play
    device_id = trainers + (cluster_manager.rank % (torch.cuda.device_count() - trainers))
    return torch.device('cuda', device_id)


if __name__ == '__main__':
    if not TRAINING_ARGS.cluster:
        TRAINING_ARGS.cluster = ClusterParams(
            num_self_play_nodes_on_cluster=1,
            num_train_nodes_on_cluster=0,
        )

    trainers = TRAINING_ARGS.cluster.num_train_nodes_on_cluster
    assert trainers in [0, 1], 'For now, only one trainer is supported'

    cluster_manager = ClusterManager(TRAINING_ARGS.cluster.num_self_play_nodes_on_cluster + trainers)
    cluster_manager.initialize()

    device = get_device(cluster_manager, trainers)

    if cluster_manager.is_root_node:
        log('Starting training')
        log('Training on:', device)
        log('Training args:')
        log(TRAINING_ARGS, use_pprint=True)

    start_usage_logger(cluster_manager.rank)

    model = Network(TRAINING_ARGS.network.num_layers, TRAINING_ARGS.network.hidden_size, device=device)
    model = try_compile(model)
    optimizer = AdamW(model.parameters(), lr=0.2, weight_decay=1e-4)

    caz = ClusterAlphaZero(model, optimizer, TRAINING_ARGS)

    if trainers == 0 and cluster_manager.rank == 0:
        caz.mix_self_play_and_train_on_cluster()
    else:
        if cluster_manager.rank < trainers:
            caz.train_on_cluster()
        else:
            caz.self_play_on_cluster()
