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

if __name__ == '__main__':
    if not TRAINING_ARGS.cluster:
        TRAINING_ARGS.cluster = ClusterParams(
            num_self_play_nodes_on_cluster=1,
            num_train_nodes_on_cluster=0,
        )
    assert TRAINING_ARGS.cluster.num_train_nodes_on_cluster in [0, 1], 'For now, only one trainer is supported'

    cluster_manager = ClusterManager(
        TRAINING_ARGS.cluster.num_self_play_nodes_on_cluster + TRAINING_ARGS.cluster.num_train_nodes_on_cluster
    )
    cluster_manager.initialize()

    if cluster_manager.is_root_node:
        log('Starting training')
        log('Training on:', 'GPU' if USE_GPU else 'CPU')
        log('Training args:')
        log(TRAINING_ARGS, use_pprint=True)

        start_usage_logger()  # TODO everyone

    device = torch.device('cuda', cluster_manager.rank % torch.cuda.device_count()) if USE_GPU else torch.device('cpu')

    model = Network(TRAINING_ARGS.network.num_layers, TRAINING_ARGS.network.hidden_size, device=device)
    model = try_compile(model)
    optimizer = AdamW(model.parameters(), lr=0.2, weight_decay=1e-4)

    ClusterAlphaZero(model, optimizer, TRAINING_ARGS, cluster_manager.rank).learn()
