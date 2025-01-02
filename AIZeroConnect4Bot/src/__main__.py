import src.environ_setup  # isort:skip # noqa import first to setup environment variables and other configurations

import torch
from torch.optim import Adam

from src.util.log import log
from src.settings import TRAINING_ARGS

from src.Network import Network
from src.alpha_zero.AlphaZero import AlphaZero
from src.cluster.ClusterAlphaZero import ClusterAlphaZero

if __name__ == '__main__':
    log('Starting training')
    log('Training on:', 'GPU' if torch.cuda.is_available() else 'CPU')
    log('Training args:')
    log(TRAINING_ARGS, use_pprint=True)

    assert (TRAINING_ARGS.cluster.num_train_nodes_on_cluster is None) == (
        TRAINING_ARGS.cluster.num_self_play_nodes_on_cluster is None
    ), 'Either both or none of the cluster args should be set'

    if TRAINING_ARGS.cluster.num_train_nodes_on_cluster is not None:
        ClusterAlphaZero(TRAINING_ARGS).learn()
    else:
        model = Network(TRAINING_ARGS.network.num_layers, TRAINING_ARGS.network.hidden_size)
        torch.set_float32_matmul_precision('high')
        if torch.cuda.is_available():
            model: Network = torch.compile(model)  # type: ignore
        optimizer = Adam(model.parameters(), lr=0.2, weight_decay=1e-4)

        AlphaZero(model, optimizer, TRAINING_ARGS).learn()
