import torch
from torch.optim import Adam

from AIZeroConnect4Bot.src.util.log import log
from AIZeroConnect4Bot.src.settings import TRAINING_ARGS

from AIZeroConnect4Bot.src.Network import Network
from AIZeroConnect4Bot.src.AlphaZero import AlphaZero
from AIZeroConnect4Bot.src.cluster.ClusterAlphaZero import ClusterAlphaZero

if __name__ == '__main__':
    model = Network()
    torch.set_float32_matmul_precision('high')
    # model: Network = torch.compile(model)  # type: ignore
    optimizer = Adam(model.parameters(), lr=0.2, weight_decay=1e-4)

    log('Starting training')
    log('Training on:', model.device)
    log('Number of parameters:', sum(p.numel() for p in model.parameters()))
    log('Training args:')
    log(TRAINING_ARGS.__dict__, use_pprint=True)

    assert not (TRAINING_ARGS.num_train_nodes_on_cluster is None) ^ (
        TRAINING_ARGS.num_self_play_nodes_on_cluster is None
    ), 'Either both or none of the cluster args should be set'

    if TRAINING_ARGS.num_train_nodes_on_cluster is not None:
        ClusterAlphaZero(model, optimizer, TRAINING_ARGS).learn()
    else:
        AlphaZero(model, optimizer, TRAINING_ARGS).learn()
