import torch
from torch.optim import Adam

from AIZeroConnect4Bot.src.ClusterAlphaZero import ClusterAlphaZero
from AIZeroConnect4Bot.src.settings import TRAINING_ARGS
from AIZeroConnect4Bot.src.AlphaZero import AlphaZero
from AIZeroConnect4Bot.src.Network import Network

if __name__ == '__main__':
    from pprint import pprint

    model = Network()
    model: Network = torch.compile(model)  # type: ignore
    optimizer = Adam(model.parameters(), lr=0.2, weight_decay=1e-4)

    print('Starting training')
    print('Training on:', model.device)
    print('Number of parameters:', sum(p.numel() for p in model.parameters()))
    print('Training args:')
    pprint(TRAINING_ARGS.__dict__)

    assert not (TRAINING_ARGS.num_train_nodes_on_cluster is None) ^ (
        TRAINING_ARGS.num_self_play_nodes_on_cluster is None
    ), 'Either both or none of the cluster args should be set'

    if TRAINING_ARGS.num_train_nodes_on_cluster is not None:
        ClusterAlphaZero(model, optimizer, TRAINING_ARGS).learn()
    else:
        AlphaZero(model, optimizer, TRAINING_ARGS).learn()
