import src.environ_setup  # isort:skip # noqa import first to setup environment variables and other configurations

from torch.optim import AdamW

from src.util.compile import try_compile
from src.util.log import log
from src.settings import TRAINING_ARGS, USE_GPU

from src.Network import Network
from src.alpha_zero.AlphaZero import AlphaZero
from src.cluster.ClusterAlphaZero import ClusterAlphaZero
from src.util.profiler import start_usage_logger

if __name__ == '__main__':
    log('Starting training')
    log('Training on:', 'GPU' if USE_GPU else 'CPU')
    log('Training args:')
    log(TRAINING_ARGS, use_pprint=True)

    start_usage_logger()

    if TRAINING_ARGS.cluster:
        ClusterAlphaZero(TRAINING_ARGS).learn()
    else:
        model = Network(TRAINING_ARGS.network.num_layers, TRAINING_ARGS.network.hidden_size, device=None)
        model = try_compile(model)
        optimizer = AdamW(model.parameters(), lr=0.2, weight_decay=1e-4)

        AlphaZero(model, optimizer, TRAINING_ARGS).learn()
