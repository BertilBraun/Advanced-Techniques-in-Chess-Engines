import src.environ_setup  # noqa # isort:skip # This import is necessary for setting up the environment variables

from src.cluster.CommanderProcess import CommanderProcess
from src.util.log import log
from src.settings import TRAINING_ARGS, USE_GPU


if __name__ == '__main__':
    log('Starting training')
    log('Training on:', 'GPU' if USE_GPU else 'CPU')
    log('Training args:')
    log(TRAINING_ARGS, use_pprint=True)

    commander = CommanderProcess(
        num_self_play_nodes=TRAINING_ARGS.cluster.num_self_play_nodes_on_cluster,
        num_inference_nodes=TRAINING_ARGS.cluster.num_inference_nodes_on_cluster,
    )

    commander.run()
    log('Training finished')
