from dataclasses import dataclass
from typing import Callable, Optional

from src.mcts.MCTSArgs import MCTSArgs


@dataclass
class TrainingArgs:
    """
    This class contains the training arguments for the AlphaZero algorithm.

    These are hyperparameters that can be tuned to improve the performance of the AlphaZero algorithm. They are not very intuitive and are usually found by trial and error. See the references for more information on the hyperparameters."""

    mcts_args: MCTSArgs

    num_iterations: int
    """This is the number of iterations to run first self-play then train"""

    num_self_play_games_per_iteration: int
    """This is the number of self-play iterations to run per iteration. I.e. the number of games to play and collect data for to train with"""

    num_epochs: int
    """This is the number of epochs to train with one batch of self-play data"""

    batch_size: int
    """This is the size of the batch to train with"""

    sampling_window: Callable[[int], int]
    """This is a function that returns the sampling window to use for the self-play data. The sampling window is the number of most recent games to sample from to train with. This is used to phase out old data that is no longer useful to train with. The function should take the current iteration as input and return the sampling window to use for that iteration.
    Example:
    def sampling_window(current_iteration: int) -> int:
        if current_iteration < 5:
            return 4
        return 4 + (current_iteration - 5) // 2
    """

    learning_rate: Callable[[int], float]
    """This is a function that returns the learning rate to use for the training. The function should take the current iteration as input and return the learning rate to use for that iteration.
    Example:
    def learning_rate(current_iteration: int) -> float:
        if current_iteration < 10:
            return 0.2
        if current_iteration < 20:
            return 0.02
        return 0.002
    """

    learning_rate_scheduler: Callable[[float, float], float]
    """This is a function that returns the learning rate to use for the training. The function should take the batch percentage and the base learning rate as input and return the learning rate to use for that batch. This is used to implement a learning rate scheduler that changes the learning rate during training. The batch percentage is the percentage of the batch that has been processed so far. The base learning rate is the learning rate to use for the current iteration.
    Example:
    def learning_rate_scheduler(batch_percentage: float, base_lr: float) -> float:
        min_lr = base_lr / 10
        return min_lr + (base_lr - min_lr) * batch_percentage
    """

    save_path: str = ''
    """This is the path to save the model to after each iteration"""

    num_train_nodes_on_cluster: Optional[int] = None
    """This is the number of separate nodes on the cluster to use to parallelize the training. None means to not use a cluster and train on the local machine"""

    num_self_play_nodes_on_cluster: Optional[int] = None
    """This is the number of separate nodes on the cluster to use to parallelize the self-play. This should most likely be 16x or more the number of nodes used for training to minimize the wait time for the training nodes to get new data to train with. None means to not use a cluster and train on the local machine"""
