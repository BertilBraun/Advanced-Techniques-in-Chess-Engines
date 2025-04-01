from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class NetworkParams:
    num_layers: int
    """This is the number of layers to use for the neural network"""

    hidden_size: int
    """This is the dimension of the hidden layers to use for the neural network. Typically 32-64 for simpler games like TicTacToe, 128-256 for medium complexity games like Connect4 and 512-1024 for complex games like Chess"""


@dataclass
class ClusterParams:
    num_self_play_nodes_on_cluster: int
    """This is the number of separate nodes on the cluster to use to parallelize the self-play. This should most likely be 16x or more the number of nodes used for training to minimize the wait time for the training nodes to get new data to train with."""


@dataclass
class TrainingParams:
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
        return lerp(min_lr, base_lr, batch_percentage)
    """

    num_workers: int = 4
    """This is the number of workers to use for the dataloader to load the self-play data. The higher the number the faster the data is loaded but the more memory is used. Typically 0-4 for training. From experience with this project, 0 seems to work best mostly."""


@dataclass
class EvaluationParams:
    num_games: int
    """This is the number of games to play for the evaluation. The more games the more accurate the evaluation but the longer the evaluation. Typically 32-256 for evaluation"""

    every_n_iterations: int
    """This is the number of iterations between each evaluation. The higher the number the less often the evaluation is run. Typically 2-10 for evaluation"""

    dataset_path: str
    """This is the path to the dataset to use for the evaluation. The dataset should contain self-play data to evaluate the model against. The more data the more accurate the evaluation but the longer the evaluation. Typically a few hundred to a few thousand games for evaluation"""


@dataclass
class TrainingArgs:
    save_path: str
    """This is the path to save the model, datasamples, training logs, etc. to after each iteration"""

    num_iterations: int
    """This is the number of iterations to run first self-play then train"""

    num_games_per_iteration: int
    """This is the number of self-play games to run per iteration. I.e. the number of games to play and collect data for to train with"""

    network: NetworkParams
    training: TrainingParams
    cluster: ClusterParams
    evaluation: Optional[EvaluationParams] = None
