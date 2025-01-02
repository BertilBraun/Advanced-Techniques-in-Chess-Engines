from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class MCTSParams:
    num_searches_per_turn: int
    """This is the maximum number of searches to run the MCTS algorithm in self-play.
    I.e. the number of times that a node is expanded and evaluated in the MCTS algorithm to get the next move to play.
    I.e. we continue to play the game util one of the players wins or the game is a draw. At each move we run the MCTS algorithm to get the next move to play for the current player. Here we run the MCTS algorithm num_searches times to get the next move to play.
    """

    dirichlet_epsilon: float
    """This is the epsilon value to use for the dirichlet noise to add to the root node in self-play to encourage exploration. I.e. the percentage of the resulting policy, that should be the dirichlet noise. The rest is the policy from the neural network. lerp(policy, dirichlet_noise, factor=dirichlet_epsilon)"""

    dirichlet_alpha: Callable[[int], float]
    """This is the alpha value function to use for the dirichlet noise to add to the root node in self-play to encourage exploration. The iteration is passed in and the alpha value to use should be returned. Apparently the value should be around 10/number_of_actions. So for Connect4 with 7 columns this value should be around 1.5, and for chess with 400 possible moves this value should be around 0.025"""

    c_param: float
    """This is the c parameter to use for the UCB1 formula in the MCTS algorithm in self-play. It is used to balance exploration and exploitation in the MCTS algorithm. Values between 1 and 6 seem sensible. The higher the value the more exploration is favored over exploitation."""


@dataclass
class NetworkParams:
    num_layers: int
    """This is the number of layers to use for the neural network"""

    hidden_size: int
    """This is the dimension of the hidden layers to use for the neural network. Typically 32-64 for simpler games like TicTacToe, 128-256 for medium complexity games like Connect4 and 512-1024 for complex games like Chess"""


@dataclass
class SelfPlayParams:
    temperature: float
    """This is the sampling temperature to use for in self-play to sample new moves from the policy. The higher the temperature the more random the moves are. The lower the temperature the more the moves are like the policy. A temperature of 1 is the same as the policy, a temperature of 0 is the argmax of the policy. Typically 1-2 for exploration and 0.1-0.5 for exploitation"""

    num_parallel_games: int
    """This is the number of games to run in parallel for self-play."""

    num_games_per_iteration: int
    """This is the number of self-play iterations to run per iteration. I.e. the number of games to play and collect data for to train with"""


@dataclass
class ClusterParams:
    num_train_nodes_on_cluster: int
    """This is the number of separate nodes on the cluster to use to parallelize the training. None means to not use a cluster and train on the local machine"""

    num_self_play_nodes_on_cluster: int
    """This is the number of separate nodes on the cluster to use to parallelize the self-play. This should most likely be 16x or more the number of nodes used for training to minimize the wait time for the training nodes to get new data to train with. None means to not use a cluster and train on the local machine"""


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


@dataclass
class EvaluationParams:
    num_searches_per_turn: int
    """This is the number of searches to run the MCTS algorithm in the evaluation. This is used to evaluate the model against itself to see how well it is doing. The higher the number the more accurate the evaluation but the slower the evaluation. Typically 20-50 for evaluation"""

    num_games: int
    """This is the number of games to play for the evaluation. The more games the more accurate the evaluation but the longer the evaluation. Typically 32-256 for evaluation"""

    every_n_iterations: int
    """This is the number of iterations between each evaluation. The higher the number the less often the evaluation is run. Typically 2-10 for evaluation"""


@dataclass
class TrainingArgs:
    save_path: str
    """This is the path to save the model, datasamples, training logs, etc. to after each iteration"""

    num_iterations: int
    """This is the number of iterations to run first self-play then train"""

    mcts: MCTSParams
    network: NetworkParams
    self_play: SelfPlayParams
    training: TrainingParams
    cluster: Optional[ClusterParams] = None
    evaluation: Optional[EvaluationParams] = None
