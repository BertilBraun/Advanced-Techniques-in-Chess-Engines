from dataclasses import dataclass
from typing import Callable, Literal, Optional


@dataclass
class MCTSParams:
    """This class contains the arguments for the MCTS algorithm."""

    num_searches_per_turn: int
    """This is the maximum number of searches to run the MCTS algorithm in self-play. 
    I.e. the number of times that a node is expanded and evaluated in the MCTS algorithm to get the next move to play.
    I.e. we continue to play the game util one of the players wins or the game is a draw. At each move we run the MCTS algorithm to get the next move to play for the current player. Here we run the MCTS algorithm num_searches times to get the next move to play.
    """

    fast_searches_proportion_of_full_searches: float
    """This is the proportion of searches that a fast search should take compared to a full search. I.e. if this is 0.5, then a fast search will take half the time of a full search. This is used to speed up the MCTS algorithm by using a fast search for certain moves, e.g. when the game is in a late stage - these fast searched moves will not be used as training data, but will speed up the game generation, allowing to play more games in the same time. Typically 1/2 - 1/10."""

    playout_cap_randomization: float
    """This is the proportion of searches that should be searched completely, i.e. with a full search. This is used to complete games faster, as a good proportion of the moves can be searched quickly, and only moves which are selected for full searches will be searched with the full MCTS budget and therefore also added as training data. For fast searches, the complete Search Tree is reused and no new noise is added. For full searches, the node is also reused, but discouted by `percentage_of_node_visits_to_keep` and new noise is added to the root node. Typically 0.1-0.5."""

    num_parallel_searches: int
    """This is the number of parallel strands which should parallelize the MCTS algorithm. Higher values enable more parallelism and thus faster search. However, it also increases the exploration of the search tree. Values between 1 and 16 seem sensible."""

    dirichlet_epsilon: float
    """This is the epsilon value to use for the dirichlet noise to add to the root node in self-play to encourage exploration. I.e. the percentage of the resulting policy, that should be the dirichlet noise. The rest is the policy from the neural network. lerp(policy, dirichlet_noise, factor=dirichlet_epsilon)"""

    dirichlet_alpha: float
    """This is the alpha value function to use for the dirichlet noise to add to the root node in self-play to encourage exploration. The iteration is passed in and the alpha value to use should be returned. Apparently the value should be around 10/number_of_actions. So for Connect4 with 7 columns this value should be around 1.5, and for chess with 400 possible moves this value should be around 0.025"""

    c_param: float
    """This is the c parameter to use for the UCB1 formula in the MCTS algorithm in self-play. It is used to balance exploration and exploitation in the MCTS algorithm. Values between 1 and 6 seem sensible. The higher the value the more exploration is favored over exploitation."""

    num_threads: int
    """The number of parallel search threads on the Cpp side"""

    percentage_of_node_visits_to_keep: float
    """Factor by which the visits and result score of a reused Tree Node should be discounted by. In range [0..1]"""

    min_visit_count: int = 0
    """The minimum number of visits that each root child should recieve. Typically this value is < 5 or in proportion to the num_searches_per_turn. This is used to ensure that the MCTS algorithm has explored the search tree enough to make a good decision. If the number of visits is too low, the MCTS algorithm might not explore enough to learn the best moves to play."""


@dataclass
class NetworkParams:
    num_layers: int
    """This is the number of layers to use for the neural network"""

    hidden_size: int
    """This is the dimension of the hidden layers to use for the neural network. Typically 32-64 for simpler games like TicTacToe, 128-256 for medium complexity games like Connect4 and 512-1024 for complex games like Chess"""

    se_positions: tuple[int, ...]
    """This is the positions of the residual blocks to upgrade to Squeeze-and-Excitation blocks. This is used to improve the performance of the neural network by adding a Squeeze-and-Excitation block after each residual block in the specified positions. The positions are 0-based indices of the residual blocks, e.g. (0, 2, 4) means that the first, third and fifth residual blocks will be upgraded to Squeeze-and-Excitation blocks."""

    num_policy_channels: int = 4
    num_value_channels: int = 2
    value_fc_size: int = 48


@dataclass
class SelfPlayParams:
    mcts: MCTSParams

    num_parallel_games: int
    """This is the number of games to run in parallel for self-play."""

    num_moves_after_which_to_play_greedy: int
    """After this many moves, the self-play search will play greedily, i.e. it will choose the move with the highest probability according to the policy. Before this number of moves, the self-play search will play according to the temperature, i.e. it will choose moves with a probability distribution that is a mix of the policy and the dirichlet noise. This is to keep the exploration high in the beginning of the game and then play out as well as possible to reduce noise in the backpropagated final game results."""

    portion_of_samples_to_keep: float
    """This is the portion of samples to keep in the self-play dataset. This is used to reduce the size of the dataset and to keep only some of the moves that were played in self-play. This reduces the risk of overfitting to given lines of play and keeps the dataset diverse. This comes at the cost of longer self-play times, as fewer samples are kept. Typically 0.1-0.3 for games like Hex, Connect4, Go, as their board positions are very static and slightly larger 0.4-0.7 for games like Chess, as their board positions are more dynamic and the dataset is larger. A value of 1.0 means that all samples are kept, which is not recommended as it leads to overfitting and a very large dataset."""

    only_store_sampled_moves: bool = False
    """This is a flag to indicate whether states which are greedily sampled (after num_moves_after_which_to_play_greedy) should be stored in the self-play dataset. If this is set to True, only the moves that were sampled from the policy will be stored in the self-play dataset. If this is set to False, all moves that were played in self-play will be stored in the self-play dataset."""

    starting_temperature: float = 1.25
    """This is the sampling temperature to use for in self-play to sample new moves from the policy. The higher the temperature the more random the moves are. The lower the temperature the more the moves are like the policy. A temperature of 1 is the same as the policy, a temperature of 0 is the argmax of the policy. Typically 1-2 for exploration and 0.1-0.5 for exploitation. This value is linearly interpolated to the final_temperature over the first num_moves_after_which_to_play_greedy moves."""

    game_outcome_discount_per_move: float = 0.00
    """This is the discount factor to use for the game outcome per move. This is used to reduce the impact of the game outcome on the score of the moves played in the very beginning of the game, as the game outcome (especially in amateur games) is not very correlated with the moves played in the beginning of the game. The higher the value, the more the game outcome is discounted. Typically 0.001-0.01 for self-play."""

    final_temperature: float = 0.1
    """This is the final temperature to use for in self-play to sample new moves from the policy after num_moves_after_which_to_play_greedy moves. The higher the temperature the more random the moves are. The lower the temperature the more the moves are like the policy. A temperature of 1 is the same as the policy, a temperature of 0 is the argmax of the policy. See ``starting_temperature`` for more details."""

    result_score_weight: float = 0.5
    """This is the weight to use for the interpolation between final game outcome as score and the mcts result score. Weight of 0 is only the final game outcome, weight of 1 is only the mcts result score."""

    num_games_after_which_to_write: int = 5
    """This is the number of games to collect before writing them to disk. Smaller values will write more often but will be slower. Larger values will write less often but will be faster. The larger the value, the longer the training delay might be, if not enough games are collected. Typically 5-50 for self-play."""

    resignation_threshold: float = -0.85
    """This is the threshold to use for the resignation of a game. If the mcts result score is below this threshold the game is resigned. The lower the threshold the more games are resigned. The higher the threshold the less games are resigned. Typically -0.85 to -0.99 for self-play."""


@dataclass
class ClusterParams:
    num_self_play_nodes_on_cluster: int
    """This is the number of separate nodes on the cluster to use to parallelize the self-play. This should most likely be 16x or more the number of nodes used for training to minimize the wait time for the training nodes to get new data to train with."""


OptimizerType = Literal['adamw', 'sgd']


@dataclass
class TrainingParams:
    num_epochs: int
    """This is the number of epochs to train with one batch of self-play data"""

    batch_size: int
    """This is the size of the batch to train with"""

    optimizer: OptimizerType
    """This is the optimizer to use for the training. Adam is typically better for most cases, but SGD is more stable and faster in some cases."""

    sampling_window: Callable[[int], int]
    """This is a function that returns the sampling window to use for the self-play data. The sampling window is the number of most recent games to sample from to train with. This is used to phase out old data that is no longer useful to train with. The function should take the current iteration as input and return the sampling window to use for that iteration.
    Example:
    def sampling_window(current_iteration: int) -> int:
        if current_iteration < 5:
            return 4
        return 4 + (current_iteration - 5) // 2
    """

    learning_rate: Callable[[int, OptimizerType], float]
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

    validation_percentage: float = 0.01
    """This is the percentage of the training data to use for validation. This is used to validate the model during training to see how well it is doing. The higher the percentage the more data is used for validation but the less data is used for training. Typically 0.01-0.1 for training"""

    max_buffer_samples: int = 4_000_000
    """This is the maximum size of the training buffer to use for the training. This is used to limit the size of the training buffer to prevent memory issues. The higher the value the more stable the training but the slower the training. Typically 1_000_000-10_000_000 for training"""

    max_grad_norm: float = 0.5
    """This is the maximum gradient norm to use for the training. This is used to prevent exploding gradients and to stabilize the training. The lower the value the more stable the training but the slower the training. Typically 0.5-1.0 for training"""

    value_loss_weight: float = 0.5
    """This is the weight to use for the value loss in the training. The value loss is the mean squared error between the predicted value and the actual value. The higher the weight the more important the value loss is in the training. Typically 0.5-1.0 for training"""
    policy_loss_weight: float = 1.0
    """This is the weight to use for the policy loss in the training. The policy loss is the cross-entropy loss between the predicted policy and the actual policy. The higher the weight the more important the policy loss is in the training. Typically 1.0-2.0 for training"""

    num_workers: int = 2
    """This is the number of workers to use for the dataloader to load the self-play data. The higher the number the faster the data is loaded but the more memory is used. Typically 0-4 for training. From experience with this project, 0 seems to work best mostly."""


@dataclass
class EvaluationParams:
    num_searches_per_turn: int
    """This is the number of searches to run the MCTS algorithm in the evaluation. This is used to evaluate the model against itself to see how well it is doing. The higher the number the more accurate the evaluation but the slower the evaluation. Typically 20-50 for evaluation"""

    num_games: int
    """This is the number of games to play for the evaluation. The more games the more accurate the evaluation but the longer the evaluation. Typically 32-256 for evaluation"""

    every_n_iterations: int
    """This is the number of iterations between each evaluation. The higher the number the less often the evaluation is run. Typically 2-10 for evaluation"""

    dataset_path: str | None = None
    """This is the path to the dataset to use for the evaluation. The dataset should contain self-play data to evaluate the model against. The more data the more accurate the evaluation but the longer the evaluation. Typically a few hundred to a few thousand games for evaluation"""


@dataclass
class GatingParams:
    num_games: int = 100
    """This is the number of games to play for the gating. The more games the more accurate the gating but the longer the gating. Typically 100-1000 for gating"""

    num_searches_per_turn: int = 100
    """This is the number of searches to run the MCTS algorithm in the gating. This is used to evaluate the model against itself to see how well it is doing. The higher the number the more accurate the evaluation but the slower the evaluation. Typically 32-800 for gating"""

    ignore_draws: bool = True
    """This is a flag to indicate whether draws should be ignored in the gating. If this is set to True, the gating score will be calculated as wins / (wins + losses) instead of (wins + draws * 0.5) / num_games. This is used to ignore draws in the gating, as they are not relevant for the gating. Typically True for gating."""

    gating_threshold: float = 0.55
    """This is the threshold to use for the gating. If the model's win rate is above this threshold, it is considered to be better than the current model. The higher the threshold the more strict the gating is. Typically 0.50-0.55 for gating."""


@dataclass
class TrainingArgs:
    save_path: str
    """This is the path to save the model, datasamples, training logs, etc. to after each iteration"""

    num_iterations: int
    """This is the number of iterations to run first self-play then train"""

    num_games_per_iteration: int
    """This is the number of self-play games to run per iteration. I.e. the number of games to play and collect data for to train with"""

    network: NetworkParams
    self_play: SelfPlayParams
    training: TrainingParams
    cluster: ClusterParams
    evaluation: Optional[EvaluationParams] = None
    gating: Optional[GatingParams] = None

    on_startup: Optional[Callable[[], None]] = None
    """This is a function that is called on startup to do any necessary setup before training starts. This can be used to ensure that the evaluation dataset exists or to set up the cluster."""
