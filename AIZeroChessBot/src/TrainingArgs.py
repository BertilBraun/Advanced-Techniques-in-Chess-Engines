from dataclasses import dataclass


@dataclass
class TrainingArgs:
    num_iterations: int
    """This is the number of iterations to run first self-play then train"""

    num_self_play_iterations: int
    """This is the number of self-play iterations to run per iteration. I.e. the number of games to play and collect data for to train with"""

    num_parallel_games: int
    """This is the number of games to run in parallel for self-play."""

    num_searches: int
    """This is the maximum number of searches to run the MCTS algorithm in self-play. 
    I.e. the number of times that a node is expanded and evaluated in the MCTS algorithm to get the next move to play.
    I.e. we continue to play the game util one of the players wins or the game is a draw. At each move we run the MCTS algorithm to get the next move to play for the current player. Here we run the MCTS algorithm num_searches times to get the next move to play.
    """

    num_epochs: int
    """This is the number of epochs to train with one batch of self-play data"""

    batch_size: int
    """This is the size of the batch to train with"""

    temperature: float
    """This is the sampling temperature to use for the MCTS algorithm in self-play"""

    dirichlet_epsilon: float
    """This is the epsilon value to use for the dirichlet noise to add to the root node in self-play to encourage exploration"""

    dirichlet_alpha: float
    """This is the alpha value to use for the dirichlet noise to add to the root node in self-play to encourage exploration"""

    c_param: float
    """This is the c parameter to use for the UCB1 formula in the MCTS algorithm in self-play"""

    save_path: str = ''
    """This is the path to save the model to after each iteration"""
