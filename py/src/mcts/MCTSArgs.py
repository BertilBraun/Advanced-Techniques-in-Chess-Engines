from dataclasses import dataclass


@dataclass
class MCTSArgs:
    """This class contains the arguments for the MCTS algorithm."""

    num_searches_per_turn: int
    """This is the maximum number of searches to run the MCTS algorithm in self-play. 
    I.e. the number of times that a node is expanded and evaluated in the MCTS algorithm to get the next move to play.
    I.e. we continue to play the game util one of the players wins or the game is a draw. At each move we run the MCTS algorithm to get the next move to play for the current player. Here we run the MCTS algorithm num_searches times to get the next move to play.
    """

    num_parallel_searches: int
    """This is the number of parallel strands which should parallelize the MCTS algorithm. Higher values enable more parallelism and thus faster search. However, it also increases the exploration of the search tree. Values between 1 and 16 seem sensible."""

    dirichlet_epsilon: float
    """This is the epsilon value to use for the dirichlet noise to add to the root node in self-play to encourage exploration. I.e. the percentage of the resulting policy, that should be the dirichlet noise. The rest is the policy from the neural network. lerp(policy, dirichlet_noise, factor=dirichlet_epsilon)"""

    dirichlet_alpha: float
    """This is the alpha value function to use for the dirichlet noise to add to the root node in self-play to encourage exploration. The iteration is passed in and the alpha value to use should be returned. Apparently the value should be around 10/number_of_actions. So for Connect4 with 7 columns this value should be around 1.5, and for chess with 400 possible moves this value should be around 0.025"""

    c_param: float
    """This is the c parameter to use for the UCB1 formula in the MCTS algorithm in self-play. It is used to balance exploration and exploitation in the MCTS algorithm. Values between 1 and 6 seem sensible. The higher the value the more exploration is favored over exploitation."""

    min_visit_count: int
    """The minimum number of visits that each root child should recieve. Typically this value is < 5 or in proportion to the num_searches_per_turn. This is used to ensure that the MCTS algorithm has explored the search tree enough to make a good decision. If the number of visits is too low, the MCTS algorithm might not explore enough to learn the best moves to play."""
