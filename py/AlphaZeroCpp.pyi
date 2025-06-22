"""
pybind11 bindings for custom MCTS + inference client
"""
from __future__ import annotations
__all__ = ['EvalMCTS', 'EvalMCTSNode', 'EvalMCTSParams', 'EvalMCTSResult', 'FunctionTimeInfo', 'InferenceClientParams', 'InferenceStatistics', 'MCTS', 'MCTSNode', 'MCTSParams', 'MCTSResult', 'MCTSResults', 'MCTSStatistics', 'TimeInfo', 'new_eval_root', 'new_root', 'test_eval_mcts_speed_cpp', 'test_inference_speed_cpp', 'test_mcts_speed_cpp']
class EvalMCTS:
    def __init__(self, client_args: InferenceClientParams, mcts_args: EvalMCTSParams) -> None:
        ...
    def eval_search(self, root: EvalMCTSNode, searches: int) -> EvalMCTSResult:
        """
                         Run evaluation MCTS search on a given root node.
                         Returns an `EvalMCTSResult` object containing the result, visits, and root node.
        """
class EvalMCTSNode:
    def best_child(self, c_param: float) -> EvalMCTSNode:
        """
                    Get the best child node based on UCB score.
                    `c_param` is the exploration constant.
        """
    def make_new_root(self, child_index: int) -> EvalMCTSNode:
        """
                    Prune the old tree and return a new root node.
                    `child_index` is the index of the child to make the new root.
        """
    @property
    def children(self) -> list[EvalMCTSNode]:
        ...
    @property
    def encoded_move(self) -> int:
        ...
    @property
    def fen(self) -> str:
        ...
    @property
    def max_depth(self) -> int:
        ...
    @property
    def move(self) -> str:
        ...
    @property
    def policy(self) -> float:
        ...
    @property
    def result_sum(self) -> float:
        ...
    @property
    def visits(self) -> int:
        ...
class EvalMCTSParams:
    c_param: float
    num_threads: int
    def __init__(self, c_param: float, num_threads: int) -> None:
        ...
class EvalMCTSResult:
    @property
    def result(self) -> float:
        ...
    @property
    def root(self) -> EvalMCTSNode:
        ...
    @property
    def visits(self) -> list[tuple[int, int]]:
        ...
class FunctionTimeInfo:
    @property
    def invocations(self) -> int:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def percent(self) -> float:
        ...
    @property
    def total(self) -> float:
        ...
class InferenceClientParams:
    currentModelPath: str
    device_id: int
    maxBatchSize: int
    def __init__(self, device_id: int, currentModelPath: str, maxBatchSize: int, microsecondsTimeoutInferenceThread: int = 500) -> None:
        ...
    @property
    def microsecondsTimeoutInferenceThread(self) -> int:
        """
                        Timeout for the inference thread in microseconds.
                        Default is 500 microseconds.
        """
    @microsecondsTimeoutInferenceThread.setter
    def microsecondsTimeoutInferenceThread(self, arg0: int) -> None:
        ...
class InferenceStatistics:
    def __init__(self) -> None:
        ...
    @property
    def averageNumberOfPositionsInInferenceCall(self) -> float:
        ...
    @property
    def cacheHitRate(self) -> float:
        ...
    @property
    def cacheSizeMB(self) -> int:
        ...
    @property
    def nnOutputValueDistribution(self) -> list[float]:
        ...
    @property
    def uniquePositions(self) -> int:
        ...
class MCTS:
    def __init__(self, client_args: InferenceClientParams, mcts_args: MCTSParams) -> None:
        ...
    def get_inference_statistics(self) -> tuple[InferenceStatistics, TimeInfo]:
        ...
    def inference(self, fen: str) -> tuple[list[tuple[int, float]], float]:
        """
                         Run inference on a given FEN string.
                         Returns a tuple of (encoded_moves: List[Tuple[int, float]], value: float).
                         The encoded moves are pairs of (encoded_move: int, score: float).
        """
    def search(self, boards: list[tuple[MCTSNode, bool]]) -> MCTSResults:
        """
                         Run MCTS search on a list of boards.
                         `boards` should be a list of tuples: (fen_str: str, prev_node: NodeId, full_search: bool).
                         Returns an `MCTSResults` object, whose `.results` is a list of `MCTSResult`:
                             - result: float
                             - visits: List of (encoded_move: int, visit_count: int)
                             - children: List of NodeId (uint32)
                         and `.mctsStats` contains avg depth/entropy/KL.
        """
class MCTSNode:
    def __repr__(self) -> str:
        ...
    def best_child(self, cParam: float) -> MCTSNode:
        ...
    def discount(self, percentage_of_node_visits_to_keep: float) -> None:
        """
                    Discount the node's score and visits by a percentage.
                    This is useful for pruning old nodes in the search tree.
        """
    def make_new_root(self, child_index: int) -> MCTSNode:
        """
                    Prune the old tree and return a new root node.
                    `child_index` is the index of the child to make the new root.
        """
    def ucb(self, cParam: float) -> float:
        """
        Calculate UCB score given exploration constant cParam.
        """
    @property
    def children(self) -> list[MCTSNode]:
        ...
    @property
    def encoded_move(self) -> int:
        ...
    @property
    def fen(self) -> str:
        ...
    @property
    def is_expanded(self) -> bool:
        ...
    @property
    def is_terminal(self) -> bool:
        ...
    @property
    def max_depth(self) -> int:
        ...
    @property
    def move(self) -> str:
        ...
    @property
    def parent(self) -> MCTSNode:
        ...
    @property
    def policy(self) -> float:
        ...
    @property
    def result_sum(self) -> float:
        ...
    @property
    def virtual_loss(self) -> float:
        ...
    @property
    def visits(self) -> int:
        ...
class MCTSParams:
    c_param: float
    dirichlet_alpha: float
    dirichlet_epsilon: float
    min_visit_count: int
    num_fast_searches: int
    num_full_searches: int
    num_parallel_searches: int
    num_threads: int
    def __init__(self, num_parallel_searches: int, num_full_searches: int, num_fast_searches: int, c_param: float, dirichlet_alpha: float, dirichlet_epsilon: float, min_visit_count: int, num_threads: int) -> None:
        ...
class MCTSResult:
    @property
    def result(self) -> float:
        ...
    @property
    def root(self) -> MCTSNode:
        ...
    @property
    def visits(self) -> list[tuple[int, int]]:
        ...
class MCTSResults:
    @property
    def mctsStats(self) -> MCTSStatistics:
        ...
    @property
    def results(self) -> list[MCTSResult]:
        ...
class MCTSStatistics:
    @property
    def averageDepth(self) -> float:
        ...
    @property
    def averageEntropy(self) -> float:
        ...
    @property
    def averageKLDivergence(self) -> float:
        ...
class TimeInfo:
    @property
    def functionTimes(self) -> list[FunctionTimeInfo]:
        ...
    @property
    def percentRecorded(self) -> float:
        ...
    @property
    def totalTime(self) -> float:
        ...
def new_eval_root(fen: str) -> EvalMCTSNode:
    """
                Create a new root node for evaluation MCTS with the given FEN string.
                Returns a shared pointer to the new EvalMCTSNode.
    """
def new_root(fen: str) -> MCTSNode:
    """
                Create a new root node for MCTS with the given FEN string.
                Returns a shared pointer to the new MCTSNode.
    """
def test_eval_mcts_speed_cpp(numBoards: int = 100, numIterations: int = 10, numSearchesPerTurn: int = 100, numParallelSearches: int = 1, numThreads: int = 1) -> None:
    """
                Test the Eval MCTS search speed.
                Runs Eval MCTS search on a specified number of boards for a given number of iterations.
                Prints the average time taken per iteration and per board.
    """
def test_inference_speed_cpp(numBoards: int = 100, numIterations: int = 10) -> None:
    """
                Test the inference speed of the InferenceClient.
                Runs inference on a specified number of boards for a given number of iterations.
                Prints the average time taken per iteration and per board.
    """
def test_mcts_speed_cpp(numBoards: int = 100, numIterations: int = 10, numSearchesPerTurn: int = 100, numParallelSearches: int = 1, numThreads: int = 1) -> None:
    """
                Test the MCTS search speed.
                Runs MCTS search on a specified number of boards for a given number of iterations.
                Prints the average time taken per iteration and per board.
    """
