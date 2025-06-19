"""
pybind11 bindings for custom MCTS + inference client
"""
from __future__ import annotations
__all__ = ['FunctionTimeInfo', 'InferenceClientParams', 'InferenceStatistics', 'MCTS', 'MCTSNode', 'MCTSParams', 'MCTSResult', 'MCTSResults', 'MCTSStatistics', 'TimeInfo', 'new_root', 'test_inference_speed_cpp']
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
    def __init__(self, device_id: int, currentModelPath: str, maxBatchSize: int) -> None:
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
    def eval_search(self, root: MCTSNode, numberOfSearches: int) -> MCTSResult:
        """
                         Evaluate a search starting from the given root.
                         Returns a `MCTSResult` object containing the average result score and visit counts.
        """
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
def new_root(fen: str) -> MCTSNode:
    """
                Create a new root node for MCTS with the given FEN string.
                Returns a shared pointer to the new MCTSNode.
    """
def test_inference_speed_cpp(numBoards: int = 100, numIterations: int = 10) -> None:
    """
                Test the inference speed of the InferenceClient.
                Runs inference on a specified number of boards for a given number of iterations.
                Prints the average time taken per iteration and per board.
    """
