from typing import List, Tuple

# We treat NodeId as int in Python
NodeId = int
INVALID_NODE: NodeId

class MCTSParams:
    num_parallel_searches: int
    c_param: float
    dirichlet_alpha: float
    dirichlet_epsilon: float
    node_reuse_discount: float
    min_visit_count: int
    num_threads: int

    def __init__(self,
                    num_parallel_searches: int,
                    c_param: float,
                    dirichlet_alpha: float,
                    dirichlet_epsilon: float,
                    node_reuse_discount: float,
                    min_visit_count: int,
                    num_threads: int) -> None:
        ...

class InferenceClientParams:
    device_id: int
    currentModelPath: str
    maxBatchSize: int

    def __init__(self,
                    device_id: int,
                    currentModelPath: str,
                    maxBatchSize: int) -> None:
        ...

class InferenceStatistics:
    cacheHitRate: float
    uniquePositions: int
    cacheSizeMB: int
    nnOutputValueDistribution: List[float]

    def __init__(self) -> None: ...

class MCTSResult:
    result: float
    visits: List[Tuple[str, int]]
    children: List[NodeId]

    # We only expose the attributes as read‐only in C++, but type checkers
    # will understand them as attributes.
    def __init__(self) -> None: ...

class MCTSStatistics:
    averageDepth: float
    averageEntropy: float
    averageKLDivergence: float

    def __init__(self) -> None: ...

class MCTSResults:
    results: List[MCTSResult]
    mctsStats: MCTSStatistics

    def __init__(self) -> None: ...

class MCTS:
    def __init__(self, client_args: InferenceClientParams, mcts_args: MCTSParams) -> None: ...
    def search(self, boards: List[Tuple[str, NodeId, int]]) -> MCTSResults: ...
    def get_inference_statistics(self) -> InferenceStatistics: ...
