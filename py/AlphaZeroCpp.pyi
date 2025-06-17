from __future__ import annotations
from typing import List, Tuple, Optional

# We treat NodeId as int in Python
NodeId = int
INVALID_NODE: NodeId

def init() -> None: ...

class MCTSParams:
    num_parallel_searches: int
    c_param: float
    dirichlet_alpha: float
    dirichlet_epsilon: float
    node_reuse_discount: float
    min_visit_count: int
    num_threads: int
    num_full_searches: int
    num_fast_searches: int

    def __init__(
        self,
        num_parallel_searches: int,
        c_param: float,
        dirichlet_alpha: float,
        dirichlet_epsilon: float,
        node_reuse_discount: float,
        min_visit_count: int,
        num_threads: int,
        num_full_searches: int,
        num_fast_searches: int,
    ) -> None: ...

class InferenceClientParams:
    device_id: int
    currentModelPath: str
    maxBatchSize: int

    def __init__(self, device_id: int, currentModelPath: str, maxBatchSize: int) -> None: ...

class InferenceStatistics:
    cacheHitRate: float
    uniquePositions: int
    cacheSizeMB: int
    nnOutputValueDistribution: List[float]
    averageNumberOfPositionsInInferenceCall: float

    def __init__(self) -> None: ...

class MCTSResult:
    result: float
    visits: List[Tuple[int, int]]  # list of (Encoded Move Id, Visit Count)
    children: List[NodeId]

    # We only expose the attributes as read‐only in C++, but type checkers
    # will understand them as attributes.
    def __init__(self) -> None: ...

class MCTSStatistics:
    averageDepth: float
    averageEntropy: float
    averageKLDivergence: float
    nodePoolCapacity: int
    liveNodeCount: int

    def __init__(self) -> None: ...

class MCTSResults:
    results: List[MCTSResult]
    mctsStats: MCTSStatistics

    def __init__(self) -> None: ...

class MCTS:
    def __init__(self, client_args: InferenceClientParams, mcts_args: MCTSParams) -> None: ...
    def search(self, boards: List[Tuple[str, NodeId, bool]]) -> MCTSResults: ...
    def get_inference_statistics(self) -> InferenceStatistics: ...
    def get_node(self, node_id: NodeId) -> MCTSNode: ...
    def clear_node_pool(self) -> None: ...
    def inference(self, board: str) -> Tuple[List[Tuple[int, float]], float]:
        """Returns a tuple of (encoded moves with probabilities, value) for the given board position."""
        ...
    def free_tree(self, node_id: NodeId) -> None:
        """Frees the tree starting from the given node ID."""
        ...
    def eval_search(self, fen: str, node_id: NodeId, num_searches: int) -> MCTSResult:
        """Performs a search and returns the result for the given FEN and node ID."""
        ...

class MCTSNode:
    """Read‑only view of an MCTS node."""

    # Tree structure ----------------------------------------------------------
    id: NodeId  # Unique identifier for this node
    @property
    def parent(self) -> Optional[MCTSNode]: ...
    @property
    def children(self) -> List[MCTSNode]: ...

    # Basic fields ------------------------------------------------------------
    fen: str  # Board position (FEN)
    move: str  # UCI move that leads to this node ("null" for root)
    visits: int  # Number of simulations that visited this node
    virtual_loss: float  # Virtual loss applied during parallel search
    result: float  # Accumulated back‑propagated value (e.g. win prob)
    policy: float  # Prior probability from neural network
    is_terminal: bool  # True if this node is a terminal node (game over)
    is_fully_expanded: bool  # True if this node has children (fully expanded)

    def ucb(self, c_param: float) -> float: ...

    # Convenience -------------------------------------------------------------
    def __repr__(self) -> str: ...
