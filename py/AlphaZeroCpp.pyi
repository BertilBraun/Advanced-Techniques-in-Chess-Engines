"""Native chess and Go search, inference, and game bindings."""

from __future__ import annotations

__all__ = [
    'AnalysisMode',
    'AnalysisResult',
    'BatchedInferenceParameters',
    'BatchedSearchParameters',
    'CandidateAnalysis',
    'ChessSearchChild',
    'ChessSearchRoot',
    'ChessSelfPlaySearch',
    'ChessSelfPlaySearchBatch',
    'ChessSelfPlaySearchParameters',
    'ChessSelfPlaySearchRequest',
    'ChessSelfPlaySearchResult',
    'ChessSelfPlaySearchStatistics',
    'FunctionTimeInfo',
    'GameSearchResult',
    'GameSearchVisit',
    'GoAreaScore',
    'GoBatchedSearch7',
    'GoBatchedSearch9',
    'GoPlayer',
    'GoPosition7',
    'GoPosition9',
    'GoRules',
    'GoSearchRoot7',
    'GoSearchRoot9',
    'GoSymmetry',
    'GoTerminalResult',
    'GoTerminationReason',
    'InferenceClientParams',
    'InferenceDimensions',
    'InferenceDevice',
    'InferenceStatistics',
    'InteractiveEngine',
    'InteractiveGame',
    'InteractiveSearchParams',
    'TimeInfo',
    'WdlPrediction',
    'encode_board_packed_bytes',
    'new_root',
]

class InferenceDimensions:
    def __init__(self, channels: int, rows: int, columns: int, actions: int, outcomes: int) -> None: ...
    @property
    def channels(self) -> int: ...
    @property
    def rows(self) -> int: ...
    @property
    def columns(self) -> int: ...
    @property
    def actions(self) -> int: ...
    @property
    def outcomes(self) -> int: ...

class GameSearchVisit:
    @property
    def action_id(self) -> int: ...
    @property
    def visit_count(self) -> int: ...

class GameSearchResult:
    @property
    def root_value(self) -> float: ...
    @property
    def visits(self) -> list[GameSearchVisit]: ...

class BatchedSearchParameters:
    def __init__(
        self,
        parallel_searches: int,
        exploration_constant: float,
        minimum_root_visits: int,
        dirichlet_alpha: float,
        dirichlet_epsilon: float,
        tree_capacity: int,
    ) -> None: ...

class BatchedInferenceParameters:
    def __init__(
        self,
        workers: int,
        batch_size: int,
        outstanding_batches_per_worker: int,
    ) -> None: ...

class GoPlayer:
    BLACK: GoPlayer
    WHITE: GoPlayer

class GoTerminationReason:
    ONGOING: GoTerminationReason
    TWO_PASSES: GoTerminationReason
    MAXIMUM_MOVES: GoTerminationReason
    @property
    def name(self) -> str: ...

class GoSymmetry:
    IDENTITY: GoSymmetry
    ROTATE_90: GoSymmetry
    ROTATE_180: GoSymmetry
    ROTATE_270: GoSymmetry
    REFLECT: GoSymmetry
    REFLECT_ROTATE_90: GoSymmetry
    REFLECT_ROTATE_180: GoSymmetry
    REFLECT_ROTATE_270: GoSymmetry

class GoRules:
    def __init__(self, komi_half_points: int, maximum_moves: int) -> None: ...
    @property
    def komi_half_points(self) -> int: ...
    @property
    def maximum_moves(self) -> int: ...

class GoAreaScore:
    @property
    def black_half_points(self) -> int: ...
    @property
    def white_half_points(self) -> int: ...
    @property
    def winner(self) -> GoPlayer | None: ...

class GoTerminalResult:
    @property
    def reason(self) -> GoTerminationReason: ...
    @property
    def score(self) -> GoAreaScore: ...
    @property
    def winner(self) -> GoPlayer | None: ...

class GoPosition7:
    def __init__(self, rules: GoRules) -> None: ...
    @property
    def board_size(self) -> int: ...
    @property
    def history_length(self) -> int: ...
    @property
    def player(self) -> GoPlayer: ...
    @property
    def is_terminal(self) -> bool: ...
    def legal_actions(self) -> list[int]: ...
    def child(self, action_id: int) -> GoPosition7: ...
    def black_points(self, history_offset: int) -> list[int]: ...
    def white_points(self, history_offset: int) -> list[int]: ...
    def packed_encoding(self) -> bytes: ...
    def terminal_result(self) -> GoTerminalResult: ...
    def terminal_value(self) -> float | None: ...

class GoPosition9:
    def __init__(self, rules: GoRules) -> None: ...
    @property
    def board_size(self) -> int: ...
    @property
    def history_length(self) -> int: ...
    @property
    def player(self) -> GoPlayer: ...
    @property
    def is_terminal(self) -> bool: ...
    def legal_actions(self) -> list[int]: ...
    def child(self, action_id: int) -> GoPosition9: ...
    def black_points(self, history_offset: int) -> list[int]: ...
    def white_points(self, history_offset: int) -> list[int]: ...
    def packed_encoding(self) -> bytes: ...
    def terminal_result(self) -> GoTerminalResult: ...
    def terminal_value(self) -> float | None: ...

class GoSearchRoot7:
    @property
    def position(self) -> GoPosition7: ...
    @property
    def is_terminal(self) -> bool: ...
    @property
    def visits(self) -> int: ...
    @property
    def live_nodes(self) -> int: ...
    @property
    def children(self) -> list[tuple[int, int]]: ...
    def play(self, action_id: int) -> None: ...

class GoSearchRoot9:
    @property
    def position(self) -> GoPosition9: ...
    @property
    def is_terminal(self) -> bool: ...
    @property
    def visits(self) -> int: ...
    @property
    def live_nodes(self) -> int: ...
    @property
    def children(self) -> list[tuple[int, int]]: ...
    def play(self, action_id: int) -> None: ...

class GoBatchedSearch7:
    def __init__(
        self,
        model_path: str,
        device: InferenceDevice,
        device_id: int,
        inference_parameters: BatchedInferenceParameters,
        search_parameters: BatchedSearchParameters,
        model_generation: int,
    ) -> None: ...
    @staticmethod
    def inference_dimensions() -> InferenceDimensions: ...
    def new_root(self, rules: GoRules) -> GoSearchRoot7: ...
    def search(self, roots: list[GoSearchRoot7], simulations: int) -> list[GameSearchResult]: ...
    def refresh_model(self, model_generation: int, model_path: str) -> None: ...
    @property
    def model_generation(self) -> int: ...

class GoBatchedSearch9:
    def __init__(
        self,
        model_path: str,
        device: InferenceDevice,
        device_id: int,
        inference_parameters: BatchedInferenceParameters,
        search_parameters: BatchedSearchParameters,
        model_generation: int,
    ) -> None: ...
    @staticmethod
    def inference_dimensions() -> InferenceDimensions: ...
    def new_root(self, rules: GoRules) -> GoSearchRoot9: ...
    def search(self, roots: list[GoSearchRoot9], simulations: int) -> list[GameSearchResult]: ...
    def refresh_model(self, model_generation: int, model_path: str) -> None: ...
    @property
    def model_generation(self) -> int: ...

class AnalysisMode:
    POLICY: AnalysisMode
    MCTS: AnalysisMode

class WdlPrediction:
    @property
    def win(self) -> float: ...
    @property
    def draw(self) -> float: ...
    @property
    def loss(self) -> float: ...
    @property
    def value(self) -> float: ...

class CandidateAnalysis:
    @property
    def move_uci(self) -> str: ...
    @property
    def policy_prior(self) -> float: ...
    @property
    def visits(self) -> int: ...
    @property
    def visit_share(self) -> float: ...
    @property
    def mean_value(self) -> float | None: ...

class AnalysisResult:
    @property
    def chosen_move_uci(self) -> str: ...
    @property
    def value(self) -> float: ...
    @property
    def outcome(self) -> WdlPrediction | None: ...
    @property
    def candidates(self) -> list[CandidateAnalysis]: ...
    @property
    def searches(self) -> int: ...
    @property
    def maximum_depth(self) -> int: ...
    @property
    def elapsed_milliseconds(self) -> int: ...
    @property
    def principal_variation(self) -> list[str]: ...

class InferenceDevice:
    AUTO: InferenceDevice
    CPU: InferenceDevice
    CUDA: InferenceDevice

class InteractiveSearchParams:
    def __init__(
        self,
        exploration_constant: float,
        inference_workers: int,
        inference_batch_size: int,
        outstanding_batches_per_worker: int = 2,
    ) -> None: ...
    exploration_constant: float
    inference_workers: int
    inference_batch_size: int
    outstanding_batches_per_worker: int

class InteractiveEngine:
    def __init__(
        self,
        client_parameters: InferenceClientParams,
        search_parameters: InteractiveSearchParams,
    ) -> None: ...
    def new_game(self, starting_fen: str, moves_uci: tuple[str, ...]) -> InteractiveGame: ...
    def get_inference_statistics(self) -> InferenceStatistics: ...

class InteractiveGame:
    def apply_move(self, move_uci: str) -> None: ...
    def analyze(
        self,
        mode: AnalysisMode,
        time_limit_seconds: int | None = None,
        search_limit: int | None = None,
    ) -> AnalysisResult: ...
    @property
    def fen(self) -> str: ...
    @property
    def starting_fen(self) -> str: ...
    @property
    def moves_uci(self) -> list[str]: ...
    @property
    def root_visits(self) -> int: ...

def encode_board_packed_bytes(fen: str) -> bytes:
    """Encode a FEN into the canonical packed plane-major byte layout."""

class FunctionTimeInfo:
    @property
    def invocations(self) -> int: ...
    @property
    def name(self) -> str: ...
    @property
    def percent(self) -> float: ...
    @property
    def total(self) -> float: ...

class InferenceClientParams:
    currentModelPath: str
    device_id: int
    maxBatchSize: int
    device: InferenceDevice
    def __init__(
        self,
        device_id: int,
        currentModelPath: str,
        maxBatchSize: int,
        microsecondsTimeoutInferenceThread: int,
        device: InferenceDevice = InferenceDevice.AUTO,
    ) -> None: ...
    @property
    def microsecondsTimeoutInferenceThread(self) -> int:
        """
        Timeout for the inference thread in microseconds.
        Default is 500 microseconds.
        """
    @microsecondsTimeoutInferenceThread.setter
    def microsecondsTimeoutInferenceThread(self, arg0: int) -> None: ...

class InferenceStatistics:
    def __init__(self) -> None: ...
    @property
    def averageNumberOfPositionsInInferenceCall(self) -> float: ...
    @property
    def evaluations(self) -> int: ...
    @property
    def modelBatchSizeHistogram(self) -> list[int]: ...
    @property
    def modelInferenceCalls(self) -> int: ...
    @property
    def modelInferencePositions(self) -> int: ...

class ChessSelfPlaySearch:
    def __init__(
        self,
        runtime_parameters: InferenceClientParams,
        search_parameters: ChessSelfPlaySearchParameters,
        inference_parameters: BatchedInferenceParameters,
        initial_model_version: int = 0,
    ) -> None: ...
    @property
    def arena_capacity(self) -> int: ...
    @property
    def model_version(self) -> int: ...
    def inference_statistics(self) -> tuple[InferenceStatistics, TimeInfo]: ...
    def refresh_model(self, model_version: int, model_path: str) -> None: ...
    def update_search_schedule(self, search_parameters: ChessSelfPlaySearchParameters) -> bool: ...
    def inference_with_history(
        self, histories: list[tuple[str, list[str]]]
    ) -> list[tuple[list[tuple[int, float]], float]]: ...
    def search(
        self,
        requests: list[ChessSelfPlaySearchRequest],
        collect_statistics: bool = False,
    ) -> ChessSelfPlaySearchBatch:
        """
        Run batched chess self-play search requests.
        Returns a ChessSelfPlaySearchBatch whose results contain:
            - root_value: float
            - visits: List of (encoded_move: int, visit_count: int)
            - root: the retained ChessSearchRoot
        """
    def new_root(self, fen: str) -> ChessSearchRoot: ...
    def new_root_with_history(
        self,
        starting_fen: str,
        moves_uci: tuple[str, ...],
    ) -> ChessSearchRoot: ...

class ChessSelfPlaySearchRequest:
    def __init__(self, root: ChessSearchRoot, full_search: bool) -> None: ...
    @property
    def root(self) -> ChessSearchRoot: ...
    @property
    def full_search(self) -> bool: ...

class ChessSearchChild:
    @property
    def encoded_move(self) -> int: ...
    @property
    def is_materialized(self) -> bool: ...
    @property
    def move(self) -> str: ...
    @property
    def policy(self) -> float: ...
    @property
    def result_sum(self) -> float: ...
    @property
    def virtual_loss(self) -> float: ...
    @property
    def visits(self) -> int: ...

class ChessSearchRoot:
    def __repr__(self) -> str: ...
    def discount(self, percentage_of_node_visits_to_keep: float) -> None:
        """
        Discount the node's score and visits by a percentage.
        Descendant materializations are explicitly pruned when required by the fixed arena.
        """
    def make_new_root(self, child_index: int) -> ChessSearchRoot:
        """
        Prune the old tree and return a new root node.
        `child_index` is the index of the child to make the new root.
        """
    @property
    def arena_capacity(self) -> int: ...
    @property
    def children(self) -> list[ChessSearchChild]: ...
    @property
    def fen(self) -> str: ...
    @property
    def is_expanded(self) -> bool: ...
    @property
    def is_terminal(self) -> bool: ...
    @property
    def live_nodes(self) -> int: ...
    @property
    def max_depth(self) -> int: ...
    @property
    def move(self) -> str: ...
    @property
    def repetition_count(self) -> int: ...
    @property
    def result_sum(self) -> float: ...
    @property
    def total_child_records(self) -> int: ...
    @property
    def virtual_loss(self) -> float: ...
    @property
    def visits(self) -> int: ...

class ChessSelfPlaySearchParameters:
    exploration_constant: float
    dirichlet_alpha: float
    dirichlet_epsilon: float
    fast_searches: int
    full_searches: int
    minimum_root_visits: int
    parallel_searches: int
    def __init__(
        self,
        parallel_searches: int,
        full_searches: int,
        fast_searches: int,
        exploration_constant: float,
        dirichlet_alpha: float,
        dirichlet_epsilon: float,
        minimum_root_visits: int,
    ) -> None: ...

class ChessSelfPlaySearchResult:
    @property
    def root_value(self) -> float: ...
    @property
    def root(self) -> ChessSearchRoot: ...
    @property
    def visits(self) -> list[tuple[int, int]]: ...

class ChessSelfPlaySearchBatch:
    @property
    def statistics(self) -> ChessSelfPlaySearchStatistics: ...
    @property
    def results(self) -> list[ChessSelfPlaySearchResult]: ...
    @property
    def simulations_completed(self) -> int: ...

class ChessSelfPlaySearchStatistics:
    @property
    def average_depth(self) -> float: ...
    @property
    def average_entropy(self) -> float: ...
    @property
    def average_kl_divergence(self) -> float: ...
    @property
    def average_policy_search_kl_divergence(self) -> float: ...
    @property
    def top_action_disagreement(self) -> float: ...
    @property
    def selected_action_prior_rank(self) -> float: ...

class TimeInfo:
    @property
    def functionTimes(self) -> list[FunctionTimeInfo]: ...
    @property
    def percentRecorded(self) -> float: ...
    @property
    def totalTime(self) -> float: ...

def new_root(fen: str, arena_capacity: int) -> ChessSearchRoot:
    """
    Create a self-play MCTS root with an explicit fixed arena capacity.
    """

def new_root_with_history(
    starting_fen: str,
    moves_uci: tuple[str, ...],
    arena_capacity: int,
) -> ChessSearchRoot:
    """Create a fixed-capacity MCTS root by replaying bounded UCI history."""
