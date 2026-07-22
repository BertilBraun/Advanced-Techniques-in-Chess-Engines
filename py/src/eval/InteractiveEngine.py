from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from AlphaZeroCpp import (
    AnalysisMode as NativeAnalysisMode,
    InferenceClientParams,
    InferenceDevice,
    InteractiveEngine as NativeInteractiveEngine,
    InteractiveGame as NativeInteractiveGame,
    InteractiveSearchParams,
)


class AnalysisMode(str, Enum):
    POLICY = "policy"
    MCTS = "mcts"


class InferenceTarget(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


@dataclass(frozen=True)
class OutcomePrediction:
    """Network W/D/L probabilities from the side-to-move perspective."""

    win: float
    draw: float
    loss: float


@dataclass(frozen=True)
class CandidateAnalysis:
    """A candidate value is from the root side-to-move perspective."""

    move_uci: str
    policy_prior: float
    visits: int
    visit_share: float
    mean_value: float | None


@dataclass(frozen=True)
class AnalysisResult:
    """Interactive analysis ordered by final preference.

    MCTS chooses maximum visits, breaking ties by higher prior and then ascending UCI.
    ``value`` and candidate means are from the root side-to-move perspective.
    ``outcome`` is the root network W/D/L evaluation, not a conversion of search value.
    """

    chosen_move_uci: str
    value: float
    outcome: OutcomePrediction | None
    candidates: tuple[CandidateAnalysis, ...]
    searches: int
    maximum_depth: int
    elapsed_milliseconds: int
    principal_variation: tuple[str, ...]


@dataclass(frozen=True)
class InferenceMetrics:
    evaluations: int
    cache_hits: int
    cache_hit_rate_percent: float
    model_calls: int
    model_positions: int
    average_model_batch_size: float
    tree_selection_nanoseconds: int
    board_encoding_nanoseconds: int
    result_processing_nanoseconds: int
    tree_backup_nanoseconds: int
    tree_owner_wait_nanoseconds: int
    direct_inference_nanoseconds: int
    direct_worker_utilization: float


class InteractiveEngine:
    def __init__(
        self,
        model_path: str,
        device_id: int,
        parallel_searches: int,
        c_param: float,
        inference_workers: int = 3,
        outstanding_batches_per_worker: int = 1,
        maximum_batch_size: int | None = None,
        batch_collection_timeout_microseconds: int = 500,
        cache_capacity: int = 250_000,
        inference_target: InferenceTarget = InferenceTarget.AUTO,
    ) -> None:
        target_mapping = {
            InferenceTarget.AUTO: InferenceDevice.AUTO,
            InferenceTarget.CPU: InferenceDevice.CPU,
            InferenceTarget.CUDA: InferenceDevice.CUDA,
        }
        resolved_batch_size = (
            parallel_searches if maximum_batch_size is None else maximum_batch_size
        )
        client_parameters = InferenceClientParams(
            device_id=device_id,
            currentModelPath=model_path,
            maxBatchSize=resolved_batch_size,
            microsecondsTimeoutInferenceThread=batch_collection_timeout_microseconds,
            cacheCapacity=cache_capacity,
            device=target_mapping[inference_target],
        )
        self._native = NativeInteractiveEngine(
            client_parameters,
            InteractiveSearchParams(
                exploration_constant=c_param,
                inference_workers=inference_workers,
                inference_batch_size=resolved_batch_size,
                outstanding_batches_per_worker=outstanding_batches_per_worker,
            ),
        )

    def new_game(
        self, starting_fen: str, moves_uci: tuple[str, ...]
    ) -> InteractiveGame:
        return InteractiveGame(self._native.new_game(starting_fen, moves_uci))

    def inference_metrics(self) -> InferenceMetrics:
        statistics = self._native.get_inference_statistics()
        return InferenceMetrics(
            evaluations=statistics.evaluations,
            cache_hits=statistics.cacheHits,
            cache_hit_rate_percent=statistics.cacheHitRate,
            model_calls=statistics.modelInferenceCalls,
            model_positions=statistics.modelInferencePositions,
            average_model_batch_size=statistics.averageNumberOfPositionsInInferenceCall,
            tree_selection_nanoseconds=statistics.treeSelectionNanoseconds,
            board_encoding_nanoseconds=statistics.boardEncodingNanoseconds,
            result_processing_nanoseconds=statistics.resultProcessingNanoseconds,
            tree_backup_nanoseconds=statistics.treeBackupNanoseconds,
            tree_owner_wait_nanoseconds=statistics.treeOwnerWaitNanoseconds,
            direct_inference_nanoseconds=statistics.directInferenceNanoseconds,
            direct_worker_utilization=statistics.directWorkerUtilization,
        )


class InteractiveGame:
    def __init__(self, native_game: NativeInteractiveGame) -> None:
        self._native = native_game

    @property
    def fen(self) -> str:
        return self._native.fen

    @property
    def starting_fen(self) -> str:
        return self._native.starting_fen

    @property
    def moves_uci(self) -> tuple[str, ...]:
        return tuple(self._native.moves_uci)

    @property
    def root_visits(self) -> int:
        return self._native.root_visits

    def apply_move(self, move_uci: str) -> None:
        self._native.apply_move(move_uci)

    def analyze(
        self,
        mode: AnalysisMode,
        time_limit_seconds: int | None = None,
        search_limit: int | None = None,
    ) -> AnalysisResult:
        native_mode = (
            NativeAnalysisMode.POLICY
            if mode is AnalysisMode.POLICY
            else NativeAnalysisMode.MCTS
        )
        result = self._native.analyze(native_mode, time_limit_seconds, search_limit)
        outcome = (
            None
            if result.outcome is None
            else OutcomePrediction(
                win=result.outcome.win,
                draw=result.outcome.draw,
                loss=result.outcome.loss,
            )
        )
        candidates = tuple(
            CandidateAnalysis(
                move_uci=candidate.move_uci,
                policy_prior=candidate.policy_prior,
                visits=candidate.visits,
                visit_share=candidate.visit_share,
                mean_value=candidate.mean_value,
            )
            for candidate in result.candidates
        )
        return AnalysisResult(
            chosen_move_uci=result.chosen_move_uci,
            value=result.value,
            outcome=outcome,
            candidates=candidates,
            searches=result.searches,
            maximum_depth=result.maximum_depth,
            elapsed_milliseconds=result.elapsed_milliseconds,
            principal_variation=tuple(result.principal_variation),
        )
