from __future__ import annotations

from dataclasses import dataclass

import chess
from AlphaZeroCpp import (
    AnalysisMode,
    AnalysisParameters,
    BatchedInferenceParameters,
    ChessAnalysis as BoundChessAnalysis,
    ChessAnalysisSession as BoundChessAnalysisSession,
    InferenceDevice,
    InferenceConfiguration,
)
from src.games.chess.interactive.analysis import (
    AnalysisRequest,
    AnalysisResult,
    CandidateAnalysis,
    CountedMctsAnalysis,
    OutcomePrediction,
    PolicyAnalysis,
    TimedMctsAnalysis,
)
from src.games.chess.interactive.configuration import InferenceTarget, InteractiveEngineConfiguration
from src.self_play.native_configuration import native_sdpa_backend


@dataclass(frozen=True)
class InferenceMetrics:
    evaluations: int
    model_calls: int
    model_positions: int
    average_model_batch_size: float
    tree_selection_nanoseconds: int
    board_encoding_nanoseconds: int
    result_processing_nanoseconds: int
    tree_backup_nanoseconds: int
    tree_owner_wait_nanoseconds: int
    inference_nanoseconds: int
    worker_utilization: float


class InteractiveEngine:
    def __init__(self, configuration: InteractiveEngineConfiguration) -> None:
        target_mapping = {
            InferenceTarget.AUTO: InferenceDevice.AUTO,
            InferenceTarget.CPU: InferenceDevice.CPU,
            InferenceTarget.CUDA: InferenceDevice.CUDA,
        }
        batch_size = configuration.resolved_batch_size
        runtime_parameters = InferenceConfiguration(
            device_id=configuration.device_id,
            model_path=configuration.model_path,
            device=target_mapping[configuration.inference_target],
            sdpa_backend=native_sdpa_backend(configuration.sdpa_backend),
        )
        self._bound_analysis = BoundChessAnalysis(
            runtime_parameters,
            AnalysisParameters(
                parallel_searches=configuration.parallel_searches,
                exploration_constant=configuration.exploration_constant,
                inference=BatchedInferenceParameters(
                    workers=configuration.inference_workers,
                    batch_size=batch_size,
                    outstanding_batches_per_worker=configuration.outstanding_batches_per_worker,
                ),
            ),
        )

    def new_game(self, starting_fen: str, moves_uci: tuple[str, ...]) -> InteractiveGame:
        return InteractiveGame(self._bound_analysis.new_session(starting_fen, moves_uci))

    def inference_metrics(self) -> InferenceMetrics:
        statistics = self._bound_analysis.inference_statistics()
        return InferenceMetrics(
            evaluations=statistics.evaluations,
            model_calls=statistics.modelInferenceCalls,
            model_positions=statistics.modelInferencePositions,
            average_model_batch_size=statistics.averageNumberOfPositionsInInferenceCall,
            tree_selection_nanoseconds=statistics.treeSelectionNanoseconds,
            board_encoding_nanoseconds=statistics.boardEncodingNanoseconds,
            result_processing_nanoseconds=statistics.resultProcessingNanoseconds,
            tree_backup_nanoseconds=statistics.treeBackupNanoseconds,
            tree_owner_wait_nanoseconds=statistics.treeOwnerWaitNanoseconds,
            inference_nanoseconds=statistics.inferenceNanoseconds,
            worker_utilization=statistics.workerUtilization,
        )


class InteractiveGame:
    def __init__(self, bound_game: BoundChessAnalysisSession) -> None:
        self._bound_game = bound_game

    @property
    def fen(self) -> str:
        return self._bound_game.fen

    @property
    def starting_fen(self) -> str:
        return self._bound_game.starting_fen

    @property
    def moves_uci(self) -> tuple[str, ...]:
        return tuple(self._bound_game.moves_uci)

    @property
    def root_visits(self) -> int:
        return self._bound_game.root_visits

    @property
    def is_game_over(self) -> bool:
        return chess.Board(self.fen).is_game_over(claim_draw=True)

    @property
    def result(self) -> str | None:
        board = chess.Board(self.fen)
        return board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else None

    def apply_move(self, move_uci: str) -> None:
        self._bound_game.apply_move(move_uci)

    def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        match request:
            case PolicyAnalysis():
                result = self._bound_game.analyze(AnalysisMode.POLICY, None, None)
            case TimedMctsAnalysis(seconds=seconds):
                result = self._bound_game.analyze(AnalysisMode.MCTS, seconds, None)
            case CountedMctsAnalysis(searches=searches):
                result = self._bound_game.analyze(AnalysisMode.MCTS, None, searches)
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
