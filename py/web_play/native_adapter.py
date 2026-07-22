from __future__ import annotations

from dataclasses import dataclass

import chess

from src.eval.InteractiveEngine import (
    AnalysisMode as OptimizedAnalysisMode,
    AnalysisResult as OptimizedAnalysisResult,
    InferenceTarget,
    InteractiveEngine as OptimizedInteractiveEngine,
    InteractiveGame as OptimizedInteractiveGame,
)
from src.settings import PLAY_C_PARAM
from web_play.contracts import (
    AnalysisLimit,
    AnalysisMode,
    AnalysisResult,
    CandidateMove,
    CountedAnalysis,
    OutcomePrediction,
    SearchMetrics,
    TimedAnalysis,
)


@dataclass(frozen=True)
class NativeEngineConfiguration:
    model_path: str
    device_id: int
    parallel_searches: int
    exploration_constant: float
    inference_workers: int
    outstanding_batches_per_worker: int
    maximum_batch_size: int
    batch_collection_timeout_microseconds: int
    cache_capacity: int
    inference_target: InferenceTarget

    @classmethod
    def for_model(cls, model_path: str, device_id: int = 0) -> NativeEngineConfiguration:
        return cls(
            model_path=model_path,
            device_id=device_id,
            parallel_searches=64,
            exploration_constant=PLAY_C_PARAM,
            inference_workers=2,
            outstanding_batches_per_worker=2,
            maximum_batch_size=64,
            batch_collection_timeout_microseconds=500,
            cache_capacity=250_000,
            inference_target=InferenceTarget.AUTO,
        )


class NativeInteractiveEngine:
    """Web adapter for the optimized long-lived interactive engine."""

    def __init__(self, configuration: NativeEngineConfiguration) -> None:
        self._engine = OptimizedInteractiveEngine(
            model_path=configuration.model_path,
            device_id=configuration.device_id,
            parallel_searches=configuration.parallel_searches,
            c_param=configuration.exploration_constant,
            inference_workers=configuration.inference_workers,
            outstanding_batches_per_worker=(configuration.outstanding_batches_per_worker),
            maximum_batch_size=configuration.maximum_batch_size,
            batch_collection_timeout_microseconds=(configuration.batch_collection_timeout_microseconds),
            cache_capacity=configuration.cache_capacity,
            inference_target=configuration.inference_target,
        )

    def new_game(self, starting_fen: str, moves_uci: tuple[str, ...]) -> NativeInteractiveGame:
        return NativeInteractiveGame(self._engine.new_game(starting_fen, moves_uci))


class NativeInteractiveGame:
    """Serial web view of a native history-aware game and reusable search tree."""

    def __init__(self, game: OptimizedInteractiveGame) -> None:
        self._game = game

    @property
    def fen(self) -> str:
        return self._game.fen

    @property
    def is_game_over(self) -> bool:
        return chess.Board(self.fen).is_game_over(claim_draw=True)

    @property
    def result(self) -> str | None:
        board = chess.Board(self.fen)
        return board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else None

    def apply_move(self, move_uci: str) -> None:
        self._game.apply_move(move_uci)

    def analyze(self, limit: AnalysisLimit) -> AnalysisResult:
        match limit:
            case TimedAnalysis(mode=mode, time_limit_seconds=seconds):
                result = self._game.analyze(
                    mode=_optimized_mode(mode),
                    time_limit_seconds=(seconds if mode is AnalysisMode.MCTS else None),
                )
            case CountedAnalysis(mode=mode, searches=searches):
                result = self._game.analyze(
                    mode=_optimized_mode(mode),
                    search_limit=searches if mode is AnalysisMode.MCTS else None,
                )
        return _web_result(result)


def _optimized_mode(mode: AnalysisMode) -> OptimizedAnalysisMode:
    match mode:
        case AnalysisMode.POLICY:
            return OptimizedAnalysisMode.POLICY
        case AnalysisMode.MCTS:
            return OptimizedAnalysisMode.MCTS


def _web_result(result: OptimizedAnalysisResult) -> AnalysisResult:
    outcome = (
        None
        if result.outcome is None
        else OutcomePrediction(
            win=result.outcome.win,
            draw=result.outcome.draw,
            loss=result.outcome.loss,
        )
    )
    return AnalysisResult(
        chosen_move_uci=result.chosen_move_uci,
        root_value=result.value,
        outcome_prediction=outcome,
        candidates=tuple(
            CandidateMove(
                move_uci=candidate.move_uci,
                policy_prior=candidate.policy_prior,
                visits=candidate.visits,
                visit_share=candidate.visit_share,
                mean_search_value=candidate.mean_value,
            )
            for candidate in result.candidates
        ),
        metrics=SearchMetrics(
            searches=result.searches,
            maximum_depth=result.maximum_depth,
            elapsed_milliseconds=result.elapsed_milliseconds,
        ),
        principal_variation=tuple(result.principal_variation) or None,
    )
