from __future__ import annotations

from dataclasses import dataclass

from src.eval.InteractiveEngine import (
    AnalysisMode,
    AnalysisResult,
    InferenceTarget,
    InteractiveEngine,
    InteractiveGame,
)
from src.uci.server import SearchMode, SearchRequest


@dataclass(frozen=True)
class OptimizedEngineConfiguration:
    model_path: str
    device_id: int
    parallel_searches: int
    exploration_constant: float
    inference_workers: int
    outstanding_batches_per_worker: int
    maximum_batch_size: int
    inference_target: InferenceTarget


class OptimizedUciEngine:
    """Thin UCI adaptation over the production direct interactive engine."""

    def __init__(self, configuration: OptimizedEngineConfiguration) -> None:
        self._engine = InteractiveEngine(
            model_path=configuration.model_path,
            device_id=configuration.device_id,
            parallel_searches=configuration.parallel_searches,
            c_param=configuration.exploration_constant,
            inference_workers=configuration.inference_workers,
            outstanding_batches_per_worker=configuration.outstanding_batches_per_worker,
            maximum_batch_size=configuration.maximum_batch_size,
            inference_target=configuration.inference_target,
        )

    def new_game(
        self, starting_fen: str, moves_uci: tuple[str, ...]
    ) -> OptimizedUciGame:
        return OptimizedUciGame(self._engine.new_game(starting_fen, moves_uci))


class OptimizedUciGame:
    def __init__(self, game: InteractiveGame) -> None:
        self._game = game

    def apply_move(self, move_uci: str) -> None:
        self._game.apply_move(move_uci)

    def analyze(self, request: SearchRequest) -> AnalysisResult:
        mode = (
            AnalysisMode.POLICY
            if request.mode is SearchMode.POLICY
            else AnalysisMode.MCTS
        )
        if mode is AnalysisMode.POLICY:
            return self._game.analyze(mode=mode)

        result: AnalysisResult | None = None
        for _ in range(request.time_limit_seconds):
            result = self._game.analyze(mode=mode, time_limit_seconds=1)
            if request.stop_event.is_set():
                break
        assert result is not None
        return result
