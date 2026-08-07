from __future__ import annotations

from dataclasses import dataclass
from threading import Event
from typing import TYPE_CHECKING

from src.games.chess.interactive.analysis import (
    AnalysisRequest,
    AnalysisResult,
    PolicyAnalysis,
    TimedMctsAnalysis,
)

if TYPE_CHECKING:
    from src.games.chess.interactive.engine import InteractiveEngine, InteractiveGame


@dataclass(frozen=True)
class UciConfiguration:
    search_slice_seconds: int = 5

    def __post_init__(self) -> None:
        if not 1 <= self.search_slice_seconds <= 30:
            raise ValueError('search_slice_seconds must be in [1, 30].')


class UciEngine:
    """Adds interruptible UCI time slicing to the shared interactive engine."""

    def __init__(self, engine: InteractiveEngine, configuration: UciConfiguration) -> None:
        self._engine = engine
        self._configuration = configuration

    def new_game(self, starting_fen: str, moves_uci: tuple[str, ...]) -> UciGame:
        return UciGame(self._engine.new_game(starting_fen, moves_uci), self._configuration)


class UciGame:
    def __init__(self, game: InteractiveGame, configuration: UciConfiguration) -> None:
        self._game = game
        self._configuration = configuration

    def apply_move(self, move_uci: str) -> None:
        self._game.apply_move(move_uci)

    def analyze(self, request: AnalysisRequest, stop_event: Event) -> AnalysisResult:
        match request:
            case PolicyAnalysis():
                return self._game.analyze(request)
            case TimedMctsAnalysis(seconds=seconds):
                result: AnalysisResult | None = None
                remaining_seconds = seconds
                while remaining_seconds > 0:
                    slice_seconds = min(remaining_seconds, self._configuration.search_slice_seconds)
                    result = self._game.analyze(TimedMctsAnalysis(seconds=slice_seconds))
                    remaining_seconds -= slice_seconds
                    if stop_event.is_set():
                        break
                assert result is not None
                return result
            case _:
                raise ValueError('UCI supports policy or timed MCTS analysis.')
