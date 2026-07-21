from __future__ import annotations

from typing import Protocol

from web_play.contracts import AnalysisLimit, AnalysisResult


class InteractiveGame(Protocol):
    @property
    def fen(self) -> str: ...

    @property
    def is_game_over(self) -> bool: ...

    @property
    def result(self) -> str | None: ...

    def apply_move(self, move_uci: str) -> None: ...

    def analyze(self, limit: AnalysisLimit) -> AnalysisResult: ...


class InteractiveEngine(Protocol):
    def new_game(
        self, starting_fen: str, moves_uci: tuple[str, ...]
    ) -> InteractiveGame: ...
