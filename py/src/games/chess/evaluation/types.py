from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from src.games.chess.board import Player
from src.games.chess.board import ChessBoard


@dataclass
class Results:
    wins: int
    losses: int
    draws: int

    def __add__(self, other: Results) -> Results:
        return Results(
            wins=self.wins + other.wins,
            losses=self.losses + other.losses,
            draws=self.draws + other.draws,
        )

    def __sub__(self, other: Results) -> Results:
        return Results(
            wins=self.wins + other.losses,
            losses=self.losses + other.wins,
            draws=self.draws + other.draws,
        )

    def update(self, result: Player | None, main_player: Player) -> None:
        if result is None:
            self.draws += 1
        elif result == main_player:
            self.wins += 1
        else:
            self.losses += 1

    def __neg__(self) -> Results:
        return Results(self.losses, self.wins, self.draws)

    def __str__(self) -> str:
        return f'W/D/L: {self.wins}/{self.draws}/{self.losses}'


EvaluationModel = Callable[[list[ChessBoard]], list[np.ndarray]]


@dataclass(frozen=True)
class EvaluationMove:
    policy: np.ndarray


@dataclass(frozen=True)
class EvaluationTerminal:
    pass


PairedEvaluationDecision = EvaluationMove | EvaluationTerminal
PairedEvaluationModel = Callable[[list[ChessBoard]], list[PairedEvaluationDecision]]
