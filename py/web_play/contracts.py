from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class AnalysisMode(str, Enum):
    POLICY = "policy"
    MCTS = "mcts"


class SideToMove(str, Enum):
    WHITE = "white"
    BLACK = "black"


class OutcomePerspective(str, Enum):
    SIDE_TO_MOVE = "side_to_move"


class OutcomePrediction(FrozenModel):
    win: float = Field(ge=0.0, le=1.0)
    draw: float = Field(ge=0.0, le=1.0)
    loss: float = Field(ge=0.0, le=1.0)
    perspective: Literal[OutcomePerspective.SIDE_TO_MOVE] = (
        OutcomePerspective.SIDE_TO_MOVE
    )


class CandidateMove(FrozenModel):
    move_uci: str
    policy_prior: float = Field(ge=0.0, le=1.0)
    visits: int = Field(ge=0)
    visit_share: float = Field(ge=0.0, le=1.0)
    mean_search_value: float | None = Field(default=None, ge=-1.0, le=1.0)


class SearchMetrics(FrozenModel):
    searches: int = Field(ge=0)
    maximum_depth: int = Field(ge=0)
    elapsed_milliseconds: int = Field(ge=0)


class AnalysisResult(FrozenModel):
    chosen_move_uci: str
    root_value: float = Field(ge=-1.0, le=1.0)
    outcome_prediction: OutcomePrediction | None
    candidates: tuple[CandidateMove, ...]
    metrics: SearchMetrics
    principal_variation: tuple[str, ...] | None


class AnalysisOptions(FrozenModel):
    mode: AnalysisMode
    time_limit_seconds: Annotated[int, Field(ge=1, le=30)]


class CreateGameRequest(FrozenModel):
    starting_fen: str
    moves_uci: Annotated[tuple[str, ...], Field(max_length=1024)]


class GameState(FrozenModel):
    starting_fen: str
    moves_uci: tuple[str, ...]
    fen: str
    side_to_move: SideToMove
    game_over: bool
    result: str | None


class CreateGameResponse(FrozenModel):
    game_token: UUID
    state: GameState


class PlayTurnRequest(FrozenModel):
    starting_fen: str
    moves_uci: Annotated[tuple[str, ...], Field(max_length=1024)]
    human_move_uci: str | None
    analysis: AnalysisOptions


class PlayTurnResponse(FrozenModel):
    state: GameState
    engine_move_uci: str | None
    analysis: AnalysisResult | None


@dataclass(frozen=True)
class TimedAnalysis:
    mode: AnalysisMode
    time_limit_seconds: int

    def __post_init__(self) -> None:
        if not 1 <= self.time_limit_seconds <= 30:
            raise ValueError("time_limit_seconds must be between 1 and 30.")


@dataclass(frozen=True)
class CountedAnalysis:
    mode: AnalysisMode
    searches: int

    def __post_init__(self) -> None:
        if self.searches < 1:
            raise ValueError("searches must be positive.")


AnalysisLimit = TimedAnalysis | CountedAnalysis
