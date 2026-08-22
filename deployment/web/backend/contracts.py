from __future__ import annotations

from enum import Enum
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field
from src.games.chess.interactive.analysis import AnalysisRequest, AnalysisResult


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')


class SideToMove(str, Enum):
    WHITE = 'white'
    BLACK = 'black'


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
    analysis: AnalysisRequest


class PlayTurnResponse(FrozenModel):
    state: GameState
    engine_move_uci: str | None
    analysis: AnalysisResult | None
