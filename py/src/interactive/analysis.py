from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from src.util.frozen_model import FrozenModel


class PolicyAnalysis(FrozenModel):
    type: Literal['policy'] = 'policy'


class TimedMctsAnalysis(FrozenModel):
    type: Literal['timed_mcts'] = 'timed_mcts'
    seconds: Annotated[int, Field(ge=1, le=30)]


class CountedMctsAnalysis(FrozenModel):
    type: Literal['counted_mcts'] = 'counted_mcts'
    searches: Annotated[int, Field(ge=1)]


AnalysisRequest = Annotated[
    PolicyAnalysis | TimedMctsAnalysis | CountedMctsAnalysis,
    Field(discriminator='type'),
]


class OutcomePrediction(FrozenModel):
    """Network W/D/L probabilities from the analyzed side-to-move perspective."""

    win: float = Field(ge=0.0, le=1.0)
    draw: float = Field(ge=0.0, le=1.0)
    loss: float = Field(ge=0.0, le=1.0)


class CandidateAnalysis(FrozenModel):
    """A candidate value is from the analyzed side-to-move perspective."""

    move_uci: str
    policy_prior: float = Field(ge=0.0, le=1.0)
    visits: int = Field(ge=0)
    visit_share: float = Field(ge=0.0, le=1.0)
    mean_value: float | None = Field(default=None, ge=-1.0, le=1.0)


class AnalysisResult(FrozenModel):
    """Analysis ordered by final preference, viewed from the side to move."""

    chosen_move_uci: str
    value: float = Field(ge=-1.0, le=1.0)
    outcome: OutcomePrediction | None
    candidates: tuple[CandidateAnalysis, ...]
    searches: int = Field(ge=0)
    maximum_depth: int = Field(ge=0)
    elapsed_milliseconds: int = Field(ge=0)
    principal_variation: tuple[str, ...]
