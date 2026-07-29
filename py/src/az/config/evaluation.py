from __future__ import annotations

from typing import Literal

from pydantic import Field, PositiveInt, model_validator

from src.az.config.base import FrozenModel
from src.az.config.search import SearchConfiguration
from src.az.games.go.configuration import GoEvaluationSuite


EvaluationSuiteConfiguration = GoEvaluationSuite


class EvaluationConfiguration(FrozenModel):
    search: SearchConfiguration
    checkpoint_elapsed_seconds: tuple[PositiveInt, ...]
    paired_games_per_checkpoint: PositiveInt
    bootstrap_samples: PositiveInt
    confidence_method: Literal['paired_bootstrap']
    confidence_level: float = Field(gt=0, lt=1)
    bootstrap_seed: int = Field(ge=0, le=2**63 - 1)
    suite: EvaluationSuiteConfiguration

    @model_validator(mode='after')
    def validate_pairs(self) -> EvaluationConfiguration:
        if self.paired_games_per_checkpoint % 2 != 0:
            raise ValueError('Paired evaluation game count must be even.')
        if tuple(sorted(set(self.checkpoint_elapsed_seconds))) != self.checkpoint_elapsed_seconds:
            raise ValueError('Evaluation checkpoint times must be unique and strictly increasing.')
        return self
