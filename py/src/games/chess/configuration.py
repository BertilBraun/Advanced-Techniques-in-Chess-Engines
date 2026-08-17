from __future__ import annotations

from typing import Literal

from pydantic import model_validator

from src.experiment.base_configuration import BaseExperimentConfiguration
from src.experiment.generation_schedule import IntegerGenerationSchedule
from src.games.chess.contract import CHESS_NETWORK_DIMENSIONS
from src.games.representation import NetworkDimensions
from src.self_play.configuration import SelfPlayConfiguration
from src.self_play.resignation import ResignationConfiguration
from src.training.configuration import TrainingObjectiveConfiguration
from src.util.frozen_model import FrozenModel


class ChessSelfPlayConfiguration(SelfPlayConfiguration):
    maximum_game_plies: IntegerGenerationSchedule | None = None
    maximum_ply_syzygy_paths: tuple[str, ...] | None = None
    resignation: ResignationConfiguration

    @model_validator(mode='after')
    def validate_syzygy_paths(self) -> ChessSelfPlayConfiguration:
        if self.maximum_ply_syzygy_paths is not None and (
            not self.maximum_ply_syzygy_paths or any(not path.strip() for path in self.maximum_ply_syzygy_paths)
        ):
            raise ValueError('Maximum-ply Syzygy paths must contain at least one nonempty directory path.')
        return self


class ChessConfiguration(FrozenModel):
    self_play: ChessSelfPlayConfiguration
    objective: TrainingObjectiveConfiguration


class ChessExperimentConfiguration(BaseExperimentConfiguration):
    game: Literal['chess'] = 'chess'
    chess: ChessConfiguration

    @property
    def network_dimensions(self) -> NetworkDimensions:
        return CHESS_NETWORK_DIMENSIONS

    @model_validator(mode='after')
    def validate_experiment(self) -> ChessExperimentConfiguration:
        self.training.validate_game(self.network_dimensions.actions, self.chess.self_play)
        if self.evaluation.engine.kind != 'stockfish':
            raise ValueError('Chess evaluation requires the Stockfish engine configuration.')
        return self
