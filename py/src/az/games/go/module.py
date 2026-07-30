from __future__ import annotations

import torch

from src.az.games.api import GameIdentifier, GameModuleRegistration
from src.az.games.go.configuration import (
    GoGameConfiguration,
    GoObjectiveConfiguration,
    ResidualGoModelConfiguration,
)
from src.az.games.go.training import GoTrainingModule


GO_GAME_MODULE = GameModuleRegistration(
    identifier=GameIdentifier.GO,
    display_name='Go',
    payload_schema_name='go-training-payload',
)


def create_go_training_module(
    game_configuration: GoGameConfiguration,
    model_configuration: ResidualGoModelConfiguration,
    objective_configuration: GoObjectiveConfiguration,
    payload_schema_version: int,
    device: torch.device,
    model_initialization_seed: int,
) -> GoTrainingModule:
    return GoTrainingModule(
        game_configuration=game_configuration,
        model_configuration=model_configuration,
        objective_configuration=objective_configuration,
        payload_schema_version=payload_schema_version,
        device=device,
        model_initialization_seed=model_initialization_seed,
    )
