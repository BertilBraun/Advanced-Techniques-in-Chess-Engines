from __future__ import annotations

import os

from src.games.chess.interactive.configuration import InteractiveEngineConfiguration
from src.games.chess.interactive.engine import InteractiveEngine

from deployment.web.backend.api import create_app
from deployment.web.backend.service import GameService


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise ValueError(f'Required local setting {name} is missing.')
    return value.strip()


model_path = _required_environment('CHESS_MODEL_PATH')
allowed_origins = tuple(
    origin.strip().rstrip('/')
    for origin in _required_environment('CHESS_WEB_ALLOWED_ORIGINS').split(',')
    if origin.strip()
)
engine = InteractiveEngine(InteractiveEngineConfiguration(model_path=model_path))
app = create_app(GameService(engine), allowed_origins)
