from __future__ import annotations

import os

from web_play.api import create_app
from web_play.native_adapter import NativeEngineConfiguration, NativeInteractiveEngine
from web_play.service import GameService


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise ValueError(f"Required local setting {name} is missing.")
    return value.strip()


model_path = _required_environment("CHESS_MODEL_PATH")
allowed_origins = tuple(
    origin.strip().rstrip("/")
    for origin in _required_environment("CHESS_WEB_ALLOWED_ORIGINS").split(",")
    if origin.strip()
)
engine = NativeInteractiveEngine(
    NativeEngineConfiguration.for_model(model_path)
)
app = create_app(GameService(engine), allowed_origins)
