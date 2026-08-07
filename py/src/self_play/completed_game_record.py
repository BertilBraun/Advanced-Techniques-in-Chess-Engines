from pathlib import Path
from typing import Annotated, TypeAlias

from pydantic import Field, TypeAdapter

from src.self_play.chess_completed_game import ChessCompletedGame
from src.self_play.completed_game import identity_from_file_name
from src.self_play.go_completed_game import GoCompletedGame


CompletedGame: TypeAlias = Annotated[ChessCompletedGame | GoCompletedGame, Field(discriminator='game')]
_COMPLETED_GAME_ADAPTER = TypeAdapter(CompletedGame)


def completed_game_from_path(path: Path) -> CompletedGame:
    game = _COMPLETED_GAME_ADAPTER.validate_json(path.read_text(encoding='utf-8'))
    if game.identity != identity_from_file_name(path.name):
        raise ValueError(f'Completed-game identity does not match its file name: {path}')
    return game
