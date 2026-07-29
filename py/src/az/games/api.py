from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class GameIdentifier(str, Enum):
    GO = 'go'


@dataclass(frozen=True)
class GameModuleRegistration:
    identifier: GameIdentifier
    display_name: str
    payload_schema_name: str


class GameRegistry:
    def resolve(self, identifier: GameIdentifier) -> GameModuleRegistration:
        match identifier:
            case GameIdentifier.GO:
                from src.az.games.go.module import GO_GAME_MODULE

                return GO_GAME_MODULE


def create_game_registry() -> GameRegistry:
    return GameRegistry()
