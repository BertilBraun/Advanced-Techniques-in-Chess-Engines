"""Game configuration aggregation for inspection outside complete experiments."""

from src.az.games.chess.configuration import ChessGameConfiguration
from src.az.games.go.configuration import GoGameConfiguration


GameConfiguration = GoGameConfiguration | ChessGameConfiguration

__all__ = ["GameConfiguration"]
