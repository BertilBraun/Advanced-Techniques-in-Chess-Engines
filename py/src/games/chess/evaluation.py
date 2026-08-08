from __future__ import annotations

from src.games.chess.contract import ChessPosition
from src.games.chess.stockfish import StockfishClient


class StockfishMatchEngine:
    def __init__(self, client: StockfishClient, skill_level: int) -> None:
        self.client = client
        self.skill_level = skill_level

    def choose_actions(
        self,
        positions: tuple[ChessPosition, ...],
        action_sequences: tuple[tuple[int, ...], ...],
    ) -> tuple[int, ...]:
        if len(positions) != len(action_sequences):
            raise ValueError('Stockfish match positions and histories must have equal lengths.')
        return tuple(self.client.choose_action(position, self.skill_level) for position in positions)

    def close(self) -> None:
        self.client.close()
