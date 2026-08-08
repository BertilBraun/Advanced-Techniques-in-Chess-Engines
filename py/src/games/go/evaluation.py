from __future__ import annotations

from src.games.go.contract import NativeGoPosition
from src.games.go.katago import KataGoClient


class KataGoMatchEngine:
    def __init__(self, client: KataGoClient) -> None:
        self.client = client

    def choose_actions(
        self,
        positions: tuple[NativeGoPosition, ...],
        action_sequences: tuple[tuple[int, ...], ...],
    ) -> tuple[int, ...]:
        if len(positions) != len(action_sequences):
            raise ValueError('KataGo match positions and histories must have equal lengths.')
        return self.client.choose_actions(action_sequences)

    def close(self) -> None:
        self.client.close()
