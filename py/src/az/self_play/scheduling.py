from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ScheduledGame:
    logical_worker_index: int
    game_index: int


@dataclass
class LogicalWorkerGameScheduler:
    first_worker_index: int
    worker_count: int
    next_game_indices: tuple[int, ...] = ()
    _worker_order: deque[int] = field(init=False)
    _next_game_indices: dict[int, int] = field(init=False)

    def __post_init__(self) -> None:
        if self.first_worker_index < 0 or self.worker_count <= 0:
            raise ValueError('Logical worker scheduling requires a nonnegative start and positive count.')
        worker_indices = range(self.first_worker_index, self.first_worker_index + self.worker_count)
        if self.next_game_indices and (
            len(self.next_game_indices) != self.worker_count or any(index < 0 for index in self.next_game_indices)
        ):
            raise ValueError('Logical worker resume indices must cover every worker and be nonnegative.')
        self._worker_order = deque(worker_indices)
        starts = self.next_game_indices or (0,) * self.worker_count
        self._next_game_indices = dict(zip(worker_indices, starts, strict=True))

    def next_game(self) -> ScheduledGame:
        logical_worker_index = self._worker_order[0]
        self._worker_order.rotate(-1)
        game_index = self._next_game_indices[logical_worker_index]
        self._next_game_indices[logical_worker_index] = game_index + 1
        return ScheduledGame(
            logical_worker_index=logical_worker_index,
            game_index=game_index,
        )
