from __future__ import annotations

import heapq
import os
from itertools import islice
from pathlib import Path

COMPLETED_GAME_SUFFIX = '.json'
WORKER_COUNTER_DIGITS = 15
DEFAULT_RENAME_CAP = 4_096


def worker_directory_path(completed_games_path: Path, worker_index: int) -> Path:
    return completed_games_path / f'worker-{worker_index}'


def worker_directory_paths(completed_games_path: Path, worker_count: int) -> tuple[Path, ...]:
    return tuple(worker_directory_path(completed_games_path, index) for index in range(worker_count))


def worker_source_file_name(counter: int, completed_game_file_name: str) -> str:
    return f'{counter:0{WORKER_COUNTER_DIGITS}d}-{completed_game_file_name}'


def parse_worker_source_file_name(file_name: str) -> tuple[int, str]:
    counter, separator, completed_game_file_name = file_name.partition('-')
    if not separator or len(counter) != WORKER_COUNTER_DIGITS or not counter.isdigit():
        raise ValueError(f'Replay worker source file name is invalid: {file_name}')
    return int(counter), completed_game_file_name


def worker_source_file_names(worker_path: Path) -> list[str]:
    try:
        with os.scandir(worker_path) as entries:
            return [entry.name for entry in entries if entry.name.endswith(COMPLETED_GAME_SUFFIX)]
    except FileNotFoundError:
        return []


def next_worker_counter(worker_path: Path) -> int:
    names = worker_source_file_names(worker_path)
    if not names:
        return 0
    return max(parse_worker_source_file_name(name)[0] for name in names) + 1


class InboxDispatcher:
    """Moves completed games out of the inbox into a per-worker directory, round-robin."""

    def __init__(self, inbox_path: Path, worker_paths: tuple[Path, ...], rename_cap: int = DEFAULT_RENAME_CAP) -> None:
        assert worker_paths
        self.inbox_path = inbox_path
        self.worker_paths = worker_paths
        self.rename_cap = rename_cap
        self._counters = [next_worker_counter(path) for path in worker_paths]
        self.dispatched_games = 0

    def dispatch_once(self) -> int:
        dispatched = 0
        try:
            entries = os.scandir(self.inbox_path)
        except FileNotFoundError:
            return 0
        # One listing per worker per pass. A blind round robin hands a stalled worker an equal share
        # every pass regardless of what it still owes, so its backlog never recovers.
        depths = [(len(worker_source_file_names(path)), index) for index, path in enumerate(self.worker_paths)]
        heapq.heapify(depths)
        with entries:
            # islice over the lazy scandir iterator keeps a pass O(rename_cap), never O(inbox depth).
            candidates = (entry for entry in entries if entry.name.endswith(COMPLETED_GAME_SUFFIX))
            for entry in islice(candidates, self.rename_cap):
                depth, index = heapq.heappop(depths)
                if self._dispatch_entry(entry.name, index):
                    dispatched += 1
                    depth += 1
                heapq.heappush(depths, (depth, index))
        self.dispatched_games += dispatched
        return dispatched

    def _dispatch_entry(self, file_name: str, index: int) -> bool:
        target = self.worker_paths[index] / worker_source_file_name(self._counters[index], file_name)
        try:
            os.rename(self.inbox_path / file_name, target)
        except OSError:
            return False
        self._counters[index] += 1
        return True
