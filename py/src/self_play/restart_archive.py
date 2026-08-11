from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3

from pydantic import TypeAdapter

from src.self_play.completed_game import CompletedSelfPlayGame
from src.self_play.parameters import RestartStateStartParameters
from src.self_play.policy import ordered_search_visits


ACTION_PREFIX_ADAPTER = TypeAdapter(tuple[int, ...])


@dataclass(frozen=True)
class ReservedRestart:
    action_prefix: tuple[int, ...]
    action_id: int


@dataclass(frozen=True)
class RestartArchiveUpdate:
    positions: int
    candidates: int
    capacity_evictions: int


@dataclass(frozen=True)
class RestartArchiveSnapshot:
    positions: int
    candidates: int
    archived_positions: int
    archived_candidates: int
    exhausted_evictions: int
    expired_evictions: int
    capacity_evictions: int


@dataclass(frozen=True)
class _ArchivePosition:
    key: str
    source_generation: int
    created_at_seconds: float
    action_prefix: tuple[int, ...]
    candidate_action_ids: tuple[int, ...]
    played_action_id: int


class RestartStateArchive:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=30.0, isolation_level=None)
        self.connection.execute('PRAGMA busy_timeout = 30000')
        self.connection.execute('PRAGMA foreign_keys = ON')
        journal_mode = str(self.connection.execute('PRAGMA journal_mode').fetchone()[0])
        if journal_mode.casefold() != 'wal':
            self.connection.execute('PRAGMA journal_mode = WAL')
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS restart_position (
                position_key TEXT PRIMARY KEY,
                source_generation INTEGER NOT NULL,
                created_at_seconds REAL NOT NULL,
                action_prefix TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS restart_candidate (
                position_key TEXT NOT NULL REFERENCES restart_position(position_key) ON DELETE CASCADE,
                action_id INTEGER NOT NULL,
                tried INTEGER NOT NULL CHECK (tried IN (0, 1)),
                PRIMARY KEY (position_key, action_id)
            );
            CREATE TABLE IF NOT EXISTS restart_counter (
                name TEXT PRIMARY KEY,
                value INTEGER NOT NULL
            );
            """,
        )

    def close(self) -> None:
        self.connection.close()

    def archive_completed_game(
        self,
        game: CompletedSelfPlayGame,
        parameters: RestartStateStartParameters,
    ) -> RestartArchiveUpdate:
        positions = _eligible_positions(game, parameters)
        inserted_positions = 0
        inserted_candidates = 0
        self._begin_write()
        try:
            for position in positions:
                cursor = self.connection.execute(
                    'INSERT OR IGNORE INTO restart_position VALUES (?, ?, ?, ?)',
                    (
                        position.key,
                        position.source_generation,
                        position.created_at_seconds,
                        ACTION_PREFIX_ADAPTER.dump_json(position.action_prefix).decode('utf-8'),
                    ),
                )
                if cursor.rowcount == 0:
                    continue
                inserted_positions += 1
                inserted_candidates += len(position.candidate_action_ids)
                self.connection.executemany(
                    'INSERT INTO restart_candidate VALUES (?, ?, ?)',
                    (
                        (position.key, action_id, int(action_id == position.played_action_id))
                        for action_id in position.candidate_action_ids
                    ),
                )
            capacity_evictions = self._evict_over_capacity(parameters.maximum_archive_positions)
            self._increment_counter('archived_positions', inserted_positions)
            self._increment_counter('archived_candidates', inserted_candidates)
            self._increment_counter('capacity_evictions', capacity_evictions)
            self.connection.commit()
        except BaseException:
            self.connection.rollback()
            raise
        return RestartArchiveUpdate(
            positions=inserted_positions,
            candidates=inserted_candidates,
            capacity_evictions=capacity_evictions,
        )

    def reserve(
        self,
        current_generation: int,
        parameters: RestartStateStartParameters,
    ) -> ReservedRestart | None:
        self._begin_write()
        try:
            expired_evictions = self._evict_expired(current_generation, parameters.maximum_age_generations)
            row = self.connection.execute(
                """
                SELECT position.position_key, position.action_prefix, candidate.action_id
                FROM restart_position AS position
                JOIN restart_candidate AS candidate USING (position_key)
                WHERE candidate.tried = 0
                ORDER BY position.created_at_seconds DESC, position.position_key, candidate.action_id
                LIMIT 1
                """,
            ).fetchone()
            self._increment_counter('expired_evictions', expired_evictions)
            if row is None:
                self.connection.commit()
                return None
            position_key = str(row[0])
            prefix = ACTION_PREFIX_ADAPTER.validate_json(str(row[1]))
            action_id = int(row[2])
            self.connection.execute(
                'UPDATE restart_candidate SET tried = 1 WHERE position_key = ? AND action_id = ?',
                (position_key, action_id),
            )
            untried = int(
                self.connection.execute(
                    'SELECT COUNT(*) FROM restart_candidate WHERE position_key = ? AND tried = 0',
                    (position_key,),
                ).fetchone()[0]
            )
            if untried == 0:
                self.connection.execute('DELETE FROM restart_position WHERE position_key = ?', (position_key,))
                self._increment_counter('exhausted_evictions', 1)
            self.connection.commit()
            return ReservedRestart(action_prefix=prefix, action_id=action_id)
        except BaseException:
            self.connection.rollback()
            raise

    def snapshot(self) -> RestartArchiveSnapshot:
        positions = int(self.connection.execute('SELECT COUNT(*) FROM restart_position').fetchone()[0])
        candidates = int(
            self.connection.execute('SELECT COUNT(*) FROM restart_candidate WHERE tried = 0').fetchone()[0]
        )
        return RestartArchiveSnapshot(
            positions=positions,
            candidates=candidates,
            archived_positions=self._counter('archived_positions'),
            archived_candidates=self._counter('archived_candidates'),
            exhausted_evictions=self._counter('exhausted_evictions'),
            expired_evictions=self._counter('expired_evictions'),
            capacity_evictions=self._counter('capacity_evictions'),
        )

    def _begin_write(self) -> None:
        self.connection.execute('BEGIN IMMEDIATE')

    def _evict_expired(self, current_generation: int, maximum_age_generations: int) -> int:
        threshold = current_generation - maximum_age_generations
        keys = tuple(
            str(row[0])
            for row in self.connection.execute(
                'SELECT position_key FROM restart_position WHERE source_generation < ?',
                (threshold,),
            )
        )
        self.connection.executemany('DELETE FROM restart_position WHERE position_key = ?', ((key,) for key in keys))
        return len(keys)

    def _evict_over_capacity(self, maximum_positions: int) -> int:
        position_count = int(self.connection.execute('SELECT COUNT(*) FROM restart_position').fetchone()[0])
        excess = max(0, position_count - maximum_positions)
        keys = tuple(
            str(row[0])
            for row in self.connection.execute(
                """
                SELECT position_key FROM restart_position
                ORDER BY created_at_seconds, position_key LIMIT ?
                """,
                (excess,),
            )
        )
        self.connection.executemany('DELETE FROM restart_position WHERE position_key = ?', ((key,) for key in keys))
        return len(keys)

    def _increment_counter(self, name: str, amount: int) -> None:
        self.connection.execute(
            """
            INSERT INTO restart_counter VALUES (?, ?)
            ON CONFLICT(name) DO UPDATE SET value = value + excluded.value
            """,
            (name, amount),
        )

    def _counter(self, name: str) -> int:
        row = self.connection.execute('SELECT value FROM restart_counter WHERE name = ?', (name,)).fetchone()
        return 0 if row is None else int(row[0])


def _eligible_positions(
    game: CompletedSelfPlayGame,
    parameters: RestartStateStartParameters,
) -> tuple[_ArchivePosition, ...]:
    eligible: list[_ArchivePosition] = []
    for observation in game.observations:
        if not observation.full_search:
            continue
        if len(game.action_ids) - observation.ply < parameters.minimum_remaining_plies:
            continue
        if abs(observation.root_value) > parameters.maximum_absolute_root_value:
            continue
        visits = ordered_search_visits(observation)
        if not visits:
            continue
        total_visits = sum(visit.visit_count for visit in visits)
        cumulative_visits = 0
        candidates = []
        for visit in visits:
            candidates.append(visit.action_id)
            cumulative_visits += visit.visit_count
            if cumulative_visits / total_visits >= parameters.candidate_visit_mass:
                break
        if not parameters.minimum_candidates <= len(candidates) <= parameters.maximum_candidates:
            continue
        eligible.append(
            _ArchivePosition(
                key=f'{game.identity.archive_key}:{observation.ply}',
                source_generation=observation.model_generation,
                created_at_seconds=game.created_at_seconds,
                action_prefix=game.action_ids[: observation.ply],
                candidate_action_ids=tuple(candidates),
                played_action_id=game.action_ids[observation.ply],
            )
        )
    return tuple(eligible)
