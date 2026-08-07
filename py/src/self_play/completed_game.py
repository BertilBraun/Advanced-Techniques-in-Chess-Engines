from __future__ import annotations

from pathlib import Path
import re
from typing import Literal

from pydantic import Field

from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel


_GAME_FILE_PATTERN = re.compile(r'run-(?P<run>\d+)-worker-(?P<worker>\d+)-game-(?P<game>\d+)\.json')


class GameIdentity(FrozenModel):
    run_id: int = Field(ge=0)
    worker_id: int = Field(ge=0)
    game_number: int = Field(ge=0)

    @property
    def file_name(self) -> str:
        return f'run-{self.run_id}-worker-{self.worker_id}-game-{self.game_number:020d}.json'

    @property
    def archive_key(self) -> str:
        return f'{self.run_id}:{self.worker_id}:{self.game_number}'


class SparseSearchVisit(FrozenModel):
    action_id: int = Field(ge=0)
    visit_count: int = Field(ge=0)


class CompletedGameRecord(FrozenModel):
    identity: GameIdentity


class CompletedGamePublisherState(FrozenModel):
    schema_version: Literal[1] = 1
    next_game_number: int = Field(ge=0)


class CompletedGamePublisher:
    def __init__(self, run_path: Path, run_id: int, worker_id: int) -> None:
        if run_id < 0 or worker_id < 0:
            raise ValueError('Run and worker identifiers must be nonnegative.')
        self.inbox_path = run_path / 'completed-games' / 'inbox'
        self.state_path = run_path / 'completed-games' / 'publishers' / f'worker-{worker_id:010d}.json'
        self.run_id = run_id
        self.worker_id = worker_id

    def reserve_identity(self) -> GameIdentity:
        state = self._load_state()
        identity = GameIdentity(
            run_id=self.run_id,
            worker_id=self.worker_id,
            game_number=state.next_game_number,
        )
        next_state = CompletedGamePublisherState(next_game_number=state.next_game_number + 1)
        write_text_atomically(self.state_path, next_state.model_dump_json(indent=2) + '\n')
        return identity

    def publish(self, game: CompletedGameRecord) -> Path:
        if game.identity.run_id != self.run_id or game.identity.worker_id != self.worker_id:
            raise ValueError('Completed-game identity does not belong to this publisher.')
        path = self.inbox_path / game.identity.file_name
        payload = game.model_dump_json() + '\n'
        if path.exists():
            if path.read_text(encoding='utf-8') != payload:
                raise ValueError(f'Completed-game file already exists with different content: {path}')
            return path
        write_text_atomically(path, payload)
        return path

    def _load_state(self) -> CompletedGamePublisherState:
        if not self.state_path.exists():
            return CompletedGamePublisherState(next_game_number=0)
        return CompletedGamePublisherState.model_validate_json(self.state_path.read_text(encoding='utf-8'))


def identity_from_file_name(file_name: str) -> GameIdentity:
    match = _GAME_FILE_PATTERN.fullmatch(file_name)
    if match is None:
        raise ValueError(f'Invalid completed-game file name: {file_name}')
    return GameIdentity(
        run_id=int(match.group('run')),
        worker_id=int(match.group('worker')),
        game_number=int(match.group('game')),
    )
