from __future__ import annotations

from enum import Enum
from math import isfinite
from pathlib import Path
import re
from typing import Literal

from pydantic import Field, model_validator

from src.self_play.chess_completed_game import SparseSearchVisit
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel


GO_COMPLETED_GAME_SCHEMA_VERSION = 2
_GAME_FILE_PATTERN = re.compile(r'run-(?P<run>\d+)-worker-(?P<worker>\d+)-game-(?P<game>\d+)\.json')


class GoTerminationReason(str, Enum):
    TWO_PASSES = 'two_passes'
    MAXIMUM_MOVES = 'maximum_moves'


class GoMoveSelectionMode(str, Enum):
    TEMPERATURE = 'temperature'
    GREEDY = 'greedy'


class GoGameIdentity(FrozenModel):
    run_id: int = Field(ge=0)
    worker_id: int = Field(ge=0)
    game_number: int = Field(ge=0)

    @property
    def file_name(self) -> str:
        return f'run-{self.run_id}-worker-{self.worker_id}-game-{self.game_number:020d}.json'

    @property
    def archive_key(self) -> str:
        return f'{self.run_id}:{self.worker_id}:{self.game_number}'


class GoRulesMetadata(FrozenModel):
    scoring: Literal['area'] = 'area'
    komi_half_points: int
    maximum_moves: int = Field(gt=0)


class GoRepresentationMetadata(FrozenModel):
    board_size: Literal[7, 9]
    history_length: Literal[8] = 8
    action_encoding: Literal['go-point-pass-v1'] = 'go-point-pass-v1'
    canonical_player_perspective: bool = True


class GoSearchObservation(FrozenModel):
    ply: int = Field(ge=0)
    model_generation: int = Field(ge=0)
    legal_action_ids: tuple[int, ...] = Field(min_length=1)
    visits: tuple[SparseSearchVisit, ...] = Field(min_length=1)
    root_value: float
    selected_action_id: int = Field(ge=0)
    move_selection_mode: GoMoveSelectionMode
    search_budget: int = Field(gt=0)
    minimum_visit_count: int = Field(default=0, ge=0)
    sample_eligible: bool = True
    sample_weight: float = Field(default=1.0, gt=0.0)

    @model_validator(mode='after')
    def validate_observation(self) -> GoSearchObservation:
        if not isfinite(self.root_value) or not -1.0 <= self.root_value <= 1.0:
            raise ValueError('Go search root value must be finite and in [-1, 1].')
        if len(set(self.legal_action_ids)) != len(self.legal_action_ids):
            raise ValueError('Go legal actions must be unique.')
        legal = set(self.legal_action_ids)
        if self.selected_action_id not in legal:
            raise ValueError('Selected Go action must be legal.')
        visit_actions = tuple(visit.action_id for visit in self.visits)
        if len(set(visit_actions)) != len(visit_actions) or not set(visit_actions) <= legal:
            raise ValueError('Go search visits must uniquely reference legal actions.')
        if sum(visit.visit_count for visit in self.visits) <= 0:
            raise ValueError('Go search visits must contain a positive total.')
        return self


class GoCompletedGame(FrozenModel):
    schema_version: Literal[2] = GO_COMPLETED_GAME_SCHEMA_VERSION
    game: Literal['go'] = 'go'
    identity: GoGameIdentity
    rules: GoRulesMetadata
    representation: GoRepresentationMetadata
    model_generation: int = Field(ge=0)
    minimum_model_generation: int = Field(ge=0)
    created_at_seconds: float = Field(ge=0.0)
    generation_seconds: float = Field(ge=0.0)
    actions: tuple[int, ...]
    final_current_player: int
    final_score: float
    termination_reason: GoTerminationReason
    observations: tuple[GoSearchObservation, ...]

    @model_validator(mode='after')
    def validate_game(self) -> GoCompletedGame:
        if self.minimum_model_generation > self.model_generation:
            raise ValueError('Minimum model generation cannot exceed the completion generation.')
        if self.final_current_player not in (-1, 1):
            raise ValueError('Final Go current player must be -1 or 1.')
        action_count = self.representation.board_size**2 + 1
        if any(not 0 <= action < action_count for action in self.actions):
            raise ValueError('Completed Go game contains an action outside its action space.')
        if tuple(observation.ply for observation in self.observations) != tuple(
            sorted(observation.ply for observation in self.observations)
        ) or len({observation.ply for observation in self.observations}) != len(self.observations):
            raise ValueError('Go search observations must have unique ordered plies.')
        if any(observation.ply >= len(self.actions) for observation in self.observations):
            raise ValueError('Go search observations must precede their selected action.')
        if not isfinite(self.final_score) or not -1.0 <= self.final_score <= 1.0:
            raise ValueError('Completed Go game requires a finite final score in [-1, 1].')
        if any(
            not self.minimum_model_generation <= observation.model_generation <= self.model_generation
            for observation in self.observations
        ):
            raise ValueError('Go observation model generation is outside the game range.')
        return self


class GoCompletedGamePublisherState(FrozenModel):
    schema_version: Literal[1] = 1
    next_game_number: int = Field(ge=0)


class GoCompletedGamePublisher:
    def __init__(self, run_path: Path, run_id: int, worker_id: int) -> None:
        if run_id < 0 or worker_id < 0:
            raise ValueError('Run and worker identifiers must be nonnegative.')
        self.inbox_path = run_path / 'completed-games' / 'inbox'
        self.state_path = run_path / 'completed-games' / 'publishers' / f'worker-{worker_id:010d}.json'
        self.run_id = run_id
        self.worker_id = worker_id

    def reserve_identity(self) -> GoGameIdentity:
        state = self._load_state()
        identity = GoGameIdentity(
            run_id=self.run_id,
            worker_id=self.worker_id,
            game_number=state.next_game_number,
        )
        write_text_atomically(
            self.state_path,
            GoCompletedGamePublisherState(next_game_number=state.next_game_number + 1).model_dump_json(indent=2) + '\n',
        )
        return identity

    def publish(self, game: GoCompletedGame) -> Path:
        if game.identity.run_id != self.run_id or game.identity.worker_id != self.worker_id:
            raise ValueError('Completed-game identity does not belong to this publisher.')
        path = self.inbox_path / game.identity.file_name
        if path.exists():
            existing = GoCompletedGame.model_validate_json(path.read_text(encoding='utf-8'))
            if existing != game:
                raise ValueError(f'Completed-game file already exists with different content: {path}')
            return path
        write_text_atomically(path, game.model_dump_json() + '\n')
        return path

    def _load_state(self) -> GoCompletedGamePublisherState:
        if not self.state_path.exists():
            return GoCompletedGamePublisherState(next_game_number=0)
        return GoCompletedGamePublisherState.model_validate_json(self.state_path.read_text(encoding='utf-8'))


def go_completed_game_from_path(path: Path) -> GoCompletedGame:
    game = GoCompletedGame.model_validate_json(path.read_text(encoding='utf-8'))
    match = _GAME_FILE_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f'Invalid completed-game file name: {path.name}')
    expected = GoGameIdentity(
        run_id=int(match.group('run')),
        worker_id=int(match.group('worker')),
        game_number=int(match.group('game')),
    )
    if game.identity != expected:
        raise ValueError(f'Completed-game identity does not match its file name: {path}')
    return game
