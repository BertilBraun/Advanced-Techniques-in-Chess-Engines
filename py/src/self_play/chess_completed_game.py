from __future__ import annotations

from enum import Enum
from math import isfinite
from pathlib import Path
import re
from typing import Literal

from pydantic import Field, model_validator

from src.self_play.value_target import TerminationReason
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel


CHESS_COMPLETED_GAME_SCHEMA_VERSION = 1
CHESS_REPRESENTATION_VERSION = 1
_GAME_FILE_PATTERN = re.compile(r'run-(?P<run>\d+)-worker-(?P<worker>\d+)-game-(?P<game>\d+)\.json')


class ChessMoveSelectionMode(str, Enum):
    TEMPERATURE = 'temperature'
    GREEDY = 'greedy'
    TERMINAL = 'terminal'


class ChessRulesMetadata(FrozenModel):
    variant: Literal['standard'] = 'standard'
    chess960: bool = False
    automatic_fifty_move_draw: bool = True
    automatic_threefold_repetition_draw: bool = True


class ChessRepresentationMetadata(FrozenModel):
    version: Literal[1] = CHESS_REPRESENTATION_VERSION
    canonical_player_perspective: bool = True
    action_encoding: str = 'chess-move2index-v1'


class ChessGameIdentity(FrozenModel):
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


class ChessSearchObservation(FrozenModel):
    ply: int = Field(ge=0)
    model_generation: int = Field(ge=0)
    legal_action_ids: tuple[int, ...]
    visits: tuple[SparseSearchVisit, ...]
    root_value: float
    selected_action_id: int | None
    move_selection_mode: ChessMoveSelectionMode
    search_budget: int = Field(gt=0)
    minimum_visit_count: int = Field(ge=0)
    sample_eligible: bool = True
    sample_weight: float = Field(default=1.0, gt=0.0)

    @model_validator(mode='after')
    def validate_observation(self) -> ChessSearchObservation:
        if not isfinite(self.root_value) or not -1.0 <= self.root_value <= 1.0:
            raise ValueError('Chess search root value must be finite and in [-1, 1].')
        if not self.legal_action_ids or len(set(self.legal_action_ids)) != len(self.legal_action_ids):
            raise ValueError('Chess search legal actions must be nonempty and unique.')
        legal_actions = set(self.legal_action_ids)
        if self.selected_action_id is not None and self.selected_action_id not in legal_actions:
            raise ValueError('Selected chess action must be legal.')
        visit_actions = tuple(visit.action_id for visit in self.visits)
        if not visit_actions or len(set(visit_actions)) != len(visit_actions):
            raise ValueError('Chess search visits must be nonempty and unique.')
        if not set(visit_actions) <= legal_actions:
            raise ValueError('Chess search visits must reference legal actions.')
        if sum(visit.visit_count for visit in self.visits) <= 0:
            raise ValueError('Chess search visits must contain a positive total.')
        return self


class ChessCompletedGame(FrozenModel):
    schema_version: Literal[1] = CHESS_COMPLETED_GAME_SCHEMA_VERSION
    game: Literal['chess'] = 'chess'
    identity: ChessGameIdentity
    rules: ChessRulesMetadata
    representation: ChessRepresentationMetadata
    model_generation: int = Field(ge=0)
    minimum_model_generation: int = Field(ge=0)
    created_at_seconds: float = Field(ge=0.0)
    generation_seconds: float = Field(ge=0.0)
    initial_fen: str
    moves_uci: tuple[str, ...]
    final_current_player: int
    final_score: float
    termination_reason: TerminationReason
    resignation_audit: bool
    resignation_threshold: float | None
    resignation_trigger_ply: int | None = Field(default=None, ge=0)
    observations: tuple[ChessSearchObservation, ...]

    @model_validator(mode='after')
    def validate_game(self) -> ChessCompletedGame:
        if self.schema_version != CHESS_COMPLETED_GAME_SCHEMA_VERSION:
            raise ValueError(f'Unsupported chess completed-game schema {self.schema_version}.')
        if self.game != 'chess':
            raise ValueError('Completed-game record must identify chess.')
        if self.minimum_model_generation > self.model_generation:
            raise ValueError('Minimum model generation cannot exceed the completion generation.')
        if self.final_current_player not in (-1, 1):
            raise ValueError('Final current player must be -1 or 1.')
        if not isfinite(self.final_score) or not -1.0 <= self.final_score <= 1.0:
            raise ValueError('Final chess score must be finite and in [-1, 1].')
        if not self.initial_fen:
            raise ValueError('Completed chess game must include an initial FEN.')
        if tuple(sorted(observation.ply for observation in self.observations)) != tuple(
            observation.ply for observation in self.observations
        ):
            raise ValueError('Chess search observations must be ordered by ply.')
        if len({observation.ply for observation in self.observations}) != len(self.observations):
            raise ValueError('Chess search observations must have unique plies.')
        if any(observation.ply > len(self.moves_uci) for observation in self.observations):
            raise ValueError('Chess search observation ply cannot follow the final position.')
        for observation in self.observations:
            terminal_observation = observation.ply == len(self.moves_uci)
            if terminal_observation != (observation.selected_action_id is None):
                raise ValueError('Only a final-position chess observation may omit its selected action.')
            if terminal_observation != (observation.move_selection_mode is ChessMoveSelectionMode.TERMINAL):
                raise ValueError('Only a final-position chess observation may use terminal move selection.')
        if any(
            not self.minimum_model_generation <= observation.model_generation <= self.model_generation
            for observation in self.observations
        ):
            raise ValueError('Chess search observation model generation is outside the game range.')
        return self


class ChessCompletedGamePublisherState(FrozenModel):
    schema_version: Literal[1] = 1
    next_game_number: int = Field(ge=0)


class ChessCompletedGamePublisher:
    def __init__(self, run_path: Path, run_id: int, worker_id: int) -> None:
        if run_id < 0 or worker_id < 0:
            raise ValueError('Run and worker identifiers must be nonnegative.')
        self.inbox_path = run_path / 'completed-games' / 'inbox'
        self.state_path = run_path / 'completed-games' / 'publishers' / f'worker-{worker_id:010d}.json'
        self.run_id = run_id
        self.worker_id = worker_id

    def reserve_identity(self) -> ChessGameIdentity:
        state = self._load_state()
        identity = ChessGameIdentity(
            run_id=self.run_id,
            worker_id=self.worker_id,
            game_number=state.next_game_number,
        )
        next_state = ChessCompletedGamePublisherState(next_game_number=state.next_game_number + 1)
        write_text_atomically(self.state_path, next_state.model_dump_json(indent=2) + '\n')
        return identity

    def publish(self, game: ChessCompletedGame) -> Path:
        if game.identity.run_id != self.run_id or game.identity.worker_id != self.worker_id:
            raise ValueError('Completed-game identity does not belong to this publisher.')
        path = self.inbox_path / game.identity.file_name
        if path.exists():
            existing = ChessCompletedGame.model_validate_json(path.read_text(encoding='utf-8'))
            if existing != game:
                raise ValueError(f'Completed-game file already exists with different content: {path}')
            return path
        write_text_atomically(path, game.model_dump_json() + '\n')
        return path

    def _load_state(self) -> ChessCompletedGamePublisherState:
        if not self.state_path.exists():
            return ChessCompletedGamePublisherState(next_game_number=0)
        return ChessCompletedGamePublisherState.model_validate_json(self.state_path.read_text(encoding='utf-8'))


def completed_game_from_path(path: Path) -> ChessCompletedGame:
    game = ChessCompletedGame.model_validate_json(path.read_text(encoding='utf-8'))
    match = _GAME_FILE_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f'Invalid completed-game file name: {path.name}')
    expected_identity = ChessGameIdentity(
        run_id=int(match.group('run')),
        worker_id=int(match.group('worker')),
        game_number=int(match.group('game')),
    )
    if game.identity != expected_identity:
        raise ValueError(f'Completed-game identity does not match its file name: {path}')
    return game
