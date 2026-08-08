from __future__ import annotations

from pathlib import Path
from enum import Enum
from math import isfinite
from typing import Literal
from uuid import UUID

from pydantic import Field

from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel
from src.games.contracts import WdlTarget


class GameIdentity(FrozenModel):
    worker_id: int = Field(ge=0)
    process_instance_id: UUID
    game_number: int = Field(ge=0)

    @property
    def file_name(self) -> str:
        return f'worker-{self.worker_id}-process-{self.process_instance_id}-game-{self.game_number:020d}.json'

    @property
    def archive_key(self) -> str:
        return f'{self.worker_id}:{self.process_instance_id}:{self.game_number}'


class SparseSearchVisit(FrozenModel):
    action_id: int = Field(ge=0)
    visit_count: int = Field(gt=0)


class TerminationReason(str, Enum):
    NATURAL = 'natural'
    MAXIMUM_PLIES = 'maximum_plies'
    RESIGNATION = 'resignation'
    ADJUDICATION = 'adjudication'


class SearchObservation(FrozenModel):
    ply: int = Field(ge=0)
    model_generation: int = Field(ge=0)
    visits: tuple[SparseSearchVisit, ...] = Field(min_length=1)
    root_value: float
    selected_action_id: int = Field(ge=0)
    full_search: bool
    sample_weight: float = Field(gt=0.0)
    search_budget: int = Field(gt=0)
    minimum_root_visits: int = Field(ge=0)

    def model_post_init(self, __context: object) -> None:
        if not isfinite(self.root_value) or not -1.0 <= self.root_value <= 1.0:
            raise ValueError('Search root value must be finite and lie in [-1, 1].')
        action_ids = tuple(visit.action_id for visit in self.visits)
        if len(set(action_ids)) != len(action_ids):
            raise ValueError('Search visits must use unique action IDs.')
        if sum(visit.visit_count for visit in self.visits) <= 0:
            raise ValueError('Search visits must contain positive total visits.')


class CompletedSelfPlayGame(FrozenModel):
    schema_version: Literal[1] = 1
    identity: GameIdentity
    created_at_seconds: float = Field(ge=0.0)
    generation_seconds: float = Field(ge=0.0)
    action_ids: tuple[int, ...]
    observations: tuple[SearchObservation, ...]
    final_wdl: WdlTarget
    termination_reason: TerminationReason

    def model_post_init(self, __context: object) -> None:
        plies = tuple(observation.ply for observation in self.observations)
        if plies != tuple(sorted(set(plies))):
            raise ValueError('Search observations must use unique ordered plies.')
        if any(observation.ply >= len(self.action_ids) for observation in self.observations):
            raise ValueError('Every search observation must precede a played action.')
        if any(observation.selected_action_id != self.action_ids[observation.ply] for observation in self.observations):
            raise ValueError('Observed selected actions must agree with the played trajectory.')


def publish_completed_self_play_game(inbox_path: Path, game: CompletedSelfPlayGame) -> Path:
    path = inbox_path / game.identity.file_name
    if path.exists():
        raise ValueError(f'Completed-game identity already exists: {path}')
    write_text_atomically(path, game.model_dump_json() + '\n')
    return path
