from __future__ import annotations

import re
from enum import Enum
from math import isclose, isfinite
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from uuid import UUID

from pydantic import ConfigDict, Field

if TYPE_CHECKING:
    from AlphaZeroCpp import GameSearchVisit
from src.games.contracts import WdlTarget
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel


class SearchVisitCounts(FrozenModel):
    action_ids: tuple[int, ...] = Field(min_length=1)
    visit_counts: tuple[int, ...] = Field(min_length=1)

    def model_post_init(self, __context: object) -> None:
        if len(self.action_ids) != len(self.visit_counts):
            raise ValueError('Search action IDs and visit counts must have equal lengths.')
        if len(set(self.action_ids)) != len(self.action_ids) or any(action_id < 0 for action_id in self.action_ids):
            raise ValueError('Search action IDs must be unique and nonnegative.')
        if any(visit_count <= 0 for visit_count in self.visit_counts):
            raise ValueError('Search visit counts must be positive.')

    @classmethod
    def from_native(cls, visits: tuple[GameSearchVisit, ...]) -> SearchVisitCounts:
        return cls(
            action_ids=tuple(visit.action_id for visit in visits),
            visit_counts=tuple(visit.visit_count for visit in visits),
        )

    @classmethod
    def from_columns(cls, columns: tuple[list[int], list[int]]) -> SearchVisitCounts:
        action_ids, visit_counts = columns
        return cls(action_ids=tuple(action_ids), visit_counts=tuple(visit_counts))

    def native_visits(self) -> tuple[GameSearchVisit, ...]:
        # Imported here so the on-disk game schema stays importable without the native extension.
        from AlphaZeroCpp import GameSearchVisit

        return tuple(
            GameSearchVisit(action_id=action_id, visit_count=visit_count)
            for action_id, visit_count in zip(self.action_ids, self.visit_counts, strict=True)
        )


class GameIdentity(FrozenModel):
    worker_id: int = Field(ge=0)
    process_instance_id: UUID
    game_number: int = Field(ge=0)

    @classmethod
    def from_file_name(cls, file_name: str) -> GameIdentity:
        match = re.fullmatch(
            r'worker-(\d+)-process-([0-9a-fA-F-]{36})-game-(\d{20})\.json',
            file_name,
        )
        if match is None:
            raise ValueError(f'Completed-game file name is invalid: {file_name}')
        identity = cls(worker_id=int(match[1]), process_instance_id=UUID(match[2]), game_number=int(match[3]))
        if identity.file_name != file_name:
            raise ValueError(f'Completed-game file name is not canonical: {file_name}')
        return identity

    @property
    def file_name(self) -> str:
        return f'worker-{self.worker_id}-process-{self.process_instance_id}-game-{self.game_number:020d}.json'

    @property
    def archive_key(self) -> str:
        return f'{self.worker_id}:{self.process_instance_id}:{self.game_number}'


class TerminationReason(str, Enum):
    NATURAL = 'natural'
    MAXIMUM_PLIES = 'maximum_plies'
    RESIGNATION = 'resignation'
    ADJUDICATION = 'adjudication'


class SearchStopReason(str, Enum):
    FIXED_LIMIT = 'fixed_limit'
    DETERMINISTIC = 'deterministic'
    LEARNED_GATE = 'learned_gate'
    MAXIMUM = 'maximum'


class SearchCheckpointObservation(FrozenModel):
    visits: int = Field(gt=0)
    leader_action_id: int = Field(ge=0)
    most_visited_action_id: int = Field(ge=0)
    top_visit_share: float = Field(ge=0.0, le=1.0)
    top_two_margin: float = Field(ge=0.0, le=1.0)
    root_value: float = Field(ge=-1.0, le=1.0)
    root_value_delta: float = Field(ge=0.0, le=2.0)
    leader_stable: bool


class SearchObservation(FrozenModel):
    model_config = ConfigDict(frozen=True, extra='forbid', arbitrary_types_allowed=True)

    ply: int = Field(ge=0)
    model_generation: int = Field(ge=0)
    policy_target_visits: SearchVisitCounts
    root_value: float
    highest_visited_child_action_id: int = Field(ge=0)
    highest_visited_child_visit_count: int = Field(gt=0)
    highest_visited_child_q: float
    selected_action_id: int | None = Field(default=None, ge=0)
    full_search: bool
    sample_weight: float = Field(gt=0.0)
    search_budget: int = Field(gt=0)
    network_root_value: float = Field(ge=-1.0, le=1.0)
    policy_correction: float = Field(ge=0.0, le=1.0)
    value_correction: float = Field(ge=0.0, le=1.0)
    search_correction_target: float = Field(ge=0.0, le=1.0)
    predicted_search_correction: float = Field(ge=0.0, le=1.0)
    starting_visits: int = Field(ge=0)
    final_visits: int = Field(gt=0)
    stop_reason: SearchStopReason
    learned_gate_evaluated: bool
    checkpoints: tuple[SearchCheckpointObservation, ...] = ()

    def model_post_init(self, __context: object, /) -> None:
        if not isfinite(self.root_value) or not -1.0 <= self.root_value <= 1.0:
            raise ValueError('Search root value must be finite and lie in [-1, 1].')
        if not isfinite(self.highest_visited_child_q) or not -1.0 <= self.highest_visited_child_q <= 1.0:
            raise ValueError('Highest-visited child Q must be finite and lie in [-1, 1].')
        if self.final_visits < self.starting_visits:
            raise ValueError('Final root visits cannot precede retained starting visits.')
        if not isclose(
            self.search_correction_target,
            max(self.policy_correction, self.value_correction),
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError('Search-correction target must equal the larger recorded correction.')
        checkpoint_visits = tuple(checkpoint.visits for checkpoint in self.checkpoints)
        if checkpoint_visits != tuple(sorted(set(checkpoint_visits))):
            raise ValueError('Adaptive search checkpoints must have unique increasing visit counts.')


class CompletedSelfPlayGame(FrozenModel):
    schema_version: Literal[6] = 6
    identity: GameIdentity
    created_at_seconds: float = Field(ge=0.0)
    generation_seconds: float = Field(ge=0.0)
    action_ids: tuple[int, ...]
    observations: tuple[SearchObservation, ...]
    final_wdl: WdlTarget
    termination_reason: TerminationReason
    is_resignation_continuation: bool = False
    resignation_threshold: float | None = Field(default=None, ge=-1.0, lt=0.0)

    def model_post_init(self, __context: object) -> None:
        plies = tuple(observation.ply for observation in self.observations)
        if plies != tuple(sorted(set(plies))):
            raise ValueError('Search observations must use unique ordered plies.')
        trailing = tuple(observation for observation in self.observations if observation.ply == len(self.action_ids))
        if any(observation.ply > len(self.action_ids) for observation in self.observations):
            raise ValueError('Search observations cannot follow the final game position.')
        if trailing and (self.termination_reason is not TerminationReason.RESIGNATION or len(trailing) != 1):
            raise ValueError('Only resignation may retain one unplayed final search observation.')
        if any(
            observation.selected_action_id != self.action_ids[observation.ply]
            for observation in self.observations
            if observation.ply < len(self.action_ids)
        ):
            raise ValueError('Observed selected actions must agree with the played trajectory.')
        if trailing and trailing[0].selected_action_id is not None:
            raise ValueError('A resignation observation cannot select an action.')
        if self.termination_reason is TerminationReason.RESIGNATION and self.is_resignation_continuation:
            raise ValueError('Continuation games cannot terminate by resignation.')


def publish_completed_self_play_game(inbox_path: Path, game: CompletedSelfPlayGame) -> Path:
    path = inbox_path / game.identity.file_name
    if path.exists():
        raise ValueError(f'Completed-game identity already exists: {path}')
    write_text_atomically(path, game.model_dump_json() + '\n')
    return path
