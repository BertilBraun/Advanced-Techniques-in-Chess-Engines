from __future__ import annotations

from enum import Enum
from math import isfinite
from typing import Literal

from pydantic import Field, model_validator

from src.self_play.value_target import TerminationReason
from src.self_play.completed_game import (
    CompletedGameRecord,
    GameIdentity,
    SparseSearchVisit,
)
from src.util.frozen_model import FrozenModel


CHESS_COMPLETED_GAME_SCHEMA_VERSION = 1
CHESS_REPRESENTATION_VERSION = 1


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


class ChessCompletedGame(CompletedGameRecord):
    schema_version: Literal[1] = CHESS_COMPLETED_GAME_SCHEMA_VERSION
    game: Literal['chess'] = 'chess'
    identity: GameIdentity
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
