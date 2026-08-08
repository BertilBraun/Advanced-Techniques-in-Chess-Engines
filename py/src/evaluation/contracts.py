from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import Field

from src.evaluation.configuration import EvaluationDefinition
from src.training.checkpoint import CheckpointReference
from src.util.frozen_model import FrozenModel


class EvaluationDatasetManifest(FrozenModel):
    schema_version: Literal[1] = 1
    game: Literal['chess', 'go']
    rules_digest: str = Field(min_length=64, max_length=64)
    representation_digest: str = Field(min_length=64, max_length=64)
    position_count: int = Field(ge=480, le=520)
    source_game_count: int = Field(gt=0)
    retained_ply_interval: Literal[3] = 3
    retained_ply_offset: int = Field(ge=0, le=2)
    random_seed: int = Field(ge=0)
    move_sampling_temperature: float = Field(gt=0.0)
    engine_identity: str = Field(min_length=1)
    engine_artifact_sha256: tuple[str, ...]
    label_search_limit: int = Field(gt=0)
    maximum_policy_entries: int = Field(gt=0, le=255)
    packed_payload_bytes: int = Field(gt=0)
    row_layout_digest: str = Field(min_length=64, max_length=64)
    data_sha256: str = Field(min_length=64, max_length=64)
    builder_source_revision: str = Field(min_length=1)


class OpeningLine(FrozenModel):
    opening_id: str = Field(min_length=1)
    action_ids: tuple[int, ...] = Field(min_length=4, max_length=4)
    path_probability: float = Field(gt=0.0, le=1.0)
    final_position_digest: str = Field(min_length=64, max_length=64)
    human_readable: str = Field(min_length=1)


class OpeningSuiteManifest(FrozenModel):
    schema_version: Literal[1] = 1
    game: Literal['chess', 'go']
    rules_digest: str = Field(min_length=64, max_length=64)
    representation_digest: str = Field(min_length=64, max_length=64)
    random_seed: int = Field(ge=0)
    engine_identity: str = Field(min_length=1)
    engine_artifact_sha256: tuple[str, ...]
    label_search_limit: int = Field(gt=0)
    expanded_actions_per_position: int = Field(gt=0)
    beam_width: int = Field(gt=0)
    openings: tuple[OpeningLine, ...] = Field(min_length=1)
    builder_source_revision: str = Field(min_length=1)


class RandomOpponent(FrozenModel):
    kind: Literal['random']


class CheckpointOpponent(FrozenModel):
    kind: Literal['checkpoint']
    checkpoint: CheckpointReference


class StockfishOpponent(FrozenModel):
    kind: Literal['stockfish']
    skill_level: int = Field(ge=0, le=20)


class KataGoOpponent(FrozenModel):
    kind: Literal['katago']


EvaluationOpponent: TypeAlias = Annotated[
    RandomOpponent | CheckpointOpponent | StockfishOpponent | KataGoOpponent,
    Field(discriminator='kind'),
]


class FixedDatasetEvaluationJob(FrozenModel):
    kind: Literal['fixed_dataset']
    job_id: str = Field(min_length=1)
    definition: EvaluationDefinition
    boundary_seconds: int = Field(gt=0)
    candidate: CheckpointReference
    device_id: int = Field(ge=0)
    deadline_seconds: float = Field(gt=0.0)
    random_seed: int = Field(ge=0)
    result_path: Path


class MatchEvaluationJob(FrozenModel):
    kind: Literal['match']
    job_id: str = Field(min_length=1)
    definition: EvaluationDefinition
    boundary_seconds: int = Field(gt=0)
    candidate: CheckpointReference
    opponent: EvaluationOpponent
    device_id: int = Field(ge=0)
    deadline_seconds: float = Field(gt=0.0)
    random_seed: int = Field(ge=0)
    result_path: Path


EvaluationJob: TypeAlias = Annotated[
    FixedDatasetEvaluationJob | MatchEvaluationJob,
    Field(discriminator='kind'),
]


class CandidateOutcome(str, Enum):
    WIN = 'win'
    DRAW = 'draw'
    LOSS = 'loss'


class EvaluationTerminationReason(str, Enum):
    NATURAL = 'natural'
    MAXIMUM_PLIES = 'maximum_plies'
    ENGINE_FAILURE = 'engine_failure'


class EvaluationGameResult(FrozenModel):
    game_index: int = Field(ge=0)
    pair_index: int = Field(ge=0)
    opening_id: str = Field(min_length=1)
    candidate_player: Literal['first', 'second']
    pair_seed: int = Field(ge=0)
    initial_action_ids: tuple[int, ...]
    played_action_ids: tuple[int, ...]
    outcome: CandidateOutcome
    termination_reason: EvaluationTerminationReason
    plies: int = Field(ge=0)
    duration_seconds: float = Field(ge=0.0)


class MatchAggregate(FrozenModel):
    wins: int = Field(ge=0)
    draws: int = Field(ge=0)
    losses: int = Field(ge=0)
    score: float = Field(ge=0.0, le=1.0)
    pair_count: int = Field(ge=0)
    score_confidence_low: float = Field(ge=0.0, le=1.0)
    score_confidence_high: float = Field(ge=0.0, le=1.0)


class FixedDatasetEvaluationResult(FrozenModel):
    kind: Literal['fixed_dataset']
    job: FixedDatasetEvaluationJob
    position_count: int = Field(ge=0)
    source_game_count: int = Field(ge=0)
    top_action_accuracy: float = Field(ge=0.0, le=1.0)
    policy_cross_entropy: float = Field(ge=0.0)
    duration_seconds: float = Field(ge=0.0)


class MatchEvaluationResult(FrozenModel):
    kind: Literal['match']
    job: MatchEvaluationJob
    games: tuple[EvaluationGameResult, ...]
    aggregate: MatchAggregate
    duration_seconds: float = Field(ge=0.0)


class EvaluationFailurePhase(str, Enum):
    VALIDATION = 'validation'
    SETUP = 'setup'
    EXECUTION = 'execution'
    DEADLINE = 'deadline'
    CANCELLED = 'cancelled'
    MISSING_ARTIFACT = 'missing_artifact'


class FailedEvaluationResult(FrozenModel):
    kind: Literal['failed']
    job: EvaluationJob
    phase: EvaluationFailurePhase
    message: str = Field(min_length=1)
    exit_code: int | None
    traceback_path: Path | None
    duration_seconds: float = Field(ge=0.0)


EvaluationResult: TypeAlias = Annotated[
    FixedDatasetEvaluationResult | MatchEvaluationResult | FailedEvaluationResult,
    Field(discriminator='kind'),
]
