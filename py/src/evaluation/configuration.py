from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import Field, model_validator

from src.self_play.configuration import BatchedInferenceParams
from src.util.frozen_model import FrozenModel


class EvaluationSearchConfiguration(FrozenModel):
    searches_per_move: int = Field(gt=0)
    parallel_searches: int = Field(gt=0)
    exploration_constant: float = Field(gt=0.0)
    inference: BatchedInferenceParams

    @model_validator(mode='after')
    def validate_searches(self) -> EvaluationSearchConfiguration:
        if self.searches_per_move <= self.parallel_searches:
            raise ValueError('Evaluation searches per move must exceed parallel searches.')
        return self


class EvaluationDatasetConfiguration(FrozenModel):
    path: str = Field(min_length=1)
    random_seed: int = Field(ge=0)
    move_sampling_temperature: float = Field(gt=0.0)


class OpeningSuiteConfiguration(FrozenModel):
    path: str = Field(min_length=1)
    random_seed: int = Field(ge=0)
    opening_count: int = Field(default=50, gt=0)
    expanded_actions_per_position: int = Field(default=8, gt=0)
    beam_width: int = Field(default=512, gt=0)

    @model_validator(mode='after')
    def validate_beam(self) -> OpeningSuiteConfiguration:
        if self.beam_width < self.opening_count:
            raise ValueError('Opening beam width must cover every requested opening.')
        return self


class StockfishEngineConfiguration(FrozenModel):
    kind: Literal['stockfish']
    executable_path: str = Field(min_length=1)
    label_nodes: int = Field(gt=0)
    match_nodes: int = Field(gt=0)
    threads: int = Field(gt=0)
    hash_mib: int = Field(gt=0)
    multi_pv: int = Field(gt=1)
    policy_softmax_temperature: float = Field(gt=0.0)


class KataGoEngineConfiguration(FrozenModel):
    kind: Literal['katago']
    executable_path: str = Field(min_length=1)
    model_path: str = Field(min_length=1)
    analysis_configuration_path: str = Field(min_length=1)
    label_max_visits: int = Field(gt=0)
    match_max_visits: int = Field(gt=0)


ExternalEngineConfiguration: TypeAlias = Annotated[
    StockfishEngineConfiguration | KataGoEngineConfiguration,
    Field(discriminator='kind'),
]


class FixedDatasetEvaluationDefinition(FrozenModel):
    kind: Literal['fixed_dataset']
    definition_id: str = Field(min_length=1)


class RandomOpponentEvaluationDefinition(FrozenModel):
    kind: Literal['random']
    definition_id: str = Field(min_length=1)
    search: EvaluationSearchConfiguration
    maximum_game_plies: int = Field(gt=0)


class PolicyRandomOpponentEvaluationDefinition(FrozenModel):
    kind: Literal['policy_random']
    definition_id: str = Field(min_length=1)
    maximum_game_plies: int = Field(gt=0)


class PreviousCheckpointEvaluationDefinition(FrozenModel):
    kind: Literal['previous_checkpoint']
    definition_id: str = Field(min_length=1)
    boundary_offset: int = Field(default=1, gt=0)
    search: EvaluationSearchConfiguration
    maximum_game_plies: int = Field(gt=0)


class FixedCheckpointEvaluationDefinition(FrozenModel):
    kind: Literal['fixed_checkpoint']
    definition_id: str = Field(min_length=1)
    generation: int = Field(ge=0)
    search: EvaluationSearchConfiguration
    maximum_game_plies: int = Field(gt=0)


class StockfishEvaluationDefinition(FrozenModel):
    kind: Literal['stockfish']
    definition_id: str = Field(min_length=1)
    skill_level: int = Field(ge=0, le=20)
    search: EvaluationSearchConfiguration
    maximum_game_plies: int = Field(gt=0)


class KataGoEvaluationDefinition(FrozenModel):
    kind: Literal['katago']
    definition_id: str = Field(min_length=1)
    search: EvaluationSearchConfiguration
    maximum_game_plies: int = Field(gt=0)


EvaluationDefinition: TypeAlias = Annotated[
    FixedDatasetEvaluationDefinition
    | RandomOpponentEvaluationDefinition
    | PolicyRandomOpponentEvaluationDefinition
    | PreviousCheckpointEvaluationDefinition
    | FixedCheckpointEvaluationDefinition
    | StockfishEvaluationDefinition
    | KataGoEvaluationDefinition,
    Field(discriminator='kind'),
]

MatchEvaluationDefinition: TypeAlias = Annotated[
    RandomOpponentEvaluationDefinition
    | PolicyRandomOpponentEvaluationDefinition
    | PreviousCheckpointEvaluationDefinition
    | FixedCheckpointEvaluationDefinition
    | StockfishEvaluationDefinition
    | KataGoEvaluationDefinition,
    Field(discriminator='kind'),
]


class EvaluationConfiguration(FrozenModel):
    cadence_seconds: int = Field(default=1200, gt=0)
    job_timeout_seconds: float = Field(gt=0.0)
    shutdown_grace_seconds: float = Field(gt=0.0)
    bootstrap_samples: int = Field(gt=0)
    dataset: EvaluationDatasetConfiguration
    openings: OpeningSuiteConfiguration
    engine: ExternalEngineConfiguration
    definitions: tuple[EvaluationDefinition, ...] = Field(min_length=1)

    @model_validator(mode='after')
    def validate_definitions(self) -> EvaluationConfiguration:
        definition_ids = tuple(definition.definition_id for definition in self.definitions)
        if len(set(definition_ids)) != len(definition_ids):
            raise ValueError('Evaluation definition IDs must be unique.')
        if not any(definition.kind == 'fixed_dataset' for definition in self.definitions):
            raise ValueError('Evaluation must contain one fixed-dataset definition.')
        if self.engine.kind == 'stockfish' and any(definition.kind == 'katago' for definition in self.definitions):
            raise ValueError('Stockfish evaluation configuration cannot contain a KataGo opponent.')
        if self.engine.kind == 'katago' and any(definition.kind == 'stockfish' for definition in self.definitions):
            raise ValueError('KataGo evaluation configuration cannot contain a Stockfish opponent.')
        return self
