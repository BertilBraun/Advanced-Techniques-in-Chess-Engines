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


class EngineSelfPlayDatasetSource(FrozenModel):
    kind: Literal['engine_self_play']
    random_seed: int = Field(ge=0)
    move_sampling_temperature: float = Field(gt=0.0)


class KataGoBookSelectionConfiguration(FrozenModel):
    export_path: str = Field(min_length=1)
    export_sha256: str = Field(min_length=64, max_length=64)
    minimum_ply: int = Field(ge=1)
    maximum_ply: int = Field(gt=1)
    maximum_absolute_black_score: float = Field(gt=0.0)
    maximum_black_win_probability_deviation: float = Field(gt=0.0, le=0.5)
    maximum_win_probability_uncertainty: float = Field(gt=0.0, le=0.5)
    maximum_score_uncertainty: float = Field(gt=0.0)
    minimum_visits: int = Field(gt=0)

    @model_validator(mode='after')
    def validate_ply_range(self) -> KataGoBookSelectionConfiguration:
        if self.maximum_ply < self.minimum_ply:
            raise ValueError('KataGo book maximum ply must not precede minimum ply.')
        return self


class KataGoBookDatasetSource(FrozenModel):
    kind: Literal['katago_book']
    position_count: int = Field(ge=480, le=520)
    selection: KataGoBookSelectionConfiguration


EvaluationDatasetSource: TypeAlias = Annotated[
    EngineSelfPlayDatasetSource | KataGoBookDatasetSource,
    Field(discriminator='kind'),
]


class EvaluationDatasetConfiguration(FrozenModel):
    path: str = Field(min_length=1)
    source: EvaluationDatasetSource


class EngineBeamOpeningSource(FrozenModel):
    kind: Literal['engine_beam']
    random_seed: int = Field(ge=0)
    expanded_actions_per_position: int = Field(default=8, gt=0)
    beam_width: int = Field(default=512, gt=0)


class KataGoBookOpeningSource(FrozenModel):
    kind: Literal['katago_book']
    selection: KataGoBookSelectionConfiguration


class ChessBookOpeningSource(FrozenModel):
    kind: Literal['chess_book']
    selection_path: str = Field(min_length=1)
    selection_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')


OpeningSuiteSource: TypeAlias = Annotated[
    EngineBeamOpeningSource | KataGoBookOpeningSource | ChessBookOpeningSource,
    Field(discriminator='kind'),
]


class OpeningSuiteConfiguration(FrozenModel):
    path: str = Field(min_length=1)
    opening_count: int = Field(default=50, gt=0)
    source: OpeningSuiteSource

    @model_validator(mode='after')
    def validate_beam(self) -> OpeningSuiteConfiguration:
        if self.source.kind == 'engine_beam' and self.source.beam_width < self.opening_count:
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


ExternalEngineConfiguration: TypeAlias = Annotated[
    StockfishEngineConfiguration | KataGoEngineConfiguration,
    Field(discriminator='kind'),
]


class FixedDatasetEvaluationDefinition(FrozenModel):
    kind: Literal['fixed_dataset']
    definition_id: str = Field(min_length=1)


class PairedMatchEvaluationDefinition(FrozenModel):
    opening_pair_count: int = Field(default=50, gt=0)
    maximum_game_plies: int = Field(gt=0)


class RandomOpponentEvaluationDefinition(PairedMatchEvaluationDefinition):
    kind: Literal['random']
    definition_id: str = Field(min_length=1)
    search: EvaluationSearchConfiguration


class PolicyRandomOpponentEvaluationDefinition(PairedMatchEvaluationDefinition):
    kind: Literal['policy_random']
    definition_id: str = Field(min_length=1)


class PreviousCheckpointEvaluationDefinition(PairedMatchEvaluationDefinition):
    kind: Literal['previous_checkpoint']
    definition_id: str = Field(min_length=1)
    boundary_offset: int = Field(default=1, gt=0)
    boundary_parity: Literal['every', 'even', 'odd'] = 'every'
    search: EvaluationSearchConfiguration


class ReferenceCheckpointEvaluationDefinition(PairedMatchEvaluationDefinition):
    kind: Literal['reference_checkpoint']
    definition_id: str = Field(min_length=1)
    manifest_path: str = Field(min_length=1)
    search: EvaluationSearchConfiguration


class StockfishEvaluationDefinition(PairedMatchEvaluationDefinition):
    kind: Literal['stockfish']
    definition_id: str = Field(min_length=1)
    skill_level: int = Field(ge=0, le=20)
    search: EvaluationSearchConfiguration


class StockfishFixedNodesEvaluationDefinition(PairedMatchEvaluationDefinition):
    kind: Literal['stockfish_fixed_nodes']
    definition_id: str = Field(min_length=1)
    nodes: int = Field(gt=0)
    search: EvaluationSearchConfiguration


class KataGoEvaluationDefinition(PairedMatchEvaluationDefinition):
    kind: Literal['katago']
    definition_id: str = Field(min_length=1)
    maximum_visits: int = Field(gt=0)
    search: EvaluationSearchConfiguration


EvaluationDefinition: TypeAlias = Annotated[
    FixedDatasetEvaluationDefinition
    | RandomOpponentEvaluationDefinition
    | PolicyRandomOpponentEvaluationDefinition
    | PreviousCheckpointEvaluationDefinition
    | ReferenceCheckpointEvaluationDefinition
    | StockfishEvaluationDefinition
    | StockfishFixedNodesEvaluationDefinition
    | KataGoEvaluationDefinition,
    Field(discriminator='kind'),
]

MatchEvaluationDefinition: TypeAlias = Annotated[
    RandomOpponentEvaluationDefinition
    | PolicyRandomOpponentEvaluationDefinition
    | PreviousCheckpointEvaluationDefinition
    | ReferenceCheckpointEvaluationDefinition
    | StockfishEvaluationDefinition
    | StockfishFixedNodesEvaluationDefinition
    | KataGoEvaluationDefinition,
    Field(discriminator='kind'),
]


class EvaluationConfiguration(FrozenModel):
    cadence_seconds: int = Field(default=1200, gt=0)
    job_timeout_seconds: float = Field(gt=0.0)
    shutdown_grace_seconds: float = Field(gt=0.0)
    bootstrap_samples: int = Field(gt=0)
    maximum_concurrent_jobs: int = Field(default=10, gt=0)
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
        if self.engine.kind == 'katago' and any(
            definition.kind in ('stockfish', 'stockfish_fixed_nodes') for definition in self.definitions
        ):
            raise ValueError('KataGo evaluation configuration cannot contain a Stockfish opponent.')
        match_definitions = tuple(
            definition for definition in self.definitions if isinstance(definition, PairedMatchEvaluationDefinition)
        )
        if any(definition.opening_pair_count > self.openings.opening_count for definition in match_definitions):
            raise ValueError('Evaluation opening suite must cover every requested opening pair.')
        fixed_node_counts = tuple(
            definition.nodes
            for definition in self.definitions
            if isinstance(definition, StockfishFixedNodesEvaluationDefinition)
        )
        if len(set(fixed_node_counts)) != len(fixed_node_counts):
            raise ValueError('Stockfish fixed-node rungs must use unique node counts.')
        return self
