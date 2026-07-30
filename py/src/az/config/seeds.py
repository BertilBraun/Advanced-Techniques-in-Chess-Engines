from __future__ import annotations

import hashlib
from enum import Enum
from typing import Annotated, Literal

from pydantic import Field

from src.az.config.base import FrozenModel
from src.az.config.canonical import canonical_json


SeedDerivationVersion = Literal['az-seed-v2']
SEED_DERIVATION_VERSION: SeedDerivationVersion = 'az-seed-v2'


class SeedPurpose(str, Enum):
    PROCESS = 'process'
    WORKER = 'worker'
    GAME = 'game'
    SEARCH = 'search'
    DIRICHLET_NOISE = 'dirichlet_noise'
    ACTION_SAMPLING = 'action_sampling'
    SEARCH_BUDGET = 'search_budget'
    REPLAY_SAMPLING = 'replay_sampling'
    MODEL_INITIALIZATION = 'model_initialization'
    TRAINING_RANDOM = 'training_random'
    AUGMENTATION = 'augmentation'
    DATA_LOADER_ORDER = 'data_loader_order'
    EVALUATION_GAME = 'evaluation_game'
    EVALUATION_SEARCH = 'evaluation_search'
    EVALUATION_ACTION = 'evaluation_action'
    SEARCH_TRACE_SAMPLE = 'search_trace_sample'


class ProcessSeedCoordinates(FrozenModel):
    purpose: Literal[SeedPurpose.PROCESS]
    process_index: int = Field(ge=0)


class WorkerSeedCoordinates(FrozenModel):
    purpose: Literal[SeedPurpose.WORKER]
    process_index: int = Field(ge=0)
    worker_index: int = Field(ge=0)


class GameSeedCoordinates(FrozenModel):
    purpose: Literal[SeedPurpose.GAME]
    process_index: int = Field(ge=0)
    worker_index: int = Field(ge=0)
    game_index: int = Field(ge=0)


class SearchSeedCoordinates(FrozenModel):
    purpose: Literal[SeedPurpose.SEARCH]
    process_index: int = Field(ge=0)
    worker_index: int = Field(ge=0)
    game_index: int = Field(ge=0)
    ply: int = Field(ge=0)


class DirichletNoiseSeedCoordinates(FrozenModel):
    purpose: Literal[SeedPurpose.DIRICHLET_NOISE]
    process_index: int = Field(ge=0)
    worker_index: int = Field(ge=0)
    game_index: int = Field(ge=0)
    ply: int = Field(ge=0)


class ActionSamplingSeedCoordinates(FrozenModel):
    purpose: Literal[SeedPurpose.ACTION_SAMPLING]
    process_index: int = Field(ge=0)
    worker_index: int = Field(ge=0)
    game_index: int = Field(ge=0)
    ply: int = Field(ge=0)


class SearchBudgetSeedCoordinates(FrozenModel):
    purpose: Literal[SeedPurpose.SEARCH_BUDGET]
    process_index: int = Field(ge=0)
    worker_index: int = Field(ge=0)
    game_index: int = Field(ge=0)
    ply: int = Field(ge=0)


class ReplaySamplingSeedCoordinates(FrozenModel):
    purpose: Literal[SeedPurpose.REPLAY_SAMPLING]
    trainer_rank: int = Field(ge=0)
    optimizer_step: int = Field(ge=0)


class ModelInitializationSeedCoordinates(FrozenModel):
    purpose: Literal[SeedPurpose.MODEL_INITIALIZATION]
    model_stage: int = Field(ge=0)


class TrainingRandomSeedCoordinates(FrozenModel):
    purpose: Literal[SeedPurpose.TRAINING_RANDOM]
    trainer_rank: int = Field(ge=0)


class AugmentationSeedCoordinates(FrozenModel):
    purpose: Literal[SeedPurpose.AUGMENTATION]
    trainer_rank: int = Field(ge=0)
    optimizer_step: int = Field(ge=0)
    sample_index: int = Field(ge=0)


class DataLoaderOrderSeedCoordinates(FrozenModel):
    purpose: Literal[SeedPurpose.DATA_LOADER_ORDER]
    trainer_rank: int = Field(ge=0)
    epoch_index: int = Field(ge=0)


class EvaluationGameSeedCoordinates(FrozenModel):
    purpose: Literal[SeedPurpose.EVALUATION_GAME]
    evaluation_index: int = Field(ge=0)
    pair_index: int = Field(ge=0)
    game_in_pair: int = Field(ge=0, le=1)


class EvaluationSearchSeedCoordinates(FrozenModel):
    purpose: Literal[SeedPurpose.EVALUATION_SEARCH]
    evaluation_index: int = Field(ge=0)
    pair_index: int = Field(ge=0)
    game_in_pair: int = Field(ge=0, le=1)
    ply: int = Field(ge=0)


class EvaluationActionSeedCoordinates(FrozenModel):
    purpose: Literal[SeedPurpose.EVALUATION_ACTION]
    evaluation_index: int = Field(ge=0)
    pair_index: int = Field(ge=0)
    game_in_pair: int = Field(ge=0, le=1)
    ply: int = Field(ge=0)


class SearchTraceSampleSeedCoordinates(FrozenModel):
    purpose: Literal[SeedPurpose.SEARCH_TRACE_SAMPLE]
    process_index: int = Field(ge=0)
    worker_index: int = Field(ge=0)
    game_index: int = Field(ge=0)
    ply: int = Field(ge=0)


SeedCoordinates = Annotated[
    ProcessSeedCoordinates
    | WorkerSeedCoordinates
    | GameSeedCoordinates
    | SearchSeedCoordinates
    | DirichletNoiseSeedCoordinates
    | ActionSamplingSeedCoordinates
    | SearchBudgetSeedCoordinates
    | ReplaySamplingSeedCoordinates
    | ModelInitializationSeedCoordinates
    | TrainingRandomSeedCoordinates
    | AugmentationSeedCoordinates
    | DataLoaderOrderSeedCoordinates
    | EvaluationGameSeedCoordinates
    | EvaluationSearchSeedCoordinates
    | EvaluationActionSeedCoordinates
    | SearchTraceSampleSeedCoordinates,
    Field(discriminator='purpose'),
]


def derive_seed(root_seed: int, coordinates: SeedCoordinates) -> int:
    if not 0 <= root_seed <= 2**63 - 1:
        raise ValueError('Root seed must be between zero and 2^63 - 1.')
    material = f'{SEED_DERIVATION_VERSION}\0{root_seed}\0{canonical_json(coordinates)}'.encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], byteorder='big') & (2**63 - 1)
