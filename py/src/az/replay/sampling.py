from __future__ import annotations

import random
from dataclasses import dataclass

from pydantic import Field

from src.az.config.base import FrozenModel
from src.az.config.seeds import (
    AugmentationSeedCoordinates,
    ReplaySamplingSeedCoordinates,
    SeedPurpose,
    derive_seed,
)
from src.az.replay.envelope import ReplayRecord
from src.az.replay.storage import IncrementalReplayCatalog


class ReplaySamplerState(FrozenModel):
    next_optimizer_step: int = Field(ge=0)


@dataclass(frozen=True)
class SampledReplay:
    records: tuple[ReplayRecord, ...]
    augmentation_seeds: tuple[int, ...]


class DeterministicReplaySampler:
    def __init__(self, root_seed: int, trainer_rank: int, state: ReplaySamplerState) -> None:
        if trainer_rank < 0:
            raise ValueError('Trainer rank cannot be negative.')
        self._root_seed = root_seed
        self._trainer_rank = trainer_rank
        self._state = state

    @property
    def state(self) -> ReplaySamplerState:
        return self._state

    def sample(self, population: tuple[ReplayRecord, ...], batch_size: int) -> SampledReplay:
        if not population:
            raise ValueError('Cannot sample from an empty replay population.')
        if batch_size <= 0:
            raise ValueError('Replay batch size must be positive.')
        ordered = tuple(sorted(population, key=lambda record: record.envelope.sample_id.bytes))
        if len({record.envelope.sample_id for record in ordered}) != len(ordered):
            raise ValueError('Replay sample identities must be unique.')
        optimizer_step = self._state.next_optimizer_step
        sampling_seed = derive_seed(
            self._root_seed,
            ReplaySamplingSeedCoordinates(
                purpose=SeedPurpose.REPLAY_SAMPLING,
                trainer_rank=self._trainer_rank,
                optimizer_step=optimizer_step,
            ),
        )
        generator = random.Random(sampling_seed)
        selected = tuple(ordered[generator.randrange(len(ordered))] for _ in range(batch_size))
        augmentation_seeds = tuple(
            derive_seed(
                self._root_seed,
                AugmentationSeedCoordinates(
                    purpose=SeedPurpose.AUGMENTATION,
                    trainer_rank=self._trainer_rank,
                    optimizer_step=optimizer_step,
                    sample_index=sample_index,
                ),
            )
            for sample_index in range(batch_size)
        )
        self._state = ReplaySamplerState(next_optimizer_step=optimizer_step + 1)
        return SampledReplay(records=selected, augmentation_seeds=augmentation_seeds)

    def sample_catalog(
        self,
        catalog: IncrementalReplayCatalog,
        batch_size: int,
    ) -> SampledReplay:
        snapshot = catalog.snapshot
        if snapshot.position_count == 0:
            raise ValueError('Cannot sample from an empty replay catalog.')
        if batch_size <= 0:
            raise ValueError('Replay batch size must be positive.')
        optimizer_step = self._state.next_optimizer_step
        sampling_seed = derive_seed(
            self._root_seed,
            ReplaySamplingSeedCoordinates(
                purpose=SeedPurpose.REPLAY_SAMPLING,
                trainer_rank=self._trainer_rank,
                optimizer_step=optimizer_step,
            ),
        )
        generator = random.Random(sampling_seed)
        selected_locations = tuple(
            snapshot.location(generator.randrange(snapshot.position_count)) for _ in range(batch_size)
        )
        augmentation_seeds = tuple(
            derive_seed(
                self._root_seed,
                AugmentationSeedCoordinates(
                    purpose=SeedPurpose.AUGMENTATION,
                    trainer_rank=self._trainer_rank,
                    optimizer_step=optimizer_step,
                    sample_index=sample_index,
                ),
            )
            for sample_index in range(batch_size)
        )
        self._state = ReplaySamplerState(next_optimizer_step=optimizer_step + 1)
        return SampledReplay(
            records=catalog.read(selected_locations),
            augmentation_seeds=augmentation_seeds,
        )
