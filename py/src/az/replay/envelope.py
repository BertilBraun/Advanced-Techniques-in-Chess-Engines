from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field, PositiveInt, field_validator, model_validator

from src.az.config.base import FrozenModel
from src.az.config.seeds import (
    ActionSamplingSeedCoordinates,
    DirichletNoiseSeedCoordinates,
    GameSeedCoordinates,
    ProcessSeedCoordinates,
    SearchSeedCoordinates,
    SearchBudgetSeedCoordinates,
    SeedDerivationVersion,
    SeedPurpose,
    WorkerSeedCoordinates,
    derive_seed,
)
from src.az.games.api import GameIdentifier


NATIVE_INT32_MAXIMUM = 2**31 - 1


class SearchStrategy(str, Enum):
    FIXED = 'fixed'
    PROGRESSIVE = 'progressive'
    MIXED = 'mixed'
    ADAPTIVE = 'adaptive'


class SearchBudgetClass(str, Enum):
    FIXED = 'fixed'
    PROGRESSIVE_STAGE = 'progressive_stage'
    MIXED_FAST = 'mixed_fast'
    MIXED_FULL = 'mixed_full'


class SearchStopReason(str, Enum):
    FULL_BUDGET = 'full_budget'
    ADAPTIVE_CONFIDENCE = 'adaptive_confidence'
    TERMINAL_ROOT = 'terminal_root'


class NoSearchCalibration(FrozenModel):
    kind: Literal['none']


class VisitMarginSearchCalibration(FrozenModel):
    kind: Literal['visit_margin']
    calibration_id: str = Field(min_length=1)


SearchCalibrationEvidence = Annotated[
    NoSearchCalibration | VisitMarginSearchCalibration,
    Field(discriminator='kind'),
]


class GameTermination(str, Enum):
    TWO_CONSECUTIVE_PASSES = 'two_consecutive_passes'
    RESIGNATION = 'resignation'
    SAFETY_PLY_CAP = 'safety_ply_cap'


UnitInterval = Annotated[float, Field(ge=0, le=1)]
NonnegativeFinite = Annotated[float, Field(ge=0)]


class SelfPlaySeedLineage(FrozenModel):
    derivation_version: SeedDerivationVersion
    root_seed: int = Field(ge=0, le=2**63 - 1)
    process_index: int = Field(ge=0)
    worker_index: int = Field(ge=0)
    game_index: int = Field(ge=0)
    ply: int = Field(ge=0)
    process_seed: int = Field(ge=0, le=2**63 - 1)
    worker_seed: int = Field(ge=0, le=2**63 - 1)
    game_seed: int = Field(ge=0, le=2**63 - 1)
    search_seed: int = Field(ge=0, le=2**63 - 1)
    root_noise_seed: int = Field(ge=0, le=2**63 - 1)
    action_sampling_seed: int = Field(ge=0, le=2**63 - 1)
    search_budget_seed: int = Field(ge=0, le=2**63 - 1)

    @model_validator(mode='after')
    def validate_derivation(self) -> SelfPlaySeedLineage:
        expected = (
            derive_seed(
                self.root_seed,
                ProcessSeedCoordinates(purpose=SeedPurpose.PROCESS, process_index=self.process_index),
            ),
            derive_seed(
                self.root_seed,
                WorkerSeedCoordinates(
                    purpose=SeedPurpose.WORKER,
                    process_index=self.process_index,
                    worker_index=self.worker_index,
                ),
            ),
            derive_seed(
                self.root_seed,
                GameSeedCoordinates(
                    purpose=SeedPurpose.GAME,
                    process_index=self.process_index,
                    worker_index=self.worker_index,
                    game_index=self.game_index,
                ),
            ),
            derive_seed(
                self.root_seed,
                SearchSeedCoordinates(
                    purpose=SeedPurpose.SEARCH,
                    process_index=self.process_index,
                    worker_index=self.worker_index,
                    game_index=self.game_index,
                    ply=self.ply,
                ),
            ),
            derive_seed(
                self.root_seed,
                DirichletNoiseSeedCoordinates(
                    purpose=SeedPurpose.DIRICHLET_NOISE,
                    process_index=self.process_index,
                    worker_index=self.worker_index,
                    game_index=self.game_index,
                    ply=self.ply,
                ),
            ),
            derive_seed(
                self.root_seed,
                ActionSamplingSeedCoordinates(
                    purpose=SeedPurpose.ACTION_SAMPLING,
                    process_index=self.process_index,
                    worker_index=self.worker_index,
                    game_index=self.game_index,
                    ply=self.ply,
                ),
            ),
            derive_seed(
                self.root_seed,
                SearchBudgetSeedCoordinates(
                    purpose=SeedPurpose.SEARCH_BUDGET,
                    process_index=self.process_index,
                    worker_index=self.worker_index,
                    game_index=self.game_index,
                    ply=self.ply,
                ),
            ),
        )
        actual = (
            self.process_seed,
            self.worker_seed,
            self.game_seed,
            self.search_seed,
            self.root_noise_seed,
            self.action_sampling_seed,
            self.search_budget_seed,
        )
        if actual != expected:
            raise ValueError('Replay self-play seed lineage does not match the stable derivations.')
        return self


def derive_self_play_seed_lineage(
    root_seed: int,
    process_index: int,
    worker_index: int,
    game_index: int,
    ply: int,
) -> SelfPlaySeedLineage:
    process_coordinates = ProcessSeedCoordinates(
        purpose=SeedPurpose.PROCESS,
        process_index=process_index,
    )
    worker_coordinates = WorkerSeedCoordinates(
        purpose=SeedPurpose.WORKER,
        process_index=process_index,
        worker_index=worker_index,
    )
    game_coordinates = GameSeedCoordinates(
        purpose=SeedPurpose.GAME,
        process_index=process_index,
        worker_index=worker_index,
        game_index=game_index,
    )
    search_coordinates = SearchSeedCoordinates(
        purpose=SeedPurpose.SEARCH,
        process_index=process_index,
        worker_index=worker_index,
        game_index=game_index,
        ply=ply,
    )
    root_noise_coordinates = DirichletNoiseSeedCoordinates(
        purpose=SeedPurpose.DIRICHLET_NOISE,
        process_index=process_index,
        worker_index=worker_index,
        game_index=game_index,
        ply=ply,
    )
    action_sampling_coordinates = ActionSamplingSeedCoordinates(
        purpose=SeedPurpose.ACTION_SAMPLING,
        process_index=process_index,
        worker_index=worker_index,
        game_index=game_index,
        ply=ply,
    )
    search_budget_coordinates = SearchBudgetSeedCoordinates(
        purpose=SeedPurpose.SEARCH_BUDGET,
        process_index=process_index,
        worker_index=worker_index,
        game_index=game_index,
        ply=ply,
    )
    return SelfPlaySeedLineage(
        derivation_version='az-seed-v2',
        root_seed=root_seed,
        process_index=process_index,
        worker_index=worker_index,
        game_index=game_index,
        ply=ply,
        process_seed=derive_seed(root_seed, process_coordinates),
        worker_seed=derive_seed(root_seed, worker_coordinates),
        game_seed=derive_seed(root_seed, game_coordinates),
        search_seed=derive_seed(root_seed, search_coordinates),
        root_noise_seed=derive_seed(root_seed, root_noise_coordinates),
        action_sampling_seed=derive_seed(root_seed, action_sampling_coordinates),
        search_budget_seed=derive_seed(root_seed, search_budget_coordinates),
    )


class RootDiagnostics(FrozenModel):
    visit_count: int = Field(ge=0, le=NATIVE_INT32_MAXIMUM)
    entropy: NonnegativeFinite
    top_two_margin: UnitInterval
    prefix_full_policy_disagreement: UnitInterval | None
    prefix_full_value_disagreement: NonnegativeFinite | None


class ReplayEnvelope(FrozenModel):
    run_id: UUID
    game_identifier: GameIdentifier
    payload_schema_version: PositiveInt
    sample_id: UUID
    game_id: UUID
    seed_lineage: SelfPlaySeedLineage
    created_at: datetime
    ply: int = Field(ge=0)
    checkpoint_id: str = Field(min_length=1)
    search_strategy: SearchStrategy
    budget_class: SearchBudgetClass
    configured_simulation_cap: int = Field(gt=0, le=NATIVE_INT32_MAXIMUM)
    actual_simulations: int = Field(ge=0, le=NATIVE_INT32_MAXIMUM)
    stop_reason: SearchStopReason
    policy_target_eligible: bool
    policy_target_weight: NonnegativeFinite
    value_target_eligible: bool
    value_target_weight: NonnegativeFinite
    root_diagnostics: RootDiagnostics
    termination: GameTermination
    replay_credit_id: UUID
    search_calibration: SearchCalibrationEvidence

    @field_validator('created_at')
    @classmethod
    def validate_utc_timestamp(cls, created_at: datetime) -> datetime:
        if created_at.tzinfo is None or created_at.utcoffset() != timezone.utc.utcoffset(created_at):
            raise ValueError('Replay creation time must be timezone-aware UTC.')
        return created_at

    @model_validator(mode='after')
    def validate_search_accounting(self) -> ReplayEnvelope:
        if self.actual_simulations > self.configured_simulation_cap:
            raise ValueError('Actual simulations cannot exceed the configured cap.')
        if self.seed_lineage.ply != self.ply:
            raise ValueError('Replay envelope ply must match its self-play seed lineage.')
        if self.policy_target_eligible != (self.policy_target_weight > 0):
            raise ValueError('Policy eligibility must exactly match positive policy weight.')
        if self.value_target_eligible != (self.value_target_weight > 0):
            raise ValueError('Value eligibility must exactly match positive value weight.')
        if self.termination is GameTermination.SAFETY_PLY_CAP and self.value_target_eligible:
            raise ValueError('Safety-ply-capped games cannot contribute value targets.')
        if self.root_diagnostics.visit_count != self.actual_simulations:
            raise ValueError('Root visit count must equal actual simulations.')
        match self.search_strategy:
            case SearchStrategy.FIXED if self.budget_class is not SearchBudgetClass.FIXED:
                raise ValueError('Fixed search requires the fixed budget class.')
            case SearchStrategy.PROGRESSIVE if self.budget_class is not SearchBudgetClass.PROGRESSIVE_STAGE:
                raise ValueError('Progressive search requires a progressive-stage budget class.')
            case SearchStrategy.MIXED if self.budget_class not in (
                SearchBudgetClass.MIXED_FAST,
                SearchBudgetClass.MIXED_FULL,
            ):
                raise ValueError('Mixed search requires a mixed fast or full budget class.')
            case SearchStrategy.ADAPTIVE if self.budget_class not in (
                SearchBudgetClass.FIXED,
                SearchBudgetClass.PROGRESSIVE_STAGE,
            ):
                raise ValueError('Adaptive stopping requires a fixed or progressive budget class.')
            case _:
                pass
        match self.search_strategy, self.search_calibration:
            case SearchStrategy.ADAPTIVE, VisitMarginSearchCalibration():
                pass
            case SearchStrategy.ADAPTIVE, NoSearchCalibration():
                raise ValueError('Adaptive search requires visit-margin calibration evidence.')
            case _, NoSearchCalibration():
                pass
            case _, VisitMarginSearchCalibration():
                raise ValueError('Visit-margin calibration evidence requires adaptive search.')
        match self.stop_reason:
            case SearchStopReason.FULL_BUDGET if self.actual_simulations != self.configured_simulation_cap:
                raise ValueError('Full-budget search must consume its configured cap.')
            case SearchStopReason.TERMINAL_ROOT if (
                self.actual_simulations != 0 or self.root_diagnostics.visit_count != 0 or self.policy_target_eligible
            ):
                raise ValueError('Terminal-root search cannot simulate or contribute a policy target.')
            case SearchStopReason.ADAPTIVE_CONFIDENCE if self.search_strategy is not SearchStrategy.ADAPTIVE:
                raise ValueError('Adaptive-confidence stopping requires adaptive search.')
            case SearchStopReason.ADAPTIVE_CONFIDENCE if self.actual_simulations >= self.configured_simulation_cap:
                raise ValueError('Adaptive-confidence stopping must occur before the configured cap.')
            case _:
                pass
        return self


@dataclass(frozen=True)
class ReplayRecord:
    envelope: ReplayEnvelope
    payload: bytes
