from __future__ import annotations

from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field, model_validator

from src.az.config.base import FrozenModel, Sha256
from src.az.config.serialization import model_sha256
from src.az.evaluation.statistics import LearningCurveStatistics, MatchStatistics
from src.az.evaluation.models import EvaluationOpponentIdentity


class AvailableMetricEvidence(FrozenModel):
    kind: Literal['available']
    value: float
    unit: str = Field(min_length=1)
    source: str = Field(min_length=1)


class UnavailableMetricEvidence(FrozenModel):
    kind: Literal['unavailable']
    reason: str = Field(min_length=1)


MetricEvidence = Annotated[
    AvailableMetricEvidence | UnavailableMetricEvidence,
    Field(discriminator='kind'),
]


class CategoryCount(FrozenModel):
    category: str = Field(min_length=1)
    count: int = Field(ge=0)


class AvailableDistributionEvidence(FrozenModel):
    kind: Literal['available']
    values: tuple[CategoryCount, ...]
    source: str = Field(min_length=1)


class UnavailableDistributionEvidence(FrozenModel):
    kind: Literal['unavailable']
    reason: str = Field(min_length=1)


DistributionEvidence = Annotated[
    AvailableDistributionEvidence | UnavailableDistributionEvidence,
    Field(discriminator='kind'),
]


class PrefixDisagreementAtCheckpoint(FrozenModel):
    simulations: int = Field(gt=0)
    observation_count: int = Field(gt=0)
    mean_policy_total_variation: float = Field(ge=0, le=1)
    mean_value_absolute_error: float = Field(ge=0, le=2)


class AvailablePrefixDisagreement(FrozenModel):
    kind: Literal['available']
    checkpoints: tuple[PrefixDisagreementAtCheckpoint, ...] = Field(min_length=1)
    source: str = Field(min_length=1)


class UnavailablePrefixDisagreement(FrozenModel):
    kind: Literal['unavailable']
    reason: str = Field(min_length=1)


PrefixDisagreementEvidence = Annotated[
    AvailablePrefixDisagreement | UnavailablePrefixDisagreement,
    Field(discriminator='kind'),
]


class CheckpointTimingEvidence(FrozenModel):
    requested_elapsed_seconds: int = Field(gt=0)
    published_elapsed_seconds: float = Field(ge=0)
    checkpoint_id: UUID
    model_artifact_sha256: Sha256


class RunDiagnostics(FrozenModel):
    committed_games: MetricEvidence
    committed_positions: MetricEvidence
    actual_simulations: MetricEvidence
    mean_actual_simulations_per_move: MetricEvidence
    p50_actual_simulations_per_move: MetricEvidence
    p95_actual_simulations_per_move: MetricEvidence
    budget_class_distribution: DistributionEvidence
    policy_eligible_positions: MetricEvidence
    policy_eligible_fraction: MetricEvidence
    policy_weight_sum: MetricEvidence
    gpu_utilization: MetricEvidence
    optimizer_steps: MetricEvidence
    replay_reuse: MetricEvidence
    stop_reason_distribution: DistributionEvidence
    adaptive_early_stop_frequency: MetricEvidence
    prefix_full_disagreement: PrefixDisagreementEvidence
    evaluation_games: MetricEvidence
    evaluation_wall_seconds: MetricEvidence
    evaluation_actual_simulations: MetricEvidence
    checkpoint_timing: tuple[CheckpointTimingEvidence, ...]


class RunIdentity(FrozenModel):
    run_id: UUID
    arm_id: UUID
    seed: int = Field(ge=0, le=2**63 - 1)
    resolved_configuration_sha256: Sha256
    source_revision: str = Field(min_length=1)
    hardware_identity: str = Field(min_length=1)


class EvaluationProtocolIdentity(FrozenModel):
    opponent: EvaluationOpponentIdentity
    common_search_sha256: Sha256
    board_size: int = Field(ge=3)
    komi_half_points: int
    scoring_rule: Literal['area']
    ko_rule: Literal['positional_superko']
    suicide_rule: Literal['illegal']


class ReportRun(FrozenModel):
    identity: RunIdentity
    evaluation_protocol: EvaluationProtocolIdentity
    diagnostics: RunDiagnostics
    final_match: MatchStatistics
    learning_curve: LearningCurveStatistics


class ResearchReport(FrozenModel):
    schema_version: Literal[1] = 1
    report_id: UUID
    title: str = Field(min_length=1)
    matrix_id: UUID
    common_controls_sha256: Sha256
    runs: tuple[ReportRun, ...] = Field(min_length=1)
    independence_note: Literal[
        'Paired games share conditions and are resampled as pairs; games within a pair are not treated as independent.'
    ]
    source_artifact_sha256s: tuple[Sha256, ...]

    @model_validator(mode='after')
    def validate_runs(self) -> ResearchReport:
        run_ids = tuple(run.identity.run_id for run in self.runs)
        if len(set(run_ids)) != len(run_ids):
            raise ValueError('Research report run IDs must be unique.')
        return self


class ResearchReportArtifact(FrozenModel):
    payload: ResearchReport
    payload_sha256: Sha256

    @model_validator(mode='after')
    def validate_digest(self) -> ResearchReportArtifact:
        if self.payload_sha256 != model_sha256(self.payload):
            raise ValueError('Research report payload SHA-256 does not match.')
        return self

    @classmethod
    def create(cls, payload: ResearchReport) -> ResearchReportArtifact:
        return cls(payload=payload, payload_sha256=model_sha256(payload))
