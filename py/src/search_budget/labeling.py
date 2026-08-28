from __future__ import annotations

import math
from dataclasses import dataclass, replace
from decimal import Decimal
from fractions import Fraction
from statistics import fmean, pvariance
from typing import Protocol

from pydantic import Field, model_validator
from src.replay.contracts import EligibleSearchBudgetTarget, ReplaySample, SparsePolicyTarget
from src.replay.shard import ReplayShardGameMetadata
from src.search_budget.allocation import (
    AllocationPosition,
    CandidateBudgetSet,
    CurveAllocationIdentity,
    CurveAllocationPurpose,
    allocate_generation_multiplier_vector,
    deep_label_visit_limit,
)
from src.search_budget.calibration import (
    BucketGenerationEvidence,
    CurveCalibrationState,
    CurveGenerationEvidence,
)
from src.search_budget.curve import CURVE_BUCKET_COUNT, SearchBudgetCurve, bucket_index, flat_curve, probe_curve
from src.search_budget.sampling import LabelPositionIdentity, select_generation_sample
from src.search_budget.targets import PolicyDistribution, midrank_quantiles, policy_kl, shadow_gain
from src.self_play.completed_game import SearchVisitCounts
from src.training.checkpoint import CheckpointReference
from src.util.frozen_model import FrozenModel


class LabelPositionSource(FrozenModel):
    identity: LabelPositionIdentity
    game: ReplayShardGameMetadata
    observation_index: int = Field(ge=0)

    @model_validator(mode='after')
    def validate_observation(self) -> LabelPositionSource:
        if self.observation_index >= len(self.game.observations):
            raise ValueError('Label position observation index is outside its source game.')
        observation = self.game.observations[self.observation_index]
        if observation.ply != self.identity.ply:
            raise ValueError('Label position identity ply does not match its source observation.')
        if self.game.source.identity.archive_key != self.identity.game_identity:
            raise ValueError('Label position identity does not match its source game.')
        return self

    @property
    def action_prefix(self) -> tuple[int, ...]:
        return self.game.action_ids[: self.identity.ply]


class LabelGenerationSource(FrozenModel):
    schema_version: int = Field(default=1, ge=1, le=1)
    source_generation: int = Field(ge=0)
    population_position_count: int = Field(gt=0)
    baseline_new_visits: int = Field(gt=0)
    checkpoint: CheckpointReference
    selected_positions: tuple[LabelPositionSource, ...] = Field(min_length=1)

    @model_validator(mode='after')
    def validate_source(self) -> LabelGenerationSource:
        identities = tuple(position.identity for position in self.selected_positions)
        if len(set(identities)) != len(identities):
            raise ValueError('Selected label position identities must be unique.')
        if any(identity.source_generation != self.source_generation for identity in identities):
            raise ValueError('Selected label positions must belong to the logical source generation.')
        if self.checkpoint.generation != self.source_generation:
            raise ValueError('A label generation must use its immutable same-generation checkpoint.')
        return self

    @property
    def deep_visit_limit(self) -> int:
        return deep_label_visit_limit(self.baseline_new_visits)


class PredictionRecord(FrozenModel):
    identity: LabelPositionIdentity
    search_budget_logit: float
    predicted_quantile: float = Field(ge=0.0, le=1.0)

    @model_validator(mode='after')
    def validate_prediction(self) -> PredictionRecord:
        if not math.isfinite(self.search_budget_logit) or not math.isfinite(self.predicted_quantile):
            raise ValueError('Search-budget predictions must be finite.')
        if self.search_budget_logit >= 0.0:
            expected = 1.0 / (1.0 + math.exp(-self.search_budget_logit))
        else:
            exponential = math.exp(self.search_budget_logit)
            expected = exponential / (1.0 + exponential)
        if not math.isclose(expected, self.predicted_quantile, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError('Bounded prediction must be the sigmoid of its raw logit.')
        return self


class PredictionShardArtifact(FrozenModel):
    schema_version: int = Field(default=1, ge=1, le=1)
    source_generation: int = Field(ge=0)
    shard_index: int = Field(ge=0)
    checkpoint_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    predictions: tuple[PredictionRecord, ...] = Field(min_length=1, max_length=512)


class PolicyCheckpointRecord(FrozenModel):
    visits: int = Field(gt=0)
    policy_target_visits: SearchVisitCounts


class DeepSearchRecord(FrozenModel):
    identity: LabelPositionIdentity
    checkpoints: tuple[PolicyCheckpointRecord, ...] = Field(min_length=1)
    final_policy_target_visits: SearchVisitCounts
    final_root_value: float = Field(ge=-1.0, le=1.0)
    starting_visits: int = Field(ge=0)
    final_visits: int = Field(gt=0)

    @model_validator(mode='after')
    def validate_checkpoints(self) -> DeepSearchRecord:
        visits = tuple(checkpoint.visits for checkpoint in self.checkpoints)
        if visits != tuple(sorted(set(visits))):
            raise ValueError('Deep-label checkpoints must be unique and increasing.')
        if self.starting_visits != 0:
            raise ValueError('Deep-label searches must reconstruct a fresh root.')
        if self.final_visits <= self.starting_visits:
            raise ValueError('Deep-label search must add visits to its fresh root.')
        return self


class DeepSearchShardArtifact(FrozenModel):
    schema_version: int = Field(default=1, ge=1, le=1)
    source_generation: int = Field(ge=0)
    shard_index: int = Field(ge=0)
    checkpoint_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    records: tuple[DeepSearchRecord, ...] = Field(min_length=1, max_length=512)


class DistributionSummary(FrozenModel):
    count: int = Field(gt=0)
    minimum: float
    maximum: float
    mean: float
    variance: float = Field(ge=0.0)
    p10: float
    p25: float
    median: float
    p75: float
    p90: float
    histogram_counts: tuple[int, int, int, int, int, int, int, int, int, int]


class CurveValidationDiagnostics(FrozenModel):
    generation_gain: float | None
    mean_assigned_new_visits: float | None = Field(default=None, gt=0.0)
    assigned_new_visits_variance: float | None = Field(default=None, ge=0.0)
    mean_kl_from_deep: float | None = Field(default=None, ge=0.0)
    exact_spend_residual: int | None


class BucketGenerationDiagnostics(FrozenModel):
    bucket_index: int = Field(ge=0, lt=CURVE_BUCKET_COUNT)
    sample_count: int = Field(ge=0)
    generation_marginal_utility: float | None
    lower_mean_visits: float | None = Field(default=None, gt=0.0)
    upper_mean_visits: float | None = Field(default=None, gt=0.0)
    checkpoint_deduplication_count: int = Field(ge=0)


@dataclass(frozen=True)
class GenerationFinalization:
    replay_samples: tuple[ReplaySample, ...]
    evidence: CurveGenerationEvidence
    validation_diagnostics: CurveValidationDiagnostics
    bucket_diagnostics: tuple[BucketGenerationDiagnostics, ...]
    prediction_distribution: DistributionSummary
    target_distribution: DistributionSummary
    raw_kl_distribution: DistributionSummary


class ReplaySampleProvider(Protocol):
    def __call__(self, source: LabelPositionSource) -> ReplaySample: ...

    def close(self) -> None: ...


class ExperimentReplaySampleProvider:
    def __init__(self, configuration_json: str) -> None:
        from src.experiment.configuration import load_experiment_configuration_json
        from src.games.composition import create_game_implementation

        configuration = load_experiment_configuration_json(configuration_json)
        self._game = create_game_implementation(configuration)
        self._maximum_policy_entries = configuration.training.lifecycle.replay.maximum_policy_entries
        self._cache: dict[str, tuple[ReplaySample, ...]] = {}

    def __call__(self, source: LabelPositionSource) -> ReplaySample:
        from src.replay.materialization import materialize_completed_game

        game_key = source.game.source.identity.archive_key
        samples = self._cache.get(game_key)
        if samples is None:
            samples = tuple(
                materialize_completed_game(
                    source.game.completed_game(),
                    self._game.state,
                    self._game.terminal_oracle,
                    self._game.target_layout,
                    self._maximum_policy_entries,
                    self._game.value_discount_per_ply,
                    censor_remaining_game_length_on_cut_games=self._game.censor_remaining_game_length_on_cut_games,
                ).samples
            )
            self._cache[game_key] = samples
        return samples[source.observation_index]

    def close(self) -> None:
        self._cache.clear()
        self._game.close()


def build_generation_source(
    source_generation: int,
    games: tuple[ReplayShardGameMetadata, ...],
    checkpoint: CheckpointReference,
    baseline_new_visits: int,
    run_seed: int,
    sample_fraction: Decimal,
) -> LabelGenerationSource | None:
    if not games:
        raise ValueError('A source generation requires complete replay-game metadata.')
    positions = tuple(
        LabelPositionSource(
            identity=LabelPositionIdentity(
                source_generation=source_generation,
                game_identity=game.source.identity.archive_key,
                ply=observation.ply,
            ),
            game=game,
            observation_index=observation_index,
        )
        for game in games
        for observation_index, observation in enumerate(game.observations)
    )
    if not positions:
        raise ValueError('A source generation must contain at least one played position.')
    by_identity = {position.identity: position for position in positions}
    if len(by_identity) != len(positions):
        raise ValueError('Complete source-generation observations contain duplicate stable identities.')
    fraction = Fraction(sample_fraction)
    selected_identities = select_generation_sample(tuple(by_identity), run_seed, fraction)
    if not selected_identities:
        return None
    return LabelGenerationSource(
        source_generation=source_generation,
        population_position_count=len(positions),
        baseline_new_visits=baseline_new_visits,
        checkpoint=checkpoint,
        selected_positions=tuple(by_identity[identity] for identity in selected_identities),
    )


def prediction_map(
    source: LabelGenerationSource,
    artifacts: tuple[PredictionShardArtifact, ...],
) -> dict[LabelPositionIdentity, PredictionRecord]:
    records = tuple(record for artifact in artifacts for record in artifact.predictions)
    expected = tuple(position.identity for position in source.selected_positions)
    if tuple(record.identity for record in records) != expected:
        raise ValueError('Prediction artifacts do not provide exact selected-position coverage.')
    if any(artifact.checkpoint_sha256 != source.checkpoint.inference_model_sha256 for artifact in artifacts):
        raise ValueError('Prediction artifacts do not use the source generation checkpoint.')
    return {record.identity: record for record in records}


def candidate_allocations(
    source: LabelGenerationSource,
    predictions: dict[LabelPositionIdentity, PredictionRecord],
    calibration: CurveCalibrationState,
    probe_ratio: float,
) -> tuple[CandidateBudgetSet, ...]:
    positions = tuple(
        AllocationPosition(position.identity, predictions[position.identity].predicted_quantile)
        for position in source.selected_positions
    )
    allocations = [
        allocate_generation_multiplier_vector(
            positions,
            source.baseline_new_visits,
            flat_curve().multipliers,
            CurveAllocationIdentity(CurveAllocationPurpose.FLAT),
        )
    ]
    if calibration.pending_curve is not None:
        allocations.append(
            allocate_generation_multiplier_vector(
                positions,
                source.baseline_new_visits,
                calibration.pending_curve.multipliers,
                CurveAllocationIdentity(CurveAllocationPurpose.PENDING_VALIDATION),
            )
        )
    for selected_bucket in range(CURVE_BUCKET_COUNT):
        for purpose, upper in (
            (CurveAllocationPurpose.PROBE_LOWER, False),
            (CurveAllocationPurpose.PROBE_UPPER, True),
        ):
            allocations.append(
                allocate_generation_multiplier_vector(
                    positions,
                    source.baseline_new_visits,
                    probe_curve(calibration.shadow_curve, selected_bucket, probe_ratio, upper),
                    CurveAllocationIdentity(purpose, selected_bucket),
                )
            )
    return tuple(allocations)


def checkpoint_visits_by_position(
    source: LabelGenerationSource,
    allocations: tuple[CandidateBudgetSet, ...],
) -> dict[LabelPositionIdentity, tuple[int, ...]]:
    visits: dict[LabelPositionIdentity, set[int]] = {
        position.identity: {source.baseline_new_visits} for position in source.selected_positions
    }
    for allocation in allocations:
        for budget in allocation.budgets:
            visits[budget.identity].add(budget.assigned_new_visits)
    return {identity: tuple(sorted(values)) for identity, values in visits.items()}


def finalize_generation(
    source: LabelGenerationSource,
    predictions: dict[LabelPositionIdentity, PredictionRecord],
    allocations: tuple[CandidateBudgetSet, ...],
    deep_artifacts: tuple[DeepSearchShardArtifact, ...],
    action_size: int,
    maximum_policy_entries: int,
    sample_provider: ReplaySampleProvider,
) -> GenerationFinalization:
    deep_records = tuple(record for artifact in deep_artifacts for record in artifact.records)
    expected = tuple(position.identity for position in source.selected_positions)
    if tuple(record.identity for record in deep_records) != expected:
        raise ValueError('Deep-search artifacts do not provide exact selected-position coverage.')
    if any(artifact.checkpoint_sha256 != source.checkpoint.inference_model_sha256 for artifact in deep_artifacts):
        raise ValueError('Deep-search artifacts do not use the source generation checkpoint.')
    deep_by_identity = {record.identity: record for record in deep_records}
    flat_policies = tuple(
        _policy_at(deep_by_identity[identity], source.baseline_new_visits, action_size) for identity in expected
    )
    deep_policies = tuple(
        _policy_distribution(deep_by_identity[identity].final_policy_target_visits, action_size)
        for identity in expected
    )
    raw_kl_values = tuple(policy_kl(deep, flat) for deep, flat in zip(deep_policies, flat_policies, strict=True))
    if any(not math.isfinite(value) for value in raw_kl_values):
        raise ValueError('Deep-label KL reconstruction must be finite.')
    normalized_targets = midrank_quantiles(raw_kl_values)
    source_by_identity = {position.identity: position for position in source.selected_positions}
    replay_samples = tuple(
        _labelled_replay_sample(
            sample_provider(source_by_identity[identity]),
            deep_by_identity[identity],
            raw_kl,
            normalized_target,
            predictions[identity],
            source,
            maximum_policy_entries,
        )
        for identity, raw_kl, normalized_target in zip(expected, raw_kl_values, normalized_targets, strict=True)
    )
    prediction_values = tuple(predictions[identity].predicted_quantile for identity in expected)
    allocations_by_identity = {allocation.identity: allocation for allocation in allocations}
    pending_allocation = allocations_by_identity.get(CurveAllocationIdentity(CurveAllocationPurpose.PENDING_VALIDATION))
    if pending_allocation is None:
        generation_gain = None
        pending_total = None
        validation_diagnostics = CurveValidationDiagnostics(
            generation_gain=None,
            mean_assigned_new_visits=None,
            assigned_new_visits_variance=None,
            mean_kl_from_deep=None,
            exact_spend_residual=None,
        )
    else:
        pending_budgets = {budget.identity: budget.assigned_new_visits for budget in pending_allocation.budgets}
        pending_policies = tuple(
            _policy_at(deep_by_identity[identity], pending_budgets[identity], action_size) for identity in expected
        )
        generation_gain = shadow_gain(deep_policies, flat_policies, pending_policies)
        pending_kl_values = tuple(
            policy_kl(deep, candidate) for deep, candidate in zip(deep_policies, pending_policies, strict=True)
        )
        assigned_visits = tuple(pending_budgets[identity] for identity in expected)
        pending_total = pending_allocation.total_assigned_new_visits
        validation_diagnostics = CurveValidationDiagnostics(
            generation_gain=generation_gain,
            mean_assigned_new_visits=pending_total / len(expected),
            assigned_new_visits_variance=pvariance(assigned_visits),
            mean_kl_from_deep=fmean(pending_kl_values),
            exact_spend_residual=pending_allocation.spend_error,
        )

    bucket_evidence: list[BucketGenerationEvidence] = []
    bucket_diagnostics: list[BucketGenerationDiagnostics] = []
    for selected_bucket in range(CURVE_BUCKET_COUNT):
        lower = allocations_by_identity[CurveAllocationIdentity(CurveAllocationPurpose.PROBE_LOWER, selected_bucket)]
        upper = allocations_by_identity[CurveAllocationIdentity(CurveAllocationPurpose.PROBE_UPPER, selected_bucket)]
        lower_budgets = {budget.identity: budget.assigned_new_visits for budget in lower.budgets}
        upper_budgets = {budget.identity: budget.assigned_new_visits for budget in upper.budgets}
        bucket_identities = tuple(
            identity
            for identity in expected
            if bucket_index(predictions[identity].predicted_quantile) == selected_bucket
        )
        utilities: list[float] = []
        deduplicated = 0
        multiplier_interval = (
            upper.allocation_multipliers[selected_bucket] - lower.allocation_multipliers[selected_bucket]
        )
        if multiplier_interval <= 0.0:
            raise ValueError('Upper local probe multiplier must exceed its lower probe multiplier.')
        for identity in bucket_identities:
            lower_visits = lower_budgets[identity]
            upper_visits = upper_budgets[identity]
            if upper_visits == lower_visits:
                deduplicated += 1
            if upper_visits < lower_visits:
                raise ValueError('Upper local probe allocated fewer visits than its lower probe.')
            deep_policy = _policy_distribution(deep_by_identity[identity].final_policy_target_visits, action_size)
            lower_policy = _policy_at(deep_by_identity[identity], lower_visits, action_size)
            upper_policy = _policy_at(deep_by_identity[identity], upper_visits, action_size)
            utilities.append(
                (policy_kl(deep_policy, lower_policy) - policy_kl(deep_policy, upper_policy)) / multiplier_interval
            )
        generation_utility = None if not utilities else fmean(utilities)
        bucket_evidence.append(
            BucketGenerationEvidence(
                bucket_index=selected_bucket,
                sample_count=len(utilities),
                generation_marginal_utility=generation_utility,
            )
        )
        bucket_diagnostics.append(
            BucketGenerationDiagnostics(
                bucket_index=selected_bucket,
                sample_count=len(utilities),
                generation_marginal_utility=generation_utility,
                lower_mean_visits=None
                if not bucket_identities
                else fmean(lower_budgets[item] for item in bucket_identities),
                upper_mean_visits=None
                if not bucket_identities
                else fmean(upper_budgets[item] for item in bucket_identities),
                checkpoint_deduplication_count=deduplicated,
            )
        )
    return GenerationFinalization(
        replay_samples=replay_samples,
        evidence=CurveGenerationEvidence(
            bucket_evidence=tuple(bucket_evidence),
            validated_curve=None
            if pending_allocation is None
            else _validated_curve_from_allocation(pending_allocation),
            generation_gain=generation_gain,
            total_assigned_new_visits=pending_total,
            flat_total_new_visits=source.baseline_new_visits * len(expected),
            position_count=len(expected),
        ),
        validation_diagnostics=validation_diagnostics,
        bucket_diagnostics=tuple(bucket_diagnostics),
        prediction_distribution=_distribution(prediction_values),
        target_distribution=_distribution(normalized_targets),
        raw_kl_distribution=_distribution(raw_kl_values),
    )


def _validated_curve_from_allocation(allocation: CandidateBudgetSet) -> SearchBudgetCurve:
    return SearchBudgetCurve(multipliers=allocation.allocation_multipliers)


def _policy_at(record: DeepSearchRecord, visits: int, action_size: int) -> PolicyDistribution:
    match = next((checkpoint for checkpoint in record.checkpoints if checkpoint.visits == visits), None)
    if match is None:
        raise ValueError(f'Deep-search record is missing required checkpoint {visits}.')
    return _policy_distribution(match.policy_target_visits, action_size)


def _policy_distribution(visits: SearchVisitCounts, action_size: int) -> PolicyDistribution:
    if any(action_id >= action_size for action_id in visits.action_ids):
        raise ValueError('Deep-label policy contains an action outside the configured action space.')
    total = sum(visits.visit_counts)
    probabilities = [0.0] * action_size
    for action_id, count in zip(visits.action_ids, visits.visit_counts, strict=True):
        probabilities[action_id] = count / total
    return PolicyDistribution(probabilities=tuple(probabilities))


def _labelled_replay_sample(
    base: ReplaySample,
    deep: DeepSearchRecord,
    raw_kl: float,
    normalized_target: float,
    prediction: PredictionRecord,
    source: LabelGenerationSource,
    maximum_policy_entries: int,
) -> ReplaySample:
    ordered_visits = sorted(
        zip(
            deep.final_policy_target_visits.action_ids,
            deep.final_policy_target_visits.visit_counts,
            strict=True,
        ),
        key=lambda item: (-item[1], item[0]),
    )[:maximum_policy_entries]
    deep_policy = SparsePolicyTarget(
        visits=SearchVisitCounts(
            action_ids=tuple(action_id for action_id, _ in ordered_visits),
            visit_counts=tuple(count for _, count in ordered_visits),
        ),
        legal_action_ids=base.policy.legal_action_ids,
    )
    target = EligibleSearchBudgetTarget(
        normalized_target=normalized_target,
        raw_kl=raw_kl,
        prediction_logit=prediction.search_budget_logit,
        predicted_quantile=prediction.predicted_quantile,
        source_generation=source.source_generation,
        model_generation=source.checkpoint.generation,
        inference_model_sha256=source.checkpoint.inference_model_sha256,
    )
    search_budget_slots = sum(auxiliary.kind == 'search_budget' for auxiliary in base.auxiliary_targets)
    if search_budget_slots != 1:
        raise ValueError('Deep-labelled replay requires exactly one configured search-budget target slot.')
    auxiliary_targets = tuple(
        target if auxiliary.kind == 'search_budget' else auxiliary for auxiliary in base.auxiliary_targets
    )
    return replace(base, policy=deep_policy, root_value=deep.final_root_value, auxiliary_targets=auxiliary_targets)


def _distribution(values: tuple[float, ...]) -> DistributionSummary:
    ordered = tuple(sorted(values))
    minimum = ordered[0]
    maximum = ordered[-1]
    histogram = [0] * 10
    if minimum == maximum:
        histogram[0] = len(ordered)
    else:
        width = maximum - minimum
        for value in ordered:
            index = min(9, int(10 * (value - minimum) / width))
            histogram[index] += 1
    return DistributionSummary(
        count=len(values),
        minimum=minimum,
        maximum=maximum,
        mean=fmean(values),
        variance=pvariance(values),
        p10=_empirical_quantile(ordered, 0.10),
        p25=_empirical_quantile(ordered, 0.25),
        median=_empirical_quantile(ordered, 0.50),
        p75=_empirical_quantile(ordered, 0.75),
        p90=_empirical_quantile(ordered, 0.90),
        histogram_counts=tuple(histogram),
    )


def _empirical_quantile(ordered: tuple[float, ...], probability: float) -> float:
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
