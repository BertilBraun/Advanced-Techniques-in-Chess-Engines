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
    allocate_candidate_budget_grid,
    deep_label_visit_limit,
)
from src.search_budget.calibration import BlendGenerationEvidence
from src.search_budget.curve import CURVE_QUANTILE_BOUNDARIES
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


class CandidateGenerationDiagnostics(FrozenModel):
    blend: Decimal = Field(ge=Decimal(0), le=Decimal(1))
    generation_gain: float
    mean_assigned_new_visits: float = Field(gt=0.0)
    assigned_new_visits_variance: float = Field(ge=0.0)
    mean_kl_from_deep: float = Field(ge=0.0)
    exact_spend_residual: int
    floor_share: float = Field(ge=0.0, le=1.0)
    ceiling_share: float = Field(ge=0.0, le=1.0)


@dataclass(frozen=True)
class GenerationFinalization:
    replay_samples: tuple[ReplaySample, ...]
    evidence: tuple[BlendGenerationEvidence, ...]
    candidate_diagnostics: tuple[CandidateGenerationDiagnostics, ...]
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
) -> tuple[CandidateBudgetSet, ...]:
    return allocate_candidate_budget_grid(
        tuple(
            AllocationPosition(position.identity, predictions[position.identity].predicted_quantile)
            for position in source.selected_positions
        ),
        source.baseline_new_visits,
    )


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
    evidence: list[BlendGenerationEvidence] = []
    diagnostics: list[CandidateGenerationDiagnostics] = []
    prediction_values = tuple(predictions[identity].predicted_quantile for identity in expected)
    floor_boundary = float(CURVE_QUANTILE_BOUNDARIES[0])
    ceiling_boundary = float(CURVE_QUANTILE_BOUNDARIES[-2])
    floor_share = sum(value < floor_boundary for value in prediction_values) / len(prediction_values)
    ceiling_share = sum(value >= ceiling_boundary for value in prediction_values) / len(prediction_values)
    for allocation in allocations:
        budget_by_identity = {budget.identity: budget.assigned_new_visits for budget in allocation.budgets}
        candidate_policies = tuple(
            _policy_at(deep_by_identity[identity], budget_by_identity[identity], action_size) for identity in expected
        )
        gain = shadow_gain(deep_policies, flat_policies, candidate_policies)
        candidate_kl_values = tuple(
            policy_kl(deep, candidate) for deep, candidate in zip(deep_policies, candidate_policies, strict=True)
        )
        assigned_visits = tuple(budget_by_identity[identity] for identity in expected)
        evidence.append(
            BlendGenerationEvidence(
                blend=allocation.blend,
                generation_gain=gain,
                total_assigned_new_visits=allocation.total_assigned_new_visits,
                flat_total_new_visits=allocation.flat_total_new_visits,
                position_count=len(expected),
            )
        )
        diagnostics.append(
            CandidateGenerationDiagnostics(
                blend=allocation.blend,
                generation_gain=gain,
                mean_assigned_new_visits=allocation.total_assigned_new_visits / len(expected),
                assigned_new_visits_variance=pvariance(assigned_visits),
                mean_kl_from_deep=fmean(candidate_kl_values),
                exact_spend_residual=allocation.spend_error,
                floor_share=floor_share,
                ceiling_share=ceiling_share,
            )
        )
    return GenerationFinalization(
        replay_samples=replay_samples,
        evidence=tuple(evidence),
        candidate_diagnostics=tuple(diagnostics),
        prediction_distribution=_distribution(prediction_values),
        target_distribution=_distribution(normalized_targets),
        raw_kl_distribution=_distribution(raw_kl_values),
    )


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
