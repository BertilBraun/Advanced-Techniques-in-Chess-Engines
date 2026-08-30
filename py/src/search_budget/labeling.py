from __future__ import annotations

import math
from base64 import b64decode, b64encode
from dataclasses import dataclass, replace
from decimal import Decimal
from fractions import Fraction
from statistics import fmean, pvariance
from typing import Protocol

import numpy as np
import numpy.typing as npt
from pydantic import Field, model_validator
from src.games.contracts import WdlTarget
from src.games.representation import PackedPlanePayload
from src.replay.contracts import (
    AuxiliaryReplayTarget,
    EligibleSearchBudgetTarget,
    ReplaySample,
    SparsePolicyTarget,
)
from src.replay.label_source import ReplayLabelGameLocator
from src.search_budget.analysis_log import ANALYSIS_RECORD_DTYPE
from src.search_budget.calibration import BudgetGenerationEvidence
from src.search_budget.policy import (
    BASELINE_CURVE_INDEX,
    BUDGET_CURVE_MULTIPLES,
    BUDGET_CURVE_POINTS,
    HALF_DEEP_CURVE_INDEX,
    SearchBudgetPolicy,
    deep_label_visit_limit,
    grid_checkpoint_visits,
    grid_visit_counts,
    log_kl_curve,
    select_budget_index,
)
from src.search_budget.sampling import LabelPositionIdentity, select_generation_sample
from src.search_budget.targets import PolicyDistribution, policy_entropy, policy_kl, shadow_gain, top_visit_share
from src.self_play.completed_game import SearchVisitCounts
from src.training.checkpoint import CheckpointReference
from src.util.frozen_model import FrozenModel


class LabelReplaySampleSource(FrozenModel):
    encoded_state_base64: str = Field(min_length=1)
    policy: SparsePolicyTarget
    wdl_target: WdlTarget
    root_value: float = Field(ge=-1.0, le=1.0)
    auxiliary_targets: tuple[AuxiliaryReplayTarget, ...]
    sample_weight: float = Field(gt=0.0)
    source_model_generation: int = Field(ge=0)
    source_created_at_seconds: float = Field(ge=0.0)

    @classmethod
    def from_replay_sample(cls, sample: ReplaySample) -> LabelReplaySampleSource:
        return cls(
            encoded_state_base64=b64encode(bytes(sample.encoded_state)).decode('ascii'),
            policy=sample.policy,
            wdl_target=sample.wdl_target,
            root_value=sample.root_value,
            auxiliary_targets=sample.auxiliary_targets,
            sample_weight=sample.sample_weight,
            source_model_generation=sample.source_model_generation,
            source_created_at_seconds=sample.source_created_at_seconds,
        )

    @model_validator(mode='after')
    def validate_encoded_state(self) -> LabelReplaySampleSource:
        try:
            payload = b64decode(self.encoded_state_base64, validate=True)
        except ValueError as error:
            raise ValueError('Label replay sample state must use valid base64.') from error
        if not payload or b64encode(payload).decode('ascii') != self.encoded_state_base64:
            raise ValueError('Label replay sample state must use canonical nonempty base64.')
        return self

    def replay_sample(self) -> ReplaySample:
        return ReplaySample(
            encoded_state=PackedPlanePayload(b64decode(self.encoded_state_base64, validate=True)),
            policy=self.policy,
            wdl_target=self.wdl_target,
            root_value=self.root_value,
            auxiliary_targets=self.auxiliary_targets,
            sample_weight=self.sample_weight,
            source_model_generation=self.source_model_generation,
            source_created_at_seconds=self.source_created_at_seconds,
        )


class LabelPositionSource(FrozenModel):
    identity: LabelPositionIdentity
    action_prefix: tuple[int, ...]
    absolute_replay_row: int = Field(ge=0)
    replay: LabelReplaySampleSource

    @model_validator(mode='after')
    def validate_observation(self) -> LabelPositionSource:
        if len(self.action_prefix) != self.identity.ply:
            raise ValueError('Label position action prefix must end at its identity ply.')
        return self


class LabelGenerationSource(FrozenModel):
    schema_version: int = Field(default=3, ge=3, le=3)
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

    @property
    def checkpoint_visits(self) -> tuple[int, ...]:
        return grid_checkpoint_visits(self.baseline_new_visits)


class PredictionRecord(FrozenModel):
    identity: LabelPositionIdentity
    predicted_curve: tuple[float, ...] = Field(min_length=BUDGET_CURVE_POINTS, max_length=BUDGET_CURVE_POINTS)

    @model_validator(mode='after')
    def validate_prediction(self) -> PredictionRecord:
        if any(not math.isfinite(value) for value in self.predicted_curve):
            raise ValueError('Search-budget curve predictions must be finite.')
        return self


class PredictionShardArtifact(FrozenModel):
    schema_version: int = Field(default=2, ge=2, le=2)
    source_generation: int = Field(ge=0)
    shard_index: int = Field(ge=0)
    checkpoint_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    predictions: tuple[PredictionRecord, ...] = Field(min_length=1, max_length=512)


class PolicyCheckpointRecord(FrozenModel):
    visits: int = Field(gt=0)
    root_value: float = Field(ge=-1.0, le=1.0)
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
    schema_version: int = Field(default=2, ge=2, le=2)
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


class CurvePointDiagnostics(FrozenModel):
    curve_index: int = Field(ge=0, lt=BUDGET_CURVE_POINTS)
    grid_visits: int = Field(gt=0)
    mean_target_log_kl: float
    mean_predicted_log_kl: float
    mean_absolute_error: float = Field(ge=0.0)
    selected_count: int = Field(ge=0)


@dataclass(frozen=True)
class GenerationFinalization:
    replay_samples: tuple[ReplaySample, ...]
    evidence: BudgetGenerationEvidence
    curve_point_diagnostics: tuple[CurvePointDiagnostics, ...]
    baseline_raw_kl_distribution: DistributionSummary
    predicted_baseline_log_kl_distribution: DistributionSummary
    target_baseline_log_kl_distribution: DistributionSummary
    assigned_new_visits_variance: float
    analysis_records: npt.NDArray[np.void]


class ReplaySampleProvider(Protocol):
    def __call__(self, absolute_replay_row: int) -> ReplaySample: ...


def build_generation_source(
    source_generation: int,
    games: tuple[ReplayLabelGameLocator, ...],
    checkpoint: CheckpointReference,
    baseline_new_visits: int,
    run_seed: int,
    sample_fraction: Decimal,
    sample_provider: ReplaySampleProvider,
) -> LabelGenerationSource | None:
    if not games:
        raise ValueError('A source generation requires complete replay-game metadata.')
    candidates = tuple(
        (
            LabelPositionIdentity(
                source_generation=source_generation,
                game_identity=game.identity.archive_key,
                ply=ply,
            ),
            game,
            observation_index,
        )
        for game in games
        for observation_index, ply in enumerate(game.observation_plies)
    )
    if not candidates:
        raise ValueError('A source generation must contain at least one played position.')
    by_identity = {identity: (game, observation_index) for identity, game, observation_index in candidates}
    if len(by_identity) != len(candidates):
        raise ValueError('Complete source-generation observations contain duplicate stable identities.')
    fraction = Fraction(sample_fraction)
    selected_identities = select_generation_sample(tuple(by_identity), run_seed, fraction)
    if not selected_identities:
        return None
    return LabelGenerationSource(
        source_generation=source_generation,
        population_position_count=len(candidates),
        baseline_new_visits=baseline_new_visits,
        checkpoint=checkpoint,
        selected_positions=tuple(
            LabelPositionSource(
                identity=identity,
                action_prefix=by_identity[identity][0].action_ids[: identity.ply],
                absolute_replay_row=by_identity[identity][0].first_absolute_replay_row + by_identity[identity][1],
                replay=LabelReplaySampleSource.from_replay_sample(
                    sample_provider(by_identity[identity][0].first_absolute_replay_row + by_identity[identity][1])
                ),
            )
            for identity in selected_identities
        ),
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


def finalize_generation(
    source: LabelGenerationSource,
    predictions: dict[LabelPositionIdentity, PredictionRecord],
    policy: SearchBudgetPolicy,
    deep_artifacts: tuple[DeepSearchShardArtifact, ...],
    action_size: int,
    maximum_policy_entries: int,
) -> GenerationFinalization:
    deep_records = tuple(record for artifact in deep_artifacts for record in artifact.records)
    expected = tuple(position.identity for position in source.selected_positions)
    if tuple(record.identity for record in deep_records) != expected:
        raise ValueError('Deep-search artifacts do not provide exact selected-position coverage.')
    if any(artifact.checkpoint_sha256 != source.checkpoint.inference_model_sha256 for artifact in deep_artifacts):
        raise ValueError('Deep-search artifacts do not use the source generation checkpoint.')
    deep_by_identity = {record.identity: record for record in deep_records}
    source_by_identity = {position.identity: position for position in source.selected_positions}
    grid_visits = grid_visit_counts(source.baseline_new_visits)

    replay_samples: list[ReplaySample] = []
    error_sums = [0.0] * BUDGET_CURVE_POINTS
    target_sums = [0.0] * BUDGET_CURVE_POINTS
    prediction_sums = [0.0] * BUDGET_CURVE_POINTS
    selected_counts = [0] * BUDGET_CURVE_POINTS
    deep_policies: list[PolicyDistribution] = []
    flat_policies: list[PolicyDistribution] = []
    assigned_policies: list[PolicyDistribution] = []
    assigned_visits_values: list[int] = []
    baseline_raw_kls: list[float] = []
    predicted_baseline_log_kls: list[float] = []
    target_baseline_log_kls: list[float] = []
    analysis_records = np.zeros(len(expected), dtype=ANALYSIS_RECORD_DTYPE)

    for row_index, identity in enumerate(expected):
        deep = deep_by_identity[identity]
        deep_policy = _policy_distribution(deep.final_policy_target_visits, action_size)
        grid_policies = tuple(_policy_at(deep, visits, action_size) for visits in grid_visits)
        raw_kls = tuple(policy_kl(deep_policy, grid_policy) for grid_policy in grid_policies)
        if any(not math.isfinite(value) for value in raw_kls):
            raise ValueError('Deep-label KL reconstruction must be finite.')
        curve_label = log_kl_curve(raw_kls)
        predicted = predictions[identity].predicted_curve
        selected = select_budget_index(predicted, policy)
        selected_counts[selected] += 1
        assigned_visits_values.append(grid_visits[selected])
        deep_policies.append(deep_policy)
        flat_policies.append(grid_policies[BASELINE_CURVE_INDEX])
        assigned_policies.append(grid_policies[selected])
        baseline_raw_kls.append(raw_kls[BASELINE_CURVE_INDEX])
        predicted_baseline_log_kls.append(predicted[BASELINE_CURVE_INDEX])
        target_baseline_log_kls.append(curve_label[BASELINE_CURVE_INDEX])
        for index in range(BUDGET_CURVE_POINTS):
            error_sums[index] += abs(predicted[index] - curve_label[index])
            target_sums[index] += curve_label[index]
            prediction_sums[index] += predicted[index]
        replay_samples.append(
            _labelled_replay_sample(
                source_by_identity[identity].replay.replay_sample(),
                deep,
                curve_label,
                raw_kls[BASELINE_CURVE_INDEX],
                source,
                maximum_policy_entries,
            )
        )
        record = analysis_records[row_index]
        record['source_generation'] = source.source_generation
        record['model_generation'] = source.checkpoint.generation
        record['ply'] = identity.ply
        record['first_absolute_replay_row'] = source_by_identity[identity].absolute_replay_row
        record['baseline_visits'] = source.baseline_new_visits
        record['policy_kl'] = raw_kls
        record['value_error'] = tuple(
            abs(_root_value_at(deep, visits) - deep.final_root_value) for visits in grid_visits
        )
        record['top_visit_share'] = top_visit_share(grid_policies[BASELINE_CURVE_INDEX])
        record['policy_entropy'] = policy_entropy(grid_policies[BASELINE_CURVE_INDEX])
        record['predicted_curve'] = predicted
        record['deep_half_kl'] = raw_kls[HALF_DEEP_CURVE_INDEX]
        record['assigned_visits'] = grid_visits[selected]
        record['selected_index'] = selected

    position_count = len(expected)
    generation_gain = shadow_gain(tuple(deep_policies), tuple(flat_policies), tuple(assigned_policies))
    evidence = BudgetGenerationEvidence(
        position_count=position_count,
        mean_absolute_curve_error=tuple(value / position_count for value in error_sums),
        generation_gain=generation_gain,
        realized_mean_multiple=fmean(BUDGET_CURVE_MULTIPLES[index] for index in _selected_indices(selected_counts)),
        realized_mean_assigned_visits=fmean(assigned_visits_values),
        flat_mean_assigned_visits=float(source.baseline_new_visits),
        selected_index_counts=tuple(selected_counts),
    )
    diagnostics = tuple(
        CurvePointDiagnostics(
            curve_index=index,
            grid_visits=grid_visits[index],
            mean_target_log_kl=target_sums[index] / position_count,
            mean_predicted_log_kl=prediction_sums[index] / position_count,
            mean_absolute_error=error_sums[index] / position_count,
            selected_count=selected_counts[index],
        )
        for index in range(BUDGET_CURVE_POINTS)
    )
    return GenerationFinalization(
        replay_samples=tuple(replay_samples),
        evidence=evidence,
        curve_point_diagnostics=diagnostics,
        baseline_raw_kl_distribution=_distribution(tuple(baseline_raw_kls)),
        predicted_baseline_log_kl_distribution=_distribution(tuple(predicted_baseline_log_kls)),
        target_baseline_log_kl_distribution=_distribution(tuple(target_baseline_log_kls)),
        assigned_new_visits_variance=pvariance(assigned_visits_values) if position_count > 1 else 0.0,
        analysis_records=analysis_records,
    )


def _selected_indices(selected_counts: list[int]) -> tuple[int, ...]:
    return tuple(index for index, count in enumerate(selected_counts) for _ in range(count))


def _policy_at(record: DeepSearchRecord, visits: int, action_size: int) -> PolicyDistribution:
    return _policy_distribution(_checkpoint_at(record, visits).policy_target_visits, action_size)


def _root_value_at(record: DeepSearchRecord, visits: int) -> float:
    return _checkpoint_at(record, visits).root_value


def _checkpoint_at(record: DeepSearchRecord, visits: int) -> PolicyCheckpointRecord:
    match = next((checkpoint for checkpoint in record.checkpoints if checkpoint.visits == visits), None)
    if match is None:
        raise ValueError(f'Deep-search record is missing required checkpoint {visits}.')
    return match


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
    curve_label: tuple[float, ...],
    baseline_raw_kl: float,
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
        curve=curve_label,
        raw_kl=baseline_raw_kl,
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
