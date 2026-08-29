from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from pathlib import Path

from pydantic import Field, ValidationError, model_validator
from src.search_budget.curve import (
    CURVE_BUCKET_COUNT,
    SearchBudgetCurve,
    analytic_initial_curve,
    bounded_curve_toward,
    flat_curve,
    update_shadow_curve,
)
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel


class CurveDecisionReason(str, Enum):
    INITIAL = 'initial'
    WARMUP = 'warmup'
    VALIDATED_PENDING = 'validated_pending'
    NO_ELIGIBLE_PENDING = 'no_eligible_pending'
    TERMINAL_FAILURE = 'terminal_failure'
    INCOMPATIBLE_STATE = 'incompatible_state'
    INVALID_COMPUTE = 'invalid_compute'
    UNREADABLE_STATE = 'unreadable_state'


class CurveEligibilityFailure(str, Enum):
    NO_PENDING_CURVE = 'no_pending_curve'
    WARMUP = 'warmup'
    NON_POSITIVE_CURRENT_GAIN = 'non_positive_current_gain'
    NON_POSITIVE_EMA_GAIN = 'non_positive_ema_gain'
    SPEND_MISMATCH = 'spend_mismatch'


@dataclass(frozen=True)
class CurveCalibrationParameters:
    warmup_completed_generations: int = 30
    bucket_utility_ema_decay: Decimal = Decimal('0.2')
    validation_gain_ema_decay: Decimal = Decimal('0.2')
    maximum_step_ratio: Decimal = Decimal('1.1')

    def __post_init__(self) -> None:
        if self.warmup_completed_generations != 30:
            raise ValueError('The first-run warm-up is exactly 30 completed label generations.')
        if self.bucket_utility_ema_decay != Decimal('0.2'):
            raise ValueError('The first-run bucket-utility EMA decay is exactly 0.2.')
        if self.validation_gain_ema_decay != Decimal('0.2'):
            raise ValueError('The first-run validation-gain EMA decay is exactly 0.2.')
        if self.maximum_step_ratio != Decimal('1.1'):
            raise ValueError('The first-run maximum multiplicative curve step is exactly 1.1.')


class BucketGenerationEvidence(FrozenModel):
    bucket_index: int = Field(ge=0, lt=CURVE_BUCKET_COUNT)
    sample_count: int = Field(ge=0)
    generation_marginal_utility: float | None

    @model_validator(mode='after')
    def validate_observation(self) -> BucketGenerationEvidence:
        if (self.sample_count == 0) != (self.generation_marginal_utility is None):
            raise ValueError('Only an empty bucket may omit generation marginal utility.')
        if self.generation_marginal_utility is not None and not math.isfinite(self.generation_marginal_utility):
            raise ValueError('Generation marginal utility must be finite.')
        return self


class CurveGenerationEvidence(FrozenModel):
    bucket_evidence: tuple[BucketGenerationEvidence, ...] = Field(
        min_length=CURVE_BUCKET_COUNT,
        max_length=CURVE_BUCKET_COUNT,
    )
    validated_curve: SearchBudgetCurve | None
    generation_gain: float | None
    total_assigned_new_visits: int | None = Field(default=None, gt=0)
    flat_total_new_visits: int = Field(gt=0)
    position_count: int = Field(gt=0)

    @model_validator(mode='after')
    def validate_evidence(self) -> CurveGenerationEvidence:
        if tuple(bucket.bucket_index for bucket in self.bucket_evidence) != tuple(range(CURVE_BUCKET_COUNT)):
            raise ValueError('Curve evidence must contain every bucket exactly once in order.')
        fields_present = (
            self.validated_curve is not None,
            self.generation_gain is not None,
            self.total_assigned_new_visits is not None,
        )
        if len(set(fields_present)) != 1:
            raise ValueError('Pending-curve validation fields must all be present or all absent.')
        if self.generation_gain is not None and not math.isfinite(self.generation_gain):
            raise ValueError('Generation validation gain must be finite.')
        return self


class BucketCalibrationState(FrozenModel):
    bucket_index: int = Field(ge=0, lt=CURVE_BUCKET_COUNT)
    sample_count: int = Field(ge=0)
    current_generation_utility: float | None
    ema_utility: float | None
    raw_log_update: float
    projection_adjustment: float

    @model_validator(mode='after')
    def validate_finite_state(self) -> BucketCalibrationState:
        values = (
            self.current_generation_utility,
            self.ema_utility,
            self.raw_log_update,
            self.projection_adjustment,
        )
        if any(value is not None and not math.isfinite(value) for value in values):
            raise ValueError('Persisted bucket calibration values must be finite.')
        return self


class CurveCalibrationState(FrozenModel):
    schema_version: int = Field(default=2, ge=2, le=2)
    configuration_sha256: str = Field(min_length=64, max_length=64)
    finalized_source_generations: tuple[int, ...]
    bucket_states: tuple[BucketCalibrationState, ...] = Field(
        min_length=CURVE_BUCKET_COUNT,
        max_length=CURVE_BUCKET_COUNT,
    )
    shadow_curve: SearchBudgetCurve
    pending_curve: SearchBudgetCurve | None
    pending_source_generation: int | None = Field(default=None, ge=0)
    previous_published_curve: SearchBudgetCurve
    published_curve: SearchBudgetCurve
    application_generation: int = Field(ge=0)
    current_validation_gain: float | None
    ema_validation_gain: float | None
    failed_eligibility_conditions: tuple[CurveEligibilityFailure, ...]
    decision_reason: CurveDecisionReason

    @model_validator(mode='after')
    def validate_state(self) -> CurveCalibrationState:
        if tuple(sorted(set(self.finalized_source_generations))) != self.finalized_source_generations:
            raise ValueError('Finalized source generations must be unique and increasing.')
        if tuple(bucket.bucket_index for bucket in self.bucket_states) != tuple(range(CURVE_BUCKET_COUNT)):
            raise ValueError('Calibration state must contain every bucket exactly once in order.')
        if (self.pending_curve is None) != (self.pending_source_generation is None):
            raise ValueError('Pending curve and its construction generation must be present together.')
        if any(
            value is not None and not math.isfinite(value)
            for value in (self.current_validation_gain, self.ema_validation_gain)
        ):
            raise ValueError('Persisted curve validation gains must be finite.')
        return self


class CurvePublication(FrozenModel):
    curve: SearchBudgetCurve
    application_generation: int = Field(ge=0)
    decision_reason: CurveDecisionReason


@dataclass(frozen=True)
class CalibrationUpdate:
    state: CurveCalibrationState
    applied: bool


def initial_calibration_state(configuration_sha256: str) -> CurveCalibrationState:
    return CurveCalibrationState(
        configuration_sha256=configuration_sha256,
        finalized_source_generations=(),
        bucket_states=tuple(
            BucketCalibrationState(
                bucket_index=index,
                sample_count=0,
                current_generation_utility=None,
                ema_utility=None,
                raw_log_update=0.0,
                projection_adjustment=0.0,
            )
            for index in range(CURVE_BUCKET_COUNT)
        ),
        shadow_curve=analytic_initial_curve(),
        pending_curve=None,
        previous_published_curve=flat_curve(),
        published_curve=flat_curve(),
        application_generation=0,
        current_validation_gain=None,
        ema_validation_gain=None,
        failed_eligibility_conditions=(CurveEligibilityFailure.NO_PENDING_CURVE,),
        decision_reason=CurveDecisionReason.INITIAL,
    )


def update_calibration(
    state: CurveCalibrationState,
    source_generation: int,
    evidence: CurveGenerationEvidence,
    first_unstarted_production_generation: int,
    parameters: CurveCalibrationParameters = CurveCalibrationParameters(),
) -> CalibrationUpdate:
    if source_generation < 0:
        raise ValueError('Source generation must be nonnegative.')
    if first_unstarted_production_generation <= source_generation:
        raise ValueError('Curve publication must apply after its source generation.')
    if source_generation in state.finalized_source_generations:
        return CalibrationUpdate(state=state, applied=False)
    if state.finalized_source_generations and source_generation < state.finalized_source_generations[-1]:
        raise ValueError('Source-generation label jobs must finalize in source order.')
    if evidence.validated_curve != state.pending_curve:
        raise ValueError('Generation evidence must validate the curve pending before its search started.')
    if state.pending_source_generation is not None and state.pending_source_generation >= source_generation:
        raise ValueError('A pending curve must be validated on a later source generation.')

    utility_decay = float(parameters.bucket_utility_ema_decay)
    previous_buckets = {bucket.bucket_index: bucket for bucket in state.bucket_states}
    ema_utilities: list[float | None] = []
    for bucket in evidence.bucket_evidence:
        previous = previous_buckets[bucket.bucket_index]
        if bucket.generation_marginal_utility is None:
            ema_utilities.append(previous.ema_utility)
        elif previous.ema_utility is None:
            ema_utilities.append(bucket.generation_marginal_utility)
        else:
            ema_utilities.append(
                (1.0 - utility_decay) * previous.ema_utility + utility_decay * bucket.generation_marginal_utility
            )
    curve_update = update_shadow_curve(
        state.shadow_curve,
        tuple(ema_utilities),
        tuple(bucket.sample_count for bucket in evidence.bucket_evidence),
        float(parameters.maximum_step_ratio),
    )
    current_gain = evidence.generation_gain
    if current_gain is None:
        ema_gain = state.ema_validation_gain
    elif state.ema_validation_gain is None:
        ema_gain = current_gain
    else:
        validation_decay = float(parameters.validation_gain_ema_decay)
        ema_gain = (1.0 - validation_decay) * state.ema_validation_gain + validation_decay * current_gain

    completed_count = len(state.finalized_source_generations) + 1
    failures = _eligibility_failures(evidence, ema_gain, completed_count, parameters)
    previous_published = published_curve_for_generation(state, max(0, first_unstarted_production_generation - 1))
    if not failures:
        assert state.pending_curve is not None
        published = state.pending_curve
        decision_reason = CurveDecisionReason.VALIDATED_PENDING
    elif CurveEligibilityFailure.WARMUP in failures:
        # Nothing has been validated yet, so there is no earlier curve to fall back towards.
        published = flat_curve()
        decision_reason = CurveDecisionReason.WARMUP
    else:
        # Single-generation gain is noisy enough to go negative on its own, and republishing a flat
        # curve threw away every validated bucket and cost ~30 generations of bounded steps to climb
        # back. Decay towards flat at the same 10% per-generation bound the curve rises by, so a noisy
        # generation pauses progress and only a persistent negative gain disables the curve.
        published = bounded_curve_toward(previous_published, flat_curve(), float(parameters.maximum_step_ratio))
        decision_reason = CurveDecisionReason.NO_ELIGIBLE_PENDING
    pending = bounded_curve_toward(published, curve_update.curve, float(parameters.maximum_step_ratio))
    bucket_states = tuple(
        BucketCalibrationState(
            bucket_index=bucket.bucket_index,
            sample_count=bucket.sample_count,
            current_generation_utility=bucket.generation_marginal_utility,
            ema_utility=ema_utility,
            raw_log_update=raw_update,
            projection_adjustment=projection_adjustment,
        )
        for bucket, ema_utility, raw_update, projection_adjustment in zip(
            evidence.bucket_evidence,
            ema_utilities,
            curve_update.raw_log_updates,
            curve_update.projection_adjustments,
            strict=True,
        )
    )
    next_state = CurveCalibrationState(
        configuration_sha256=state.configuration_sha256,
        finalized_source_generations=(*state.finalized_source_generations, source_generation),
        bucket_states=bucket_states,
        shadow_curve=curve_update.curve,
        pending_curve=pending,
        pending_source_generation=source_generation,
        previous_published_curve=previous_published,
        published_curve=published,
        application_generation=first_unstarted_production_generation,
        current_validation_gain=current_gain,
        ema_validation_gain=ema_gain,
        failed_eligibility_conditions=failures,
        decision_reason=decision_reason,
    )
    return CalibrationUpdate(state=next_state, applied=True)


def published_curve_for_generation(state: CurveCalibrationState, production_generation: int) -> SearchBudgetCurve:
    if production_generation < 0:
        raise ValueError('Production generation must be nonnegative.')
    if production_generation < state.application_generation:
        return state.previous_published_curve
    return state.published_curve


def publish_fail_closed(
    state: CurveCalibrationState,
    first_unstarted_production_generation: int,
    reason: CurveDecisionReason,
) -> CurveCalibrationState:
    if reason not in {
        CurveDecisionReason.TERMINAL_FAILURE,
        CurveDecisionReason.INCOMPATIBLE_STATE,
        CurveDecisionReason.INVALID_COMPUTE,
        CurveDecisionReason.UNREADABLE_STATE,
    }:
        raise ValueError('Fail-closed publication requires a failure decision reason.')
    previous_curve = published_curve_for_generation(state, max(0, first_unstarted_production_generation - 1))
    return state.model_copy(
        update={
            'previous_published_curve': previous_curve,
            'published_curve': flat_curve(),
            'application_generation': first_unstarted_production_generation,
            'decision_reason': reason,
        }
    )


def save_calibration_state(path: Path, state: CurveCalibrationState) -> None:
    write_text_atomically(path, state.model_dump_json(indent=2) + '\n')


def load_calibration_state_fail_closed(
    path: Path,
    expected_configuration_sha256: str,
    first_unstarted_production_generation: int,
) -> CurveCalibrationState:
    try:
        state = CurveCalibrationState.model_validate_json(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, ValidationError):
        return publish_fail_closed(
            initial_calibration_state(expected_configuration_sha256),
            first_unstarted_production_generation,
            CurveDecisionReason.UNREADABLE_STATE,
        )
    if state.configuration_sha256 != expected_configuration_sha256:
        return publish_fail_closed(
            initial_calibration_state(expected_configuration_sha256),
            first_unstarted_production_generation,
            CurveDecisionReason.INCOMPATIBLE_STATE,
        )
    return state


def publication_for_generation(state: CurveCalibrationState, production_generation: int) -> CurvePublication:
    return CurvePublication(
        curve=published_curve_for_generation(state, production_generation),
        application_generation=state.application_generation,
        decision_reason=state.decision_reason,
    )


def _eligibility_failures(
    evidence: CurveGenerationEvidence,
    ema_gain: float | None,
    completed_count: int,
    parameters: CurveCalibrationParameters,
) -> tuple[CurveEligibilityFailure, ...]:
    failures: list[CurveEligibilityFailure] = []
    if evidence.validated_curve is None:
        failures.append(CurveEligibilityFailure.NO_PENDING_CURVE)
    if completed_count < parameters.warmup_completed_generations:
        failures.append(CurveEligibilityFailure.WARMUP)
    if evidence.generation_gain is None or evidence.generation_gain <= 0.0:
        failures.append(CurveEligibilityFailure.NON_POSITIVE_CURRENT_GAIN)
    if ema_gain is None or ema_gain <= 0.0:
        failures.append(CurveEligibilityFailure.NON_POSITIVE_EMA_GAIN)
    if (
        evidence.total_assigned_new_visits is not None
        and evidence.total_assigned_new_visits != evidence.flat_total_new_visits
    ):
        failures.append(CurveEligibilityFailure.SPEND_MISMATCH)
    return tuple(failures)
