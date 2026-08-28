from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from pathlib import Path

from pydantic import Field, ValidationError, model_validator
from src.search_budget.curve import BLEND_CANDIDATES
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel


class BlendDecisionReason(str, Enum):
    INITIAL = 'initial'
    WARMUP = 'warmup'
    BEST_ELIGIBLE = 'best_eligible'
    NO_ELIGIBLE_NONZERO = 'no_eligible_nonzero'
    TERMINAL_FAILURE = 'terminal_failure'
    INCOMPATIBLE_STATE = 'incompatible_state'
    INVALID_COMPUTE = 'invalid_compute'
    UNREADABLE_STATE = 'unreadable_state'


class BlendEligibilityFailure(str, Enum):
    ZERO_CANDIDATE = 'zero_candidate'
    WARMUP = 'warmup'
    NON_POSITIVE_CURRENT_GAIN = 'non_positive_current_gain'
    NON_POSITIVE_EMA_GAIN = 'non_positive_ema_gain'
    SPEND_MISMATCH = 'spend_mismatch'
    UPWARD_STEP_LIMIT = 'upward_step_limit'


@dataclass(frozen=True)
class BlendCalibrationParameters:
    candidate_blends: tuple[Decimal, ...] = BLEND_CANDIDATES
    warmup_completed_generations: int = 30
    ema_decay: Decimal = Decimal('0.2')
    maximum_upward_step: Decimal = Decimal('0.1')

    def __post_init__(self) -> None:
        if self.candidate_blends != BLEND_CANDIDATES:
            raise ValueError('The first-run blend grid must be [0.0, 0.1, ..., 1.0].')
        if self.warmup_completed_generations != 30:
            raise ValueError('The first-run warm-up is exactly 30 completed label generations.')
        if self.ema_decay != Decimal('0.2'):
            raise ValueError('The first-run EMA decay is exactly 0.2.')
        if self.maximum_upward_step != Decimal('0.1'):
            raise ValueError('The first-run maximum upward step is exactly 0.1.')


class BlendGenerationEvidence(FrozenModel):
    blend: Decimal = Field(ge=Decimal(0), le=Decimal(1))
    generation_gain: float
    total_assigned_new_visits: int = Field(gt=0)
    flat_total_new_visits: int = Field(gt=0)
    position_count: int = Field(gt=0)

    @model_validator(mode='after')
    def validate_gain(self) -> BlendGenerationEvidence:
        if not Decimal(str(self.generation_gain)).is_finite():
            raise ValueError('Generation gain must be finite.')
        return self


class BlendCandidateState(FrozenModel):
    blend: Decimal = Field(ge=Decimal(0), le=Decimal(1))
    current_generation_gain: float
    ema_gain: float
    total_assigned_new_visits: int = Field(gt=0)
    flat_total_new_visits: int = Field(gt=0)
    position_count: int = Field(gt=0)
    failed_eligibility_conditions: tuple[BlendEligibilityFailure, ...]


class BlendCalibrationState(FrozenModel):
    schema_version: int = Field(default=1, ge=1, le=1)
    configuration_sha256: str = Field(min_length=64, max_length=64)
    finalized_source_generations: tuple[int, ...]
    previous_blend: Decimal = Field(ge=Decimal(0), le=Decimal(1))
    selected_blend: Decimal = Field(ge=Decimal(0), le=Decimal(1))
    application_generation: int = Field(ge=0)
    candidate_states: tuple[BlendCandidateState, ...]
    decision_reason: BlendDecisionReason

    @model_validator(mode='after')
    def validate_state(self) -> BlendCalibrationState:
        if tuple(sorted(set(self.finalized_source_generations))) != self.finalized_source_generations:
            raise ValueError('Finalized source generations must be unique and increasing.')
        if self.candidate_states and tuple(candidate.blend for candidate in self.candidate_states) != BLEND_CANDIDATES:
            raise ValueError('Candidate states must use the complete settled blend grid.')
        return self


class BlendPublication(FrozenModel):
    blend: Decimal = Field(ge=Decimal(0), le=Decimal(1))
    application_generation: int = Field(ge=0)
    decision_reason: BlendDecisionReason


@dataclass(frozen=True)
class CalibrationUpdate:
    state: BlendCalibrationState
    applied: bool


def initial_calibration_state(configuration_sha256: str) -> BlendCalibrationState:
    return BlendCalibrationState(
        configuration_sha256=configuration_sha256,
        finalized_source_generations=(),
        previous_blend=Decimal(0),
        selected_blend=Decimal(0),
        application_generation=0,
        candidate_states=(),
        decision_reason=BlendDecisionReason.INITIAL,
    )


def update_calibration(
    state: BlendCalibrationState,
    source_generation: int,
    evidence: tuple[BlendGenerationEvidence, ...],
    first_unstarted_production_generation: int,
    parameters: BlendCalibrationParameters = BlendCalibrationParameters(),
) -> CalibrationUpdate:
    if source_generation < 0:
        raise ValueError('Source generation must be nonnegative.')
    if first_unstarted_production_generation <= source_generation:
        raise ValueError('Blend publication must apply after its source generation.')
    if source_generation in state.finalized_source_generations:
        return CalibrationUpdate(state=state, applied=False)
    if state.finalized_source_generations and source_generation < state.finalized_source_generations[-1]:
        raise ValueError('Source-generation label jobs must finalize in source order.')
    ordered_evidence = tuple(sorted(evidence, key=lambda candidate: candidate.blend))
    if tuple(candidate.blend for candidate in ordered_evidence) != parameters.candidate_blends:
        raise ValueError('Generation evidence must contain every settled blend candidate exactly once.')

    previous_candidates = {candidate.blend: candidate for candidate in state.candidate_states}
    completed_count = len(state.finalized_source_generations) + 1
    candidate_states: list[BlendCandidateState] = []
    eligible_candidates: list[BlendCandidateState] = []
    for candidate in ordered_evidence:
        previous = previous_candidates.get(candidate.blend)
        ema_gain = (
            candidate.generation_gain
            if previous is None
            else (1.0 - float(parameters.ema_decay)) * previous.ema_gain
            + float(parameters.ema_decay) * candidate.generation_gain
        )
        failures = _eligibility_failures(candidate, ema_gain, completed_count, state.selected_blend, parameters)
        candidate_state = BlendCandidateState(
            blend=candidate.blend,
            current_generation_gain=candidate.generation_gain,
            ema_gain=ema_gain,
            total_assigned_new_visits=candidate.total_assigned_new_visits,
            flat_total_new_visits=candidate.flat_total_new_visits,
            position_count=candidate.position_count,
            failed_eligibility_conditions=failures,
        )
        candidate_states.append(candidate_state)
        if not failures:
            eligible_candidates.append(candidate_state)

    if completed_count < parameters.warmup_completed_generations:
        selected_blend = Decimal(0)
        decision_reason = BlendDecisionReason.WARMUP
    elif eligible_candidates:
        selected_blend = min(eligible_candidates, key=lambda candidate: (-candidate.ema_gain, candidate.blend)).blend
        decision_reason = BlendDecisionReason.BEST_ELIGIBLE
    else:
        selected_blend = Decimal(0)
        decision_reason = BlendDecisionReason.NO_ELIGIBLE_NONZERO

    previous_blend = (
        Decimal(0)
        if first_unstarted_production_generation == 0
        else published_blend_for_generation(state, first_unstarted_production_generation - 1)
    )
    next_state = BlendCalibrationState(
        configuration_sha256=state.configuration_sha256,
        finalized_source_generations=(*state.finalized_source_generations, source_generation),
        previous_blend=previous_blend,
        selected_blend=selected_blend,
        application_generation=first_unstarted_production_generation,
        candidate_states=tuple(candidate_states),
        decision_reason=decision_reason,
    )
    return CalibrationUpdate(state=next_state, applied=True)


def published_blend_for_generation(state: BlendCalibrationState, production_generation: int) -> Decimal:
    if production_generation < 0:
        raise ValueError('Production generation must be nonnegative.')
    if production_generation < state.application_generation:
        return state.previous_blend
    return state.selected_blend


def publish_fail_closed(
    state: BlendCalibrationState,
    first_unstarted_production_generation: int,
    reason: BlendDecisionReason,
) -> BlendCalibrationState:
    if reason not in {
        BlendDecisionReason.TERMINAL_FAILURE,
        BlendDecisionReason.INCOMPATIBLE_STATE,
        BlendDecisionReason.INVALID_COMPUTE,
        BlendDecisionReason.UNREADABLE_STATE,
    }:
        raise ValueError('Fail-closed publication requires a failure decision reason.')
    previous_blend = (
        Decimal(0)
        if first_unstarted_production_generation == 0
        else published_blend_for_generation(state, first_unstarted_production_generation - 1)
    )
    return state.model_copy(
        update={
            'previous_blend': previous_blend,
            'selected_blend': Decimal(0),
            'application_generation': first_unstarted_production_generation,
            'decision_reason': reason,
        }
    )


def save_calibration_state(path: Path, state: BlendCalibrationState) -> None:
    write_text_atomically(path, state.model_dump_json(indent=2) + '\n')


def load_calibration_state_fail_closed(
    path: Path,
    expected_configuration_sha256: str,
    first_unstarted_production_generation: int,
) -> BlendCalibrationState:
    try:
        state = BlendCalibrationState.model_validate_json(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, ValidationError):
        return publish_fail_closed(
            initial_calibration_state(expected_configuration_sha256),
            first_unstarted_production_generation,
            BlendDecisionReason.UNREADABLE_STATE,
        )
    if state.configuration_sha256 != expected_configuration_sha256:
        return publish_fail_closed(
            initial_calibration_state(expected_configuration_sha256),
            first_unstarted_production_generation,
            BlendDecisionReason.INCOMPATIBLE_STATE,
        )
    return state


def publication_for_generation(state: BlendCalibrationState, production_generation: int) -> BlendPublication:
    return BlendPublication(
        blend=published_blend_for_generation(state, production_generation),
        application_generation=state.application_generation,
        decision_reason=state.decision_reason,
    )


def _eligibility_failures(
    evidence: BlendGenerationEvidence,
    ema_gain: float,
    completed_count: int,
    current_blend: Decimal,
    parameters: BlendCalibrationParameters,
) -> tuple[BlendEligibilityFailure, ...]:
    failures: list[BlendEligibilityFailure] = []
    if evidence.blend == 0:
        failures.append(BlendEligibilityFailure.ZERO_CANDIDATE)
    if completed_count < parameters.warmup_completed_generations:
        failures.append(BlendEligibilityFailure.WARMUP)
    if evidence.generation_gain <= 0.0:
        failures.append(BlendEligibilityFailure.NON_POSITIVE_CURRENT_GAIN)
    if ema_gain <= 0.0:
        failures.append(BlendEligibilityFailure.NON_POSITIVE_EMA_GAIN)
    if evidence.total_assigned_new_visits != evidence.flat_total_new_visits:
        failures.append(BlendEligibilityFailure.SPEND_MISMATCH)
    if evidence.blend > current_blend + parameters.maximum_upward_step:
        failures.append(BlendEligibilityFailure.UPWARD_STEP_LIMIT)
    return tuple(failures)
