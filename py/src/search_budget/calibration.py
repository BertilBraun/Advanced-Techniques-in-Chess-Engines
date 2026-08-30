from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from pathlib import Path

from pydantic import Field, ValidationError, model_validator
from src.search_budget.policy import BUDGET_CURVE_POINTS, SearchBudgetPolicy
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel


class BudgetDecisionReason(str, Enum):
    INITIAL = 'initial'
    WARMUP = 'warmup'
    APPLIED = 'applied'
    GATE_CLOSED = 'gate_closed'
    TERMINAL_FAILURE = 'terminal_failure'
    INCOMPATIBLE_STATE = 'incompatible_state'
    INVALID_COMPUTE = 'invalid_compute'
    UNREADABLE_STATE = 'unreadable_state'


class BudgetEligibilityFailure(str, Enum):
    WARMUP = 'warmup'
    NON_POSITIVE_CURRENT_GAIN = 'non_positive_current_gain'
    NON_POSITIVE_EMA_GAIN = 'non_positive_ema_gain'


_FAIL_CLOSED_REASONS = frozenset(
    {
        BudgetDecisionReason.TERMINAL_FAILURE,
        BudgetDecisionReason.INCOMPATIBLE_STATE,
        BudgetDecisionReason.INVALID_COMPUTE,
        BudgetDecisionReason.UNREADABLE_STATE,
    }
)


@dataclass(frozen=True)
class BudgetCalibrationParameters:
    warmup_completed_generations: int = 30
    sigma_ema_decay: Decimal = Decimal('0.1')
    validation_gain_ema_decay: Decimal = Decimal('0.2')
    initial_tau: Decimal = Decimal('0.1')
    tau_step_ratio: Decimal = Decimal('1.05')
    selection_threshold: Decimal = Decimal('0.8')

    def __post_init__(self) -> None:
        if self.warmup_completed_generations <= 0:
            raise ValueError('Budget warm-up must span a positive number of completed label generations.')
        if not Decimal(0) < self.sigma_ema_decay <= Decimal(1):
            raise ValueError('Sigma EMA decay must lie in (0, 1].')
        if not Decimal(0) < self.validation_gain_ema_decay <= Decimal(1):
            raise ValueError('Validation-gain EMA decay must lie in (0, 1].')
        if self.initial_tau <= Decimal(0):
            raise ValueError('The initial tau threshold must be positive.')
        if self.tau_step_ratio <= Decimal(1):
            raise ValueError('The per-generation tau step ratio must exceed one.')
        if not Decimal(0) < self.selection_threshold < Decimal(1):
            raise ValueError('The selection threshold must lie strictly in (0, 1).')


class BudgetGenerationEvidence(FrozenModel):
    position_count: int = Field(gt=0)
    mean_absolute_curve_error: tuple[float, ...] = Field(
        min_length=BUDGET_CURVE_POINTS,
        max_length=BUDGET_CURVE_POINTS,
    )
    generation_gain: float
    realized_mean_multiple: float = Field(gt=0.0)
    realized_mean_assigned_visits: float = Field(gt=0.0)
    flat_mean_assigned_visits: float = Field(gt=0.0)
    selected_index_counts: tuple[int, ...] = Field(
        min_length=BUDGET_CURVE_POINTS,
        max_length=BUDGET_CURVE_POINTS,
    )

    @model_validator(mode='after')
    def validate_evidence(self) -> BudgetGenerationEvidence:
        if any(not math.isfinite(value) or value < 0.0 for value in self.mean_absolute_curve_error):
            raise ValueError('Curve-error evidence must be finite and nonnegative.')
        if not math.isfinite(self.generation_gain):
            raise ValueError('Generation validation gain must be finite.')
        if not math.isfinite(self.realized_mean_multiple):
            raise ValueError('Realized mean multiple must be finite.')
        if any(count < 0 for count in self.selected_index_counts):
            raise ValueError('Selected-index counts must be nonnegative.')
        if sum(self.selected_index_counts) != self.position_count:
            raise ValueError('Selected-index counts must cover every labelled position exactly once.')
        return self


class BudgetCalibrationState(FrozenModel):
    schema_version: int = Field(default=3, ge=3, le=3)
    configuration_sha256: str = Field(min_length=64, max_length=64)
    finalized_source_generations: tuple[int, ...]
    sigma: tuple[float, ...] = Field(min_length=BUDGET_CURVE_POINTS, max_length=BUDGET_CURVE_POINTS)
    log_tau: float
    current_validation_gain: float | None
    ema_validation_gain: float | None
    realized_mean_multiple: float | None
    previous_published_policy: SearchBudgetPolicy
    published_policy: SearchBudgetPolicy
    application_generation: int = Field(ge=0)
    failed_eligibility_conditions: tuple[BudgetEligibilityFailure, ...]
    decision_reason: BudgetDecisionReason

    @model_validator(mode='after')
    def validate_state(self) -> BudgetCalibrationState:
        if tuple(sorted(set(self.finalized_source_generations))) != self.finalized_source_generations:
            raise ValueError('Finalized source generations must be unique and increasing.')
        if any(not math.isfinite(value) or value <= 0.0 for value in self.sigma):
            raise ValueError('Persisted sigma values must be finite and positive.')
        if not math.isfinite(self.log_tau):
            raise ValueError('Persisted log tau must be finite.')
        if any(
            value is not None and not math.isfinite(value)
            for value in (self.current_validation_gain, self.ema_validation_gain, self.realized_mean_multiple)
        ):
            raise ValueError('Persisted calibration statistics must be finite.')
        return self


class BudgetPolicyPublication(FrozenModel):
    policy: SearchBudgetPolicy
    application_generation: int = Field(ge=0)
    decision_reason: BudgetDecisionReason


@dataclass(frozen=True)
class CalibrationUpdate:
    state: BudgetCalibrationState
    applied: bool


def initial_calibration_state(
    configuration_sha256: str,
    parameters: BudgetCalibrationParameters = BudgetCalibrationParameters(),
) -> BudgetCalibrationState:
    policy = SearchBudgetPolicy(
        sigma=(1.0,) * BUDGET_CURVE_POINTS,
        log_tau=_initial_log_tau(parameters),
        selection_threshold=float(parameters.selection_threshold),
        apply_learned=False,
    )
    return BudgetCalibrationState(
        configuration_sha256=configuration_sha256,
        finalized_source_generations=(),
        sigma=policy.sigma,
        log_tau=policy.log_tau,
        current_validation_gain=None,
        ema_validation_gain=None,
        realized_mean_multiple=None,
        previous_published_policy=policy,
        published_policy=policy,
        application_generation=0,
        failed_eligibility_conditions=(BudgetEligibilityFailure.WARMUP,),
        decision_reason=BudgetDecisionReason.INITIAL,
    )


def working_policy(state: BudgetCalibrationState) -> SearchBudgetPolicy:
    """The learned rule as the calibrator sees it, used for shadow allocation whatever the gate says."""
    return SearchBudgetPolicy(
        sigma=state.sigma,
        log_tau=state.log_tau,
        selection_threshold=state.published_policy.selection_threshold,
        apply_learned=True,
    )


def update_calibration(
    state: BudgetCalibrationState,
    source_generation: int,
    evidence: BudgetGenerationEvidence,
    first_unstarted_production_generation: int,
    parameters: BudgetCalibrationParameters = BudgetCalibrationParameters(),
) -> CalibrationUpdate:
    if source_generation < 0:
        raise ValueError('Source generation must be nonnegative.')
    if first_unstarted_production_generation <= source_generation:
        raise ValueError('Budget-policy publication must apply after its source generation.')
    if source_generation in state.finalized_source_generations:
        return CalibrationUpdate(state=state, applied=False)
    if state.finalized_source_generations and source_generation < state.finalized_source_generations[-1]:
        raise ValueError('Source-generation label jobs must finalize in source order.')

    sigma_decay = float(parameters.sigma_ema_decay)
    sigma = tuple(
        (1.0 - sigma_decay) * previous + sigma_decay * max(error, 1e-9)
        for previous, error in zip(state.sigma, evidence.mean_absolute_curve_error, strict=True)
    )
    maximum_step = math.log(float(parameters.tau_step_ratio))
    tau_step = min(max(math.log(evidence.realized_mean_multiple), -maximum_step), maximum_step)
    log_tau = state.log_tau + tau_step

    current_gain = evidence.generation_gain
    if state.ema_validation_gain is None:
        ema_gain = current_gain
    else:
        gain_decay = float(parameters.validation_gain_ema_decay)
        ema_gain = (1.0 - gain_decay) * state.ema_validation_gain + gain_decay * current_gain

    completed_count = len(state.finalized_source_generations) + 1
    failures = _eligibility_failures(current_gain, ema_gain, completed_count, parameters)
    published = SearchBudgetPolicy(
        sigma=sigma,
        log_tau=log_tau,
        selection_threshold=float(parameters.selection_threshold),
        apply_learned=not failures,
    )
    if not failures:
        decision_reason = BudgetDecisionReason.APPLIED
    elif BudgetEligibilityFailure.WARMUP in failures:
        decision_reason = BudgetDecisionReason.WARMUP
    else:
        decision_reason = BudgetDecisionReason.GATE_CLOSED
    previous_published = published_policy_for_generation(state, max(0, first_unstarted_production_generation - 1))
    next_state = BudgetCalibrationState(
        configuration_sha256=state.configuration_sha256,
        finalized_source_generations=(*state.finalized_source_generations, source_generation),
        sigma=sigma,
        log_tau=log_tau,
        current_validation_gain=current_gain,
        ema_validation_gain=ema_gain,
        realized_mean_multiple=evidence.realized_mean_multiple,
        previous_published_policy=previous_published,
        published_policy=published,
        application_generation=first_unstarted_production_generation,
        failed_eligibility_conditions=failures,
        decision_reason=decision_reason,
    )
    return CalibrationUpdate(state=next_state, applied=True)


def published_policy_for_generation(state: BudgetCalibrationState, production_generation: int) -> SearchBudgetPolicy:
    if production_generation < 0:
        raise ValueError('Production generation must be nonnegative.')
    if production_generation < state.application_generation:
        return state.previous_published_policy
    return state.published_policy


def publish_fail_closed(
    state: BudgetCalibrationState,
    first_unstarted_production_generation: int,
    reason: BudgetDecisionReason,
) -> BudgetCalibrationState:
    if reason not in _FAIL_CLOSED_REASONS:
        raise ValueError('Fail-closed publication requires a failure decision reason.')
    previous_policy = published_policy_for_generation(state, max(0, first_unstarted_production_generation - 1))
    return state.model_copy(
        update={
            'previous_published_policy': previous_policy,
            'published_policy': state.published_policy.model_copy(update={'apply_learned': False}),
            'application_generation': first_unstarted_production_generation,
            'decision_reason': reason,
        }
    )


def save_calibration_state(path: Path, state: BudgetCalibrationState) -> None:
    write_text_atomically(path, state.model_dump_json(indent=2) + '\n')


def load_calibration_state_fail_closed(
    path: Path,
    expected_configuration_sha256: str,
    first_unstarted_production_generation: int,
    parameters: BudgetCalibrationParameters = BudgetCalibrationParameters(),
) -> BudgetCalibrationState:
    try:
        state = BudgetCalibrationState.model_validate_json(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, ValidationError):
        return publish_fail_closed(
            initial_calibration_state(expected_configuration_sha256, parameters),
            first_unstarted_production_generation,
            BudgetDecisionReason.UNREADABLE_STATE,
        )
    if state.configuration_sha256 != expected_configuration_sha256:
        return publish_fail_closed(
            initial_calibration_state(expected_configuration_sha256, parameters),
            first_unstarted_production_generation,
            BudgetDecisionReason.INCOMPATIBLE_STATE,
        )
    return state


def publication_for_generation(state: BudgetCalibrationState, production_generation: int) -> BudgetPolicyPublication:
    return BudgetPolicyPublication(
        policy=published_policy_for_generation(state, production_generation),
        application_generation=state.application_generation,
        decision_reason=state.decision_reason,
    )


def _initial_log_tau(parameters: BudgetCalibrationParameters) -> float:
    return math.log(float(parameters.initial_tau))


def _eligibility_failures(
    current_gain: float,
    ema_gain: float,
    completed_count: int,
    parameters: BudgetCalibrationParameters,
) -> tuple[BudgetEligibilityFailure, ...]:
    failures: list[BudgetEligibilityFailure] = []
    if completed_count < parameters.warmup_completed_generations:
        failures.append(BudgetEligibilityFailure.WARMUP)
    if current_gain <= 0.0:
        failures.append(BudgetEligibilityFailure.NON_POSITIVE_CURRENT_GAIN)
    if ema_gain <= 0.0:
        failures.append(BudgetEligibilityFailure.NON_POSITIVE_EMA_GAIN)
    return tuple(failures)
