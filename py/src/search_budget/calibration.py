from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from pathlib import Path

from pydantic import Field, ValidationError, model_validator
from src.search_budget.policy import (
    BUDGET_CURVE_MULTIPLES,
    BUDGET_CURVE_POINTS,
    SearchBudgetPolicy,
    project_non_increasing,
)
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
    UNCONVERGED_SPEND = 'unconverged_spend'


@dataclass(frozen=True)
class CorrectorReference:
    path: Path | None
    sha256: str | None

    def __post_init__(self) -> None:
        if (self.path is None) != (self.sha256 is None):
            raise ValueError('A corrector reference requires both its path and its digest.')


IDENTITY_CORRECTOR_REFERENCE = CorrectorReference(path=None, sha256=None)

_MINIMUM_TRACKED_LAGRANGE_MULTIPLIER = 1e-12


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
    spend_ema_decay: Decimal = Decimal('0.2')
    lambda_trust_ratio: Decimal = Decimal('2.0')
    lambda_reseed_ratio: Decimal = Decimal('100.0')

    def __post_init__(self) -> None:
        if self.warmup_completed_generations <= 0:
            raise ValueError('Budget warm-up must span a positive number of completed label generations.')
        if not Decimal(0) < self.sigma_ema_decay <= Decimal(1):
            raise ValueError('Sigma EMA decay must lie in (0, 1].')
        if not Decimal(0) < self.validation_gain_ema_decay <= Decimal(1):
            raise ValueError('Validation-gain EMA decay must lie in (0, 1].')
        if not Decimal(0) < self.spend_ema_decay <= Decimal(1):
            raise ValueError('Spend EMA decay must lie in (0, 1].')
        if self.lambda_trust_ratio <= Decimal(1):
            raise ValueError('The per-generation lambda trust ratio must exceed one.')
        if self.lambda_reseed_ratio <= self.lambda_trust_ratio:
            raise ValueError('The lambda reseed ratio must exceed the trust ratio.')


class BudgetGenerationEvidence(FrozenModel):
    position_count: int = Field(gt=0)
    mean_absolute_curve_error: tuple[float, ...] = Field(
        min_length=BUDGET_CURVE_POINTS,
        max_length=BUDGET_CURVE_POINTS,
    )
    generation_gain: float
    target_raw_kl_curves: tuple[tuple[float, ...], ...] = Field(min_length=1)
    selection_raw_kl_curves: tuple[tuple[float, ...], ...] = Field(min_length=1)
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
        for curves, subject in (
            (self.target_raw_kl_curves, 'Measured'),
            (self.selection_raw_kl_curves, 'Selection'),
        ):
            if len(curves) != self.position_count:
                raise ValueError(f'{subject} raw-KL curves must cover every labelled position exactly once.')
            for curve in curves:
                if len(curve) != BUDGET_CURVE_POINTS:
                    raise ValueError(f'Every {subject.lower()} raw-KL curve requires one value per grid point.')
                if any(not math.isfinite(value) or value < 0.0 for value in curve):
                    raise ValueError(f'{subject} raw-KL curves must be finite and nonnegative.')
        if not math.isfinite(self.realized_mean_multiple):
            raise ValueError('Realized mean multiple must be finite.')
        if any(count < 0 for count in self.selected_index_counts):
            raise ValueError('Selected-index counts must be nonnegative.')
        if sum(self.selected_index_counts) != self.position_count:
            raise ValueError('Selected-index counts must cover every labelled position exactly once.')
        return self


class BudgetCalibrationState(FrozenModel):
    schema_version: int = Field(default=6, ge=6, le=6)
    configuration_sha256: str = Field(min_length=64, max_length=64)
    finalized_source_generations: tuple[int, ...]
    sigma: tuple[float, ...] = Field(min_length=BUDGET_CURVE_POINTS, max_length=BUDGET_CURVE_POINTS)
    lagrange_multiplier: float
    corrector_path: Path | None
    corrector_sha256: str | None
    current_validation_gain: float | None
    ema_validation_gain: float | None
    realized_mean_multiple: float | None
    ema_realized_mean_multiple: float | None
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
        if not math.isfinite(self.lagrange_multiplier) or self.lagrange_multiplier < 0.0:
            raise ValueError('The persisted Lagrange multiplier must be finite and nonnegative.')
        if (self.corrector_path is None) != (self.corrector_sha256 is None):
            raise ValueError('A persisted corrector reference requires both its path and its digest.')
        if any(
            value is not None and not math.isfinite(value)
            for value in (
                self.current_validation_gain,
                self.ema_validation_gain,
                self.realized_mean_multiple,
                self.ema_realized_mean_multiple,
            )
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


def initial_calibration_state(configuration_sha256: str) -> BudgetCalibrationState:
    policy = SearchBudgetPolicy(
        lagrange_multiplier=0.0,
        corrector_path=None,
        corrector_sha256=None,
        apply_learned=False,
    )
    return BudgetCalibrationState(
        configuration_sha256=configuration_sha256,
        finalized_source_generations=(),
        sigma=(1.0,) * BUDGET_CURVE_POINTS,
        lagrange_multiplier=0.0,
        corrector_path=None,
        corrector_sha256=None,
        current_validation_gain=None,
        ema_validation_gain=None,
        realized_mean_multiple=None,
        ema_realized_mean_multiple=None,
        previous_published_policy=policy,
        published_policy=policy,
        application_generation=0,
        failed_eligibility_conditions=(BudgetEligibilityFailure.WARMUP,),
        decision_reason=BudgetDecisionReason.INITIAL,
    )


def working_policy(state: BudgetCalibrationState) -> SearchBudgetPolicy:
    """The learned rule as the calibration loop sees it, used for shadow allocation whatever the gate says."""
    return SearchBudgetPolicy(
        lagrange_multiplier=state.lagrange_multiplier,
        corrector_path=state.corrector_path,
        corrector_sha256=state.corrector_sha256,
        apply_learned=True,
    )


def solve_spend_matched_lagrange_multiplier(
    raw_kl_curves: tuple[tuple[float, ...], ...],
    bisection_iterations: int = 60,
) -> float:
    """The dual value whose Lagrangian selection spends a mean multiple of one on the given curves.

    Solve on the curves selection actually sees. Measured curves carry exact plateaus wherever the
    checkpointed visit distribution stops changing, and ties resolve to the cheaper index, so a dual
    solved on them sits orders of magnitude below the one the smooth predicted curves need.
    """
    if not raw_kl_curves:
        raise ValueError('Lagrange solving requires at least one curve.')
    projected_curves = tuple(project_non_increasing(curve) for curve in raw_kl_curves)

    def mean_multiple(multiplier: float) -> float:
        total = 0.0
        for curve in projected_curves:
            best_index = 0
            best_objective = math.inf
            for index in range(BUDGET_CURVE_POINTS):
                objective = curve[index] + multiplier * BUDGET_CURVE_MULTIPLES[index]
                if objective < best_objective:
                    best_objective = objective
                    best_index = index
            total += BUDGET_CURVE_MULTIPLES[best_index]
        return total / len(projected_curves)

    low = 0.0
    if mean_multiple(low) <= 1.0:
        return low
    high = 1.0
    for _ in range(bisection_iterations):
        if mean_multiple(high) <= 1.0:
            break
        high *= 2.0
    for _ in range(bisection_iterations):
        midpoint = 0.5 * (low + high)
        if mean_multiple(midpoint) > 1.0:
            low = midpoint
        else:
            high = midpoint
    return high


def _tracked_lagrange_multiplier(
    state: BudgetCalibrationState,
    evidence: BudgetGenerationEvidence,
    parameters: BudgetCalibrationParameters,
) -> float:
    """Re-solve the dual every generation against the curves selection sees, inside a trust region.

    A bounded per-generation step cannot cross the decades between a dual solved on a random network
    and the one a trained network needs, so a value that far from its solution is replaced outright
    rather than walked toward.
    """
    solved = solve_spend_matched_lagrange_multiplier(evidence.selection_raw_kl_curves)
    if not state.finalized_source_generations:
        return solved
    previous = max(state.lagrange_multiplier, _MINIMUM_TRACKED_LAGRANGE_MULTIPLIER)
    reseed_ratio = float(parameters.lambda_reseed_ratio)
    if not previous / reseed_ratio <= max(solved, _MINIMUM_TRACKED_LAGRANGE_MULTIPLIER) <= previous * reseed_ratio:
        return solved
    trust_ratio = float(parameters.lambda_trust_ratio)
    return min(max(solved, previous / trust_ratio), previous * trust_ratio)


def update_calibration(
    state: BudgetCalibrationState,
    source_generation: int,
    evidence: BudgetGenerationEvidence,
    first_unstarted_production_generation: int,
    parameters: BudgetCalibrationParameters = BudgetCalibrationParameters(),
    corrector: CorrectorReference = IDENTITY_CORRECTOR_REFERENCE,
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
    lagrange_multiplier = _tracked_lagrange_multiplier(state, evidence, parameters)

    current_gain = evidence.generation_gain
    if state.ema_validation_gain is None:
        ema_gain = current_gain
    else:
        gain_decay = float(parameters.validation_gain_ema_decay)
        ema_gain = (1.0 - gain_decay) * state.ema_validation_gain + gain_decay * current_gain

    if state.ema_realized_mean_multiple is None:
        ema_multiple = evidence.realized_mean_multiple
    else:
        spend_decay = float(parameters.spend_ema_decay)
        ema_multiple = (
            1.0 - spend_decay
        ) * state.ema_realized_mean_multiple + spend_decay * evidence.realized_mean_multiple

    completed_count = len(state.finalized_source_generations) + 1
    failures = _eligibility_failures(current_gain, ema_gain, completed_count, ema_multiple, parameters)
    published = SearchBudgetPolicy(
        lagrange_multiplier=lagrange_multiplier,
        corrector_path=corrector.path,
        corrector_sha256=corrector.sha256,
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
        lagrange_multiplier=lagrange_multiplier,
        corrector_path=corrector.path,
        corrector_sha256=corrector.sha256,
        current_validation_gain=current_gain,
        ema_validation_gain=ema_gain,
        realized_mean_multiple=evidence.realized_mean_multiple,
        ema_realized_mean_multiple=ema_multiple,
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
) -> BudgetCalibrationState:
    try:
        state = BudgetCalibrationState.model_validate_json(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, ValidationError):
        return publish_fail_closed(
            initial_calibration_state(expected_configuration_sha256),
            first_unstarted_production_generation,
            BudgetDecisionReason.UNREADABLE_STATE,
        )
    if state.configuration_sha256 != expected_configuration_sha256:
        return publish_fail_closed(
            initial_calibration_state(expected_configuration_sha256),
            first_unstarted_production_generation,
            BudgetDecisionReason.INCOMPATIBLE_STATE,
        )
    # A dangling corrector reference would crash native policy loading at the next generation
    # start, so a state whose corrector artifact is gone or altered fails closed instead.
    for path, sha256 in (
        (state.corrector_path, state.corrector_sha256),
        (state.published_policy.corrector_path, state.published_policy.corrector_sha256),
        (state.previous_published_policy.corrector_path, state.previous_published_policy.corrector_sha256),
    ):
        if path is None:
            continue
        try:
            content = path.read_bytes()
        except OSError:
            content = None
        if content is None or hashlib.sha256(content).hexdigest() != sha256:
            return publish_fail_closed(
                initial_calibration_state(expected_configuration_sha256),
                first_unstarted_production_generation,
                BudgetDecisionReason.UNREADABLE_STATE,
            )
    return state


def publication_for_generation(state: BudgetCalibrationState, production_generation: int) -> BudgetPolicyPublication:
    return BudgetPolicyPublication(
        policy=published_policy_for_generation(state, production_generation),
        application_generation=state.application_generation,
        decision_reason=state.decision_reason,
    )


SPEND_CONVERGENCE_BAND = (0.95, 1.05)


def _eligibility_failures(
    current_gain: float,
    ema_gain: float,
    completed_count: int,
    ema_realized_mean_multiple: float,
    parameters: BudgetCalibrationParameters,
) -> tuple[BudgetEligibilityFailure, ...]:
    failures: list[BudgetEligibilityFailure] = []
    if completed_count < parameters.warmup_completed_generations:
        failures.append(BudgetEligibilityFailure.WARMUP)
    if current_gain <= 0.0:
        failures.append(BudgetEligibilityFailure.NON_POSITIVE_CURRENT_GAIN)
    if ema_gain <= 0.0:
        failures.append(BudgetEligibilityFailure.NON_POSITIVE_EMA_GAIN)
    # Selection for a generation runs on the previous generation's dual, so instantaneous spend
    # oscillates around the target even when the dual is right; the constraint is on the average.
    low, high = SPEND_CONVERGENCE_BAND
    if not low <= ema_realized_mean_multiple <= high:
        failures.append(BudgetEligibilityFailure.UNCONVERGED_SPEND)
    return tuple(failures)
