from __future__ import annotations

import hashlib
from enum import Enum
from pathlib import Path
from typing import TypeVar

from pydantic import Field, ValidationError
from src.search_stopping.configuration import SearchStoppingConfiguration
from src.search_stopping.policy import SearchStopPolicy, closed_policy
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel


class StopDecisionReason(str, Enum):
    INITIAL = 'initial'
    WARMUP = 'warmup'
    APPLIED = 'applied'
    NO_SAFE_THRESHOLD = 'no_safe_threshold'
    PREDICTOR_REJECTED = 'predictor_rejected'
    SPEND_BREAKER = 'spend_breaker'
    TERMINAL_FAILURE = 'terminal_failure'
    INCOMPATIBLE_STATE = 'incompatible_state'
    UNREADABLE_STATE = 'unreadable_state'


# Structural failures publish the closed policy; NO_SAFE_THRESHOLD / PREDICTOR_REJECTED /
# SPEND_BREAKER also close, but through the ordinary calibration decision, not this path.
_FAIL_CLOSED_REASONS = frozenset(
    {
        StopDecisionReason.TERMINAL_FAILURE,
        StopDecisionReason.INCOMPATIBLE_STATE,
        StopDecisionReason.UNREADABLE_STATE,
    }
)


class StopCalibrationState(FrozenModel):
    schema_version: int = Field(default=1, ge=1, le=1)
    configuration_sha256: str = Field(min_length=64, max_length=64)
    finalized_source_generations: tuple[int, ...]
    solved_eps_pi: float | None
    eps_saturated_at_maximum: bool
    predictor_path: Path | None
    predictor_sha256: str | None
    previous_published_policy: SearchStopPolicy
    published_policy: SearchStopPolicy
    application_generation: int = Field(ge=0)
    decision_reason: StopDecisionReason

    def model_post_init(self, __context: object) -> None:
        if tuple(sorted(set(self.finalized_source_generations))) != self.finalized_source_generations:
            raise ValueError('Finalized source generations must be unique and increasing.')
        if (self.predictor_path is None) != (self.predictor_sha256 is None):
            raise ValueError('A persisted predictor reference requires both its path and its digest.')


class StopPolicyPublication(FrozenModel):
    policy: SearchStopPolicy
    application_generation: int = Field(ge=0)
    decision_reason: StopDecisionReason


def initial_calibration_state(
    configuration: SearchStoppingConfiguration,
    configuration_sha256: str,
) -> StopCalibrationState:
    policy = closed_policy(configuration)
    return StopCalibrationState(
        configuration_sha256=configuration_sha256,
        finalized_source_generations=(),
        solved_eps_pi=None,
        eps_saturated_at_maximum=False,
        predictor_path=None,
        predictor_sha256=None,
        previous_published_policy=policy,
        published_policy=policy,
        application_generation=0,
        decision_reason=StopDecisionReason.INITIAL,
    )


def published_policy_for_generation(state: StopCalibrationState, production_generation: int) -> SearchStopPolicy:
    if production_generation < 0:
        raise ValueError('Production generation must be nonnegative.')
    if production_generation < state.application_generation:
        return state.previous_published_policy
    return state.published_policy


def publication_for_generation(state: StopCalibrationState, production_generation: int) -> StopPolicyPublication:
    return StopPolicyPublication(
        policy=published_policy_for_generation(state, production_generation),
        application_generation=state.application_generation,
        decision_reason=state.decision_reason,
    )


def publish_fail_closed(
    state: StopCalibrationState,
    configuration: SearchStoppingConfiguration,
    first_unstarted_production_generation: int,
    reason: StopDecisionReason,
) -> StopCalibrationState:
    if reason not in _FAIL_CLOSED_REASONS:
        raise ValueError('Fail-closed publication requires a structural failure decision reason.')
    previous_policy = published_policy_for_generation(state, max(0, first_unstarted_production_generation - 1))
    return state.model_copy(
        update={
            'previous_published_policy': previous_policy,
            'published_policy': closed_policy(configuration),
            'application_generation': first_unstarted_production_generation,
            'decision_reason': reason,
        }
    )


def save_calibration_state(path: Path, state: StopCalibrationState) -> None:
    write_text_atomically(path, state.model_dump_json(indent=2) + '\n')


def load_calibration_state_fail_closed(
    path: Path,
    configuration: SearchStoppingConfiguration,
    expected_configuration_sha256: str,
    first_unstarted_production_generation: int,
) -> StopCalibrationState:
    try:
        state = StopCalibrationState.model_validate_json(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, ValidationError):
        return publish_fail_closed(
            initial_calibration_state(configuration, expected_configuration_sha256),
            configuration,
            first_unstarted_production_generation,
            StopDecisionReason.UNREADABLE_STATE,
        )
    if state.configuration_sha256 != expected_configuration_sha256:
        return publish_fail_closed(
            initial_calibration_state(configuration, expected_configuration_sha256),
            configuration,
            first_unstarted_production_generation,
            StopDecisionReason.INCOMPATIBLE_STATE,
        )
    # A dangling predictor reference would crash native policy loading at the next generation
    # start, so a state whose predictor artifact is gone or altered fails closed instead.
    for reference_path, reference_sha256 in (
        (state.predictor_path, state.predictor_sha256),
        (state.published_policy.predictor_path, state.published_policy.predictor_sha256),
        (state.previous_published_policy.predictor_path, state.previous_published_policy.predictor_sha256),
    ):
        if reference_path is None:
            continue
        try:
            content = reference_path.read_bytes()
        except OSError:
            content = None
        if content is None or hashlib.sha256(content).hexdigest() != reference_sha256:
            return publish_fail_closed(
                initial_calibration_state(configuration, expected_configuration_sha256),
                configuration,
                first_unstarted_production_generation,
                StopDecisionReason.UNREADABLE_STATE,
            )
    return state


PersistedModelT = TypeVar('PersistedModelT', bound=FrozenModel)


def write_persisted_model(path: Path, model: FrozenModel) -> None:
    write_text_atomically(path, model.model_dump_json(indent=2) + '\n')


def load_persisted_model(path: Path, model_type: type[PersistedModelT]) -> PersistedModelT:
    return model_type.model_validate_json(path.read_text(encoding='utf-8'))
