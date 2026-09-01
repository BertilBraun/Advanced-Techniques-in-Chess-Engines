from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest
from src.search_stopping.calibration import (
    StopDecisionReason,
    initial_calibration_state,
    load_calibration_state_fail_closed,
    publication_for_generation,
    publish_fail_closed,
    save_calibration_state,
)
from src.search_stopping.configuration import SearchStoppingConfiguration
from src.search_stopping.policy import SearchStopPolicy, closed_policy

CONFIGURATION_SHA = 'a' * 64


def _configuration() -> SearchStoppingConfiguration:
    return SearchStoppingConfiguration(
        audit_sample_fraction=Decimal('0.01'),
        anchor_fraction=Decimal('0.05'),
        anchor_visit_multiple=4.0,
        checkpoint_multiples=(1.0 / 3.0, 0.5, 2.0 / 3.0, 1.0, 1.5),
        cap_multiple=2.0,
        eps_pi_minimum=0.02,
        eps_pi_maximum=0.3,
        eps_v=0.3,
        movement_guard_epsilon=0.05,
        false_stop_rate_ceiling=0.01,
        minimum_evidence_trigger_count=100,
        confidence_level=0.95,
        first_production_generation=10,
        maximum_realized_mean_spend=1.3,
        window_generations=10,
        maximum_unstarted_generation_lag=2,
    )


def test_initial_state_publishes_the_closed_policy() -> None:
    state = initial_calibration_state(_configuration(), CONFIGURATION_SHA)
    publication = publication_for_generation(state, 0)
    assert not publication.policy.apply_learned
    assert publication.decision_reason is StopDecisionReason.INITIAL


def test_closed_policy_has_no_predictor_and_no_learned_application() -> None:
    policy = closed_policy(_configuration())
    assert not policy.apply_learned and policy.predictor_path is None


def test_an_applied_policy_requires_a_predictor_reference() -> None:
    with pytest.raises(ValueError, match='requires a published predictor'):
        SearchStopPolicy(
            checkpoint_multiples=(0.5, 1.0),
            thresholds=(0.1, 0.2),
            movement_guard_epsilon=0.05,
            cap_multiple=2.0,
            predictor_path=None,
            predictor_sha256=None,
            apply_learned=True,
        )


def test_fail_closed_keeps_the_running_generation_policy() -> None:
    configuration = _configuration()
    state = initial_calibration_state(configuration, CONFIGURATION_SHA)
    failed = publish_fail_closed(state, configuration, 12, StopDecisionReason.TERMINAL_FAILURE)
    assert failed.application_generation == 12
    assert publication_for_generation(failed, 11).policy == state.published_policy
    assert not publication_for_generation(failed, 12).policy.apply_learned


def test_fail_closed_rejects_ordinary_decision_reasons() -> None:
    configuration = _configuration()
    state = initial_calibration_state(configuration, CONFIGURATION_SHA)
    with pytest.raises(ValueError, match='structural failure'):
        publish_fail_closed(state, configuration, 5, StopDecisionReason.SPEND_BREAKER)


def test_unreadable_state_fails_closed(tmp_path: Path) -> None:
    configuration = _configuration()
    path = tmp_path / 'stop-calibration.json'
    path.write_text('{not json', encoding='utf-8')
    state = load_calibration_state_fail_closed(path, configuration, CONFIGURATION_SHA, 7)
    assert state.decision_reason is StopDecisionReason.UNREADABLE_STATE
    assert not state.published_policy.apply_learned


def test_configuration_sha_mismatch_fails_closed(tmp_path: Path) -> None:
    configuration = _configuration()
    path = tmp_path / 'stop-calibration.json'
    save_calibration_state(path, initial_calibration_state(configuration, 'b' * 64))
    state = load_calibration_state_fail_closed(path, configuration, CONFIGURATION_SHA, 7)
    assert state.decision_reason is StopDecisionReason.INCOMPATIBLE_STATE


def test_missing_predictor_artifact_fails_closed(tmp_path: Path) -> None:
    configuration = _configuration()
    state = initial_calibration_state(configuration, CONFIGURATION_SHA)
    dangling = state.model_copy(update={'predictor_path': tmp_path / 'missing.jit.pt', 'predictor_sha256': 'c' * 64})
    path = tmp_path / 'stop-calibration.json'
    save_calibration_state(path, dangling)
    state = load_calibration_state_fail_closed(path, configuration, CONFIGURATION_SHA, 7)
    assert state.decision_reason is StopDecisionReason.UNREADABLE_STATE


def test_intact_state_round_trips(tmp_path: Path) -> None:
    configuration = _configuration()
    original = initial_calibration_state(configuration, CONFIGURATION_SHA)
    path = tmp_path / 'stop-calibration.json'
    save_calibration_state(path, original)
    assert load_calibration_state_fail_closed(path, configuration, CONFIGURATION_SHA, 7) == original
