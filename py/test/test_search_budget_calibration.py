from __future__ import annotations

import math
from decimal import Decimal
from pathlib import Path

import pytest
from src.search_budget.calibration import (
    BudgetCalibrationParameters,
    BudgetCalibrationState,
    BudgetDecisionReason,
    BudgetEligibilityFailure,
    BudgetGenerationEvidence,
    initial_calibration_state,
    load_calibration_state_fail_closed,
    publication_for_generation,
    publish_fail_closed,
    save_calibration_state,
    update_calibration,
    working_policy,
)
from src.search_budget.policy import BUDGET_CURVE_POINTS

CONFIGURATION_SHA = 'c' * 64


def evidence(
    gain: float = 0.05,
    realized_mean_multiple: float = 1.0,
    errors: tuple[float, ...] = (0.5,) * BUDGET_CURVE_POINTS,
) -> BudgetGenerationEvidence:
    return BudgetGenerationEvidence(
        position_count=4,
        mean_absolute_curve_error=errors,
        generation_gain=gain,
        realized_mean_multiple=realized_mean_multiple,
        realized_mean_assigned_visits=600.0 * realized_mean_multiple,
        flat_mean_assigned_visits=600.0,
        selected_index_counts=(4, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    )


def parameters(warmup: int = 2) -> BudgetCalibrationParameters:
    return BudgetCalibrationParameters(warmup_completed_generations=warmup)


def completed_state(generations: int, gain: float = 0.05, warmup: int = 2) -> BudgetCalibrationState:
    state = initial_calibration_state(CONFIGURATION_SHA)
    for generation in range(generations):
        state = update_calibration(state, generation, evidence(gain), generation + 1, parameters(warmup)).state
    return state


def test_initial_state_publishes_flat_with_unit_sigma_and_configured_tau() -> None:
    state = initial_calibration_state(CONFIGURATION_SHA)
    assert not state.published_policy.apply_learned
    assert state.sigma == (1.0,) * BUDGET_CURVE_POINTS
    assert state.log_tau == pytest.approx(math.log(0.1))
    assert state.decision_reason is BudgetDecisionReason.INITIAL


def test_working_policy_applies_the_learned_rule_regardless_of_the_gate() -> None:
    state = initial_calibration_state(CONFIGURATION_SHA)
    policy = working_policy(state)
    assert policy.apply_learned
    assert policy.sigma == state.sigma
    assert policy.log_tau == state.log_tau


def test_sigma_updates_with_ema_decay_from_unit_initialisation() -> None:
    state = initial_calibration_state(CONFIGURATION_SHA)
    updated = update_calibration(state, 0, evidence(errors=(2.0,) * 10), 1, parameters()).state
    assert updated.sigma == pytest.approx((0.9 * 1.0 + 0.1 * 2.0,) * 10)


def test_log_tau_moves_toward_lower_spend_when_realized_multiple_is_high() -> None:
    state = initial_calibration_state(CONFIGURATION_SHA)
    overspend = update_calibration(state, 0, evidence(realized_mean_multiple=1.02), 1, parameters()).state
    assert overspend.log_tau == pytest.approx(state.log_tau + math.log(1.02))


def test_log_tau_step_is_bounded_to_the_configured_ratio() -> None:
    state = initial_calibration_state(CONFIGURATION_SHA)
    surge = update_calibration(state, 0, evidence(realized_mean_multiple=4.0), 1, parameters()).state
    collapse = update_calibration(state, 0, evidence(realized_mean_multiple=0.2), 1, parameters()).state
    assert surge.log_tau == pytest.approx(state.log_tau + math.log(1.05))
    assert collapse.log_tau == pytest.approx(state.log_tau - math.log(1.05))


def test_warmup_keeps_the_gate_closed_while_calibration_progresses() -> None:
    state = completed_state(1, warmup=5)
    assert not state.published_policy.apply_learned
    assert state.decision_reason is BudgetDecisionReason.WARMUP
    assert BudgetEligibilityFailure.WARMUP in state.failed_eligibility_conditions
    assert state.current_validation_gain == pytest.approx(0.05)


def test_gate_opens_after_warmup_with_positive_current_and_ema_gain() -> None:
    state = completed_state(2)
    assert state.published_policy.apply_learned
    assert state.decision_reason is BudgetDecisionReason.APPLIED
    assert state.failed_eligibility_conditions == ()


def test_negative_current_gain_closes_the_gate_to_flat() -> None:
    state = completed_state(2)
    closed = update_calibration(state, 2, evidence(gain=-0.05), 3, parameters()).state
    assert not closed.published_policy.apply_learned
    assert closed.decision_reason is BudgetDecisionReason.GATE_CLOSED
    assert BudgetEligibilityFailure.NON_POSITIVE_CURRENT_GAIN in closed.failed_eligibility_conditions


def test_ema_gain_uses_configured_decay_and_gates_independently() -> None:
    state = completed_state(2, gain=0.1)
    dipped = update_calibration(state, 2, evidence(gain=-0.01), 3, parameters()).state
    assert dipped.ema_validation_gain == pytest.approx(0.8 * state.ema_validation_gain + 0.2 * -0.01)
    assert BudgetEligibilityFailure.NON_POSITIVE_CURRENT_GAIN in dipped.failed_eligibility_conditions
    assert BudgetEligibilityFailure.NON_POSITIVE_EMA_GAIN not in dipped.failed_eligibility_conditions


def test_publication_applies_only_from_its_application_generation() -> None:
    state = completed_state(2)
    later = update_calibration(state, 2, evidence(gain=-0.05), 7, parameters()).state
    assert publication_for_generation(later, 6).policy == state.published_policy
    assert publication_for_generation(later, 7).policy == later.published_policy


def test_reprocessing_a_finalized_generation_is_idempotent() -> None:
    state = completed_state(2)
    repeated = update_calibration(state, 1, evidence(), 3, parameters())
    assert not repeated.applied
    assert repeated.state == state


def test_out_of_order_finalization_is_rejected() -> None:
    state = initial_calibration_state(CONFIGURATION_SHA)
    state = update_calibration(state, 0, evidence(), 1, parameters()).state
    state = update_calibration(state, 2, evidence(), 3, parameters()).state
    with pytest.raises(ValueError, match='source order'):
        update_calibration(state, 1, evidence(), 3, parameters())
    with pytest.raises(ValueError, match='after its source generation'):
        update_calibration(state, 5, evidence(), 5, parameters())


def test_fail_closed_disables_the_learned_rule_but_keeps_calibration() -> None:
    state = completed_state(2)
    failed = publish_fail_closed(state, 5, BudgetDecisionReason.TERMINAL_FAILURE)
    assert not failed.published_policy.apply_learned
    assert failed.sigma == state.sigma
    assert failed.log_tau == state.log_tau
    assert failed.decision_reason is BudgetDecisionReason.TERMINAL_FAILURE
    with pytest.raises(ValueError, match='failure decision reason'):
        publish_fail_closed(state, 5, BudgetDecisionReason.APPLIED)


def test_calibration_state_round_trips_through_disk(tmp_path: Path) -> None:
    state = completed_state(2)
    path = tmp_path / 'calibration-state.json'
    save_calibration_state(path, state)
    loaded = load_calibration_state_fail_closed(path, CONFIGURATION_SHA, 3)
    assert loaded == state


def test_unreadable_state_fails_closed_to_flat(tmp_path: Path) -> None:
    path = tmp_path / 'calibration-state.json'
    path.write_text('{not json', encoding='utf-8')
    loaded = load_calibration_state_fail_closed(path, CONFIGURATION_SHA, 3)
    assert not loaded.published_policy.apply_learned
    assert loaded.decision_reason is BudgetDecisionReason.UNREADABLE_STATE


def test_configuration_digest_mismatch_fails_closed(tmp_path: Path) -> None:
    state = completed_state(2)
    path = tmp_path / 'calibration-state.json'
    save_calibration_state(path, state)
    loaded = load_calibration_state_fail_closed(path, 'd' * 64, 3)
    assert not loaded.published_policy.apply_learned
    assert loaded.decision_reason is BudgetDecisionReason.INCOMPATIBLE_STATE


def test_parameters_reject_invalid_configuration() -> None:
    with pytest.raises(ValueError, match='tau step ratio'):
        BudgetCalibrationParameters(tau_step_ratio=Decimal(1))
    with pytest.raises(ValueError, match='initial tau'):
        BudgetCalibrationParameters(initial_tau=Decimal(0))
    with pytest.raises(ValueError, match='selection threshold'):
        BudgetCalibrationParameters(selection_threshold=Decimal(1))
