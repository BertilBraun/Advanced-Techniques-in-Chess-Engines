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
    CorrectorReference,
    initial_calibration_state,
    load_calibration_state_fail_closed,
    publication_for_generation,
    publish_fail_closed,
    save_calibration_state,
    solve_spend_matched_lagrange_multiplier,
    update_calibration,
    working_policy,
)
from src.search_budget.policy import BUDGET_CURVE_MULTIPLES, BUDGET_CURVE_POINTS

CONFIGURATION_SHA = 'c' * 64

STEEP_CURVE = tuple(float(10 - index) for index in range(BUDGET_CURVE_POINTS))
FLAT_CURVE = (0.5,) * BUDGET_CURVE_POINTS


def evidence(
    gain: float = 0.05,
    realized_mean_multiple: float = 1.0,
    errors: tuple[float, ...] = (0.5,) * BUDGET_CURVE_POINTS,
    curves: tuple[tuple[float, ...], ...] = (STEEP_CURVE, STEEP_CURVE, FLAT_CURVE, FLAT_CURVE),
    selection_curves: tuple[tuple[float, ...], ...] | None = None,
) -> BudgetGenerationEvidence:
    return BudgetGenerationEvidence(
        position_count=len(curves),
        mean_absolute_curve_error=errors,
        generation_gain=gain,
        target_raw_kl_curves=curves,
        selection_raw_kl_curves=curves if selection_curves is None else selection_curves,
        realized_mean_multiple=realized_mean_multiple,
        realized_mean_assigned_visits=600.0 * realized_mean_multiple,
        flat_mean_assigned_visits=600.0,
        selected_index_counts=(len(curves), 0, 0, 0, 0, 0, 0, 0),
    )


def parameters(warmup: int = 2) -> BudgetCalibrationParameters:
    return BudgetCalibrationParameters(warmup_completed_generations=warmup)


def completed_state(generations: int, gain: float = 0.05, warmup: int = 2) -> BudgetCalibrationState:
    state = initial_calibration_state(CONFIGURATION_SHA)
    for generation in range(generations):
        state = update_calibration(state, generation, evidence(gain), generation + 1, parameters(warmup)).state
    return state


def mean_multiple_at(curves: tuple[tuple[float, ...], ...], multiplier: float) -> float:
    total = 0.0
    for curve in curves:
        projected = list(curve)
        for index in range(1, BUDGET_CURVE_POINTS):
            projected[index] = min(projected[index], projected[index - 1])
        objectives = tuple(
            value + multiplier * multiple for value, multiple in zip(projected, BUDGET_CURVE_MULTIPLES, strict=True)
        )
        total += BUDGET_CURVE_MULTIPLES[min(range(BUDGET_CURVE_POINTS), key=objectives.__getitem__)]
    return total / len(curves)


def test_initial_state_publishes_flat_with_unit_sigma_and_identity_correction() -> None:
    state = initial_calibration_state(CONFIGURATION_SHA)
    assert not state.published_policy.apply_learned
    assert state.sigma == (1.0,) * BUDGET_CURVE_POINTS
    assert state.lagrange_multiplier == 0.0
    assert state.corrector_path is None
    assert state.corrector_sha256 is None
    assert state.decision_reason is BudgetDecisionReason.INITIAL


def test_working_policy_applies_the_learned_rule_regardless_of_the_gate() -> None:
    state = initial_calibration_state(CONFIGURATION_SHA)
    policy = working_policy(state)
    assert policy.apply_learned
    assert policy.lagrange_multiplier == state.lagrange_multiplier
    assert policy.corrector_path == state.corrector_path
    assert policy.corrector_sha256 == state.corrector_sha256


def test_sigma_updates_with_ema_decay_from_unit_initialisation() -> None:
    state = initial_calibration_state(CONFIGURATION_SHA)
    updated = update_calibration(state, 0, evidence(errors=(2.0,) * BUDGET_CURVE_POINTS), 1, parameters()).state
    assert updated.sigma == pytest.approx((0.9 * 1.0 + 0.1 * 2.0,) * BUDGET_CURVE_POINTS)


def test_lambda_seeding_bisects_to_the_boundary_of_unit_mean_spend() -> None:
    curves = (STEEP_CURVE, STEEP_CURVE, FLAT_CURVE, FLAT_CURVE)
    seeded = solve_spend_matched_lagrange_multiplier(curves)
    assert seeded > 0.0
    assert mean_multiple_at(curves, seeded) <= 1.0
    assert mean_multiple_at(curves, seeded * 0.99) > 1.0


def test_lambda_seeding_returns_zero_when_even_free_spend_stays_at_baseline() -> None:
    # A flat measured curve gains nothing from any spend, so every position takes the cheapest
    # grid point already at a zero dual.
    assert solve_spend_matched_lagrange_multiplier((FLAT_CURVE, FLAT_CURVE)) == 0.0


def test_the_first_generation_seeds_lambda_from_the_selection_curves() -> None:
    state = initial_calibration_state(CONFIGURATION_SHA)
    selection = (STEEP_CURVE, STEEP_CURVE)
    seeded = update_calibration(
        state, 0, evidence(curves=(FLAT_CURVE, FLAT_CURVE), selection_curves=selection), 1, parameters()
    ).state
    assert seeded.lagrange_multiplier == pytest.approx(solve_spend_matched_lagrange_multiplier(selection))


def test_lambda_ignores_the_measured_curves_when_selection_sees_different_ones() -> None:
    state = initial_calibration_state(CONFIGURATION_SHA)
    seeded = update_calibration(
        state,
        0,
        evidence(curves=(FLAT_CURVE, FLAT_CURVE), selection_curves=(STEEP_CURVE, STEEP_CURVE)),
        1,
        parameters(),
    ).state
    assert seeded.lagrange_multiplier == pytest.approx(
        solve_spend_matched_lagrange_multiplier((STEEP_CURVE, STEEP_CURVE))
    )


def test_lambda_tracks_the_solved_value_within_the_trust_region() -> None:
    seeded = update_calibration(initial_calibration_state(CONFIGURATION_SHA), 0, evidence(), 1, parameters()).state
    tracked = update_calibration(seeded, 1, evidence(), 2, parameters()).state
    assert tracked.lagrange_multiplier == pytest.approx(seeded.lagrange_multiplier)


def test_lambda_movement_is_bounded_to_the_trust_ratio() -> None:
    state = initial_calibration_state(CONFIGURATION_SHA)
    seeded = update_calibration(state, 0, evidence(), 1, parameters()).state
    nudged = tuple(tuple(value * 8.0 for value in curve) for curve in evidence().selection_raw_kl_curves)
    stepped = update_calibration(seeded, 1, evidence(selection_curves=nudged), 2, parameters()).state
    assert stepped.lagrange_multiplier == pytest.approx(seeded.lagrange_multiplier * 2.0)


def test_a_stale_lambda_reseeds_instead_of_crawling_toward_its_solution() -> None:
    state = initial_calibration_state(CONFIGURATION_SHA)
    flat = (FLAT_CURVE, FLAT_CURVE)
    stale = update_calibration(state, 0, evidence(curves=flat, selection_curves=flat), 1, parameters()).state
    assert stale.lagrange_multiplier == 0.0
    selection = (STEEP_CURVE, STEEP_CURVE)
    recovered = update_calibration(stale, 1, evidence(curves=flat, selection_curves=selection), 2, parameters()).state
    assert recovered.lagrange_multiplier == pytest.approx(solve_spend_matched_lagrange_multiplier(selection))


def test_the_fitted_corrector_reference_is_published_with_the_policy() -> None:
    state = initial_calibration_state(CONFIGURATION_SHA)
    corrector = CorrectorReference(path=Path('corrector-generation-00000000.jit.pt'), sha256='a' * 64)
    updated = update_calibration(state, 0, evidence(), 1, parameters(), corrector).state
    assert updated.published_policy.corrector_path == corrector.path
    assert updated.published_policy.corrector_sha256 == corrector.sha256
    assert working_policy(updated).corrector_path == corrector.path


def test_unconverged_spend_keeps_the_gate_closed_even_with_positive_gain() -> None:
    # A positive gain while mean spend still sits far above baseline only says that a larger budget
    # beats a smaller one, so it must not be allowed to open the gate.
    state = completed_state(40, warmup=5)
    overspending = update_calibration(
        state, 40, evidence(gain=0.5, realized_mean_multiple=3.0), 41, parameters(warmup=5)
    ).state
    assert BudgetEligibilityFailure.UNCONVERGED_SPEND in overspending.failed_eligibility_conditions
    assert not overspending.published_policy.apply_learned


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
    assert failed.lagrange_multiplier == state.lagrange_multiplier
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


def test_a_previous_schema_state_fails_closed_as_unreadable(tmp_path: Path) -> None:
    state = completed_state(2)
    path = tmp_path / 'calibration-state.json'
    payload = state.model_dump_json().replace('"schema_version":6', '"schema_version":5')
    path.write_text(payload, encoding='utf-8')
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


def test_evidence_requires_one_measured_curve_per_position() -> None:
    with pytest.raises(ValueError, match='every labelled position'):
        BudgetGenerationEvidence(
            position_count=2,
            mean_absolute_curve_error=(0.5,) * BUDGET_CURVE_POINTS,
            generation_gain=0.0,
            target_raw_kl_curves=(STEEP_CURVE,),
            selection_raw_kl_curves=(STEEP_CURVE,),
            realized_mean_multiple=1.0,
            realized_mean_assigned_visits=600.0,
            flat_mean_assigned_visits=600.0,
            selected_index_counts=(2, 0, 0, 0, 0, 0, 0, 0),
        )


def test_parameters_reject_invalid_configuration() -> None:
    with pytest.raises(ValueError, match='lambda trust ratio'):
        BudgetCalibrationParameters(lambda_trust_ratio=Decimal(1))
    with pytest.raises(ValueError, match='warm-up'):
        BudgetCalibrationParameters(warmup_completed_generations=0)


def test_seeded_lambda_is_finite_and_reasonable_for_production_scale_curves() -> None:
    curves = tuple(tuple(0.3 * (0.7**index) for index in range(BUDGET_CURVE_POINTS)) for _ in range(50))
    seeded = solve_spend_matched_lagrange_multiplier(curves)
    assert math.isfinite(seeded)
    assert mean_multiple_at(curves, seeded) <= 1.0


def test_a_dangling_corrector_reference_fails_closed_on_load(tmp_path: Path) -> None:
    state = completed_state(2)
    corrector = CorrectorReference(path=tmp_path / 'missing.jit.pt', sha256='a' * 64)
    updated = update_calibration(state, 2, evidence(), 3, parameters(), corrector).state
    path = tmp_path / 'calibration-state.json'
    save_calibration_state(path, updated)
    loaded = load_calibration_state_fail_closed(path, CONFIGURATION_SHA, 3)
    assert not loaded.published_policy.apply_learned
    assert loaded.decision_reason is BudgetDecisionReason.UNREADABLE_STATE


def test_a_matching_corrector_artifact_round_trips_through_disk(tmp_path: Path) -> None:
    import hashlib

    artifact = tmp_path / 'corrector.jit.pt'
    artifact.write_bytes(b'corrector-bytes')
    corrector = CorrectorReference(path=artifact, sha256=hashlib.sha256(b'corrector-bytes').hexdigest())
    state = update_calibration(completed_state(2), 2, evidence(), 3, parameters(), corrector).state
    path = tmp_path / 'calibration-state.json'
    save_calibration_state(path, state)
    assert load_calibration_state_fail_closed(path, CONFIGURATION_SHA, 3) == state


def test_an_altered_corrector_artifact_fails_closed_on_load(tmp_path: Path) -> None:
    import hashlib

    artifact = tmp_path / 'corrector.jit.pt'
    artifact.write_bytes(b'corrector-bytes')
    corrector = CorrectorReference(path=artifact, sha256=hashlib.sha256(b'corrector-bytes').hexdigest())
    state = update_calibration(completed_state(2), 2, evidence(), 3, parameters(), corrector).state
    path = tmp_path / 'calibration-state.json'
    save_calibration_state(path, state)
    artifact.write_bytes(b'altered-bytes')
    loaded = load_calibration_state_fail_closed(path, CONFIGURATION_SHA, 3)
    assert not loaded.published_policy.apply_learned
    assert loaded.decision_reason is BudgetDecisionReason.UNREADABLE_STATE


def test_oscillating_spend_keeps_the_gate_open_while_its_average_holds() -> None:
    state = completed_state(4, warmup=2)
    for generation, multiple in enumerate((1.25, 0.80, 1.20, 0.85), start=4):
        update = update_calibration(
            state, generation, evidence(realized_mean_multiple=multiple), generation + 1, parameters()
        )
        state = update.state
    assert state.ema_realized_mean_multiple == pytest.approx(1.0, abs=0.05)
    assert BudgetEligibilityFailure.UNCONVERGED_SPEND not in state.failed_eligibility_conditions
    assert state.published_policy.apply_learned


def test_sustained_spend_drift_still_closes_the_gate() -> None:
    state = completed_state(4, warmup=2)
    for generation in range(4, 20):
        state = update_calibration(
            state, generation, evidence(realized_mean_multiple=1.6), generation + 1, parameters()
        ).state
    assert state.ema_realized_mean_multiple > 1.05
    assert BudgetEligibilityFailure.UNCONVERGED_SPEND in state.failed_eligibility_conditions
    assert not state.published_policy.apply_learned
