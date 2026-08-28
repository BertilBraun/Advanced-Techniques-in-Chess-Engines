from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from src.search_budget.calibration import (
    BlendCalibrationState,
    BlendDecisionReason,
    BlendGenerationEvidence,
    initial_calibration_state,
    load_calibration_state_fail_closed,
    publish_fail_closed,
    published_blend_for_generation,
    save_calibration_state,
    update_calibration,
)

CONFIGURATION_SHA256 = 'a' * 64


def evidence(
    gains: tuple[float, ...], spend_errors: tuple[int, ...] | None = None
) -> tuple[BlendGenerationEvidence, ...]:
    errors = spend_errors if spend_errors is not None else (0,) * 11
    return tuple(
        BlendGenerationEvidence(
            blend=Decimal(index) / 10,
            generation_gain=gain,
            total_assigned_new_visits=6000 + errors[index],
            flat_total_new_visits=6000,
            position_count=10,
        )
        for index, gain in enumerate(gains)
    )


def warm_state(gains: tuple[float, ...]) -> BlendCalibrationState:
    state = initial_calibration_state(CONFIGURATION_SHA256)
    for generation in range(29):
        update = update_calibration(state, generation, evidence(gains), generation + 1)
        state = update.state
        assert state.selected_blend == 0
    return state


def test_ema_initializes_from_first_generation_and_updates_with_decay() -> None:
    state = initial_calibration_state(CONFIGURATION_SHA256)
    first = update_calibration(state, 0, evidence((0.0,) + (1.0,) * 10), 1).state
    assert first.candidate_states[1].ema_gain == 1.0
    second = update_calibration(first, 1, evidence((0.0,) + (0.0,) * 10), 2).state
    assert second.candidate_states[1].ema_gain == 0.8


def test_warmup_holds_zero_then_upward_step_limits_activation() -> None:
    increasing_gains = tuple(float(index) for index in range(11))
    state = warm_state(increasing_gains)
    activated = update_calibration(state, 29, evidence(increasing_gains), 30).state
    assert activated.selected_blend == Decimal('0.1')
    advanced = update_calibration(activated, 30, evidence(increasing_gains), 31).state
    assert advanced.selected_blend == Decimal('0.2')


def test_controller_retreats_immediately_and_breaks_ties_lower() -> None:
    increasing_gains = tuple(float(index) for index in range(11))
    state = warm_state(increasing_gains)
    state = update_calibration(state, 29, evidence(increasing_gains), 30).state
    state = update_calibration(state, 30, evidence(increasing_gains), 31).state
    retreat_gains = (0.0, 2.0, -1.0) + (-1.0,) * 8
    retreated = update_calibration(state, 31, evidence(retreat_gains), 32).state
    assert retreated.selected_blend == Decimal('0.1')

    equalized_candidates = tuple(
        candidate.model_copy(update={'ema_gain': 2.0})
        if candidate.blend in {Decimal('0.1'), Decimal('0.2')}
        else candidate
        for candidate in retreated.candidate_states
    )
    tie_start = retreated.model_copy(
        update={'selected_blend': Decimal('0.2'), 'candidate_states': equalized_candidates}
    )
    tie_gains = (0.0, 2.0, 2.0) + (-1.0,) * 8
    tied = update_calibration(tie_start, 32, evidence(tie_gains), 33).state
    assert tied.selected_blend == Decimal('0.1')


def test_nonpositive_or_spend_mismatched_candidates_fall_back_to_zero() -> None:
    state = warm_state((0.0,) + (1.0,) * 10)
    spend_errors = (0,) + (1,) * 10
    failed = update_calibration(state, 29, evidence((0.0,) + (1.0,) * 10, spend_errors), 30).state
    assert failed.selected_blend == 0
    assert failed.decision_reason == BlendDecisionReason.NO_ELIGIBLE_NONZERO

    nonpositive = update_calibration(state, 29, evidence((0.0,) * 11), 30).state
    assert nonpositive.selected_blend == 0
    assert nonpositive.decision_reason == BlendDecisionReason.NO_ELIGIBLE_NONZERO


def test_finalization_is_idempotent_and_publication_has_one_generation_lag() -> None:
    state = warm_state((0.0,) + (1.0,) * 10)
    update = update_calibration(state, 29, evidence((0.0,) + (1.0,) * 10), 35)
    assert update.applied
    assert published_blend_for_generation(update.state, 34) == 0
    assert published_blend_for_generation(update.state, 35) == Decimal('0.1')
    repeated = update_calibration(update.state, 29, evidence((0.0,) + (1.0,) * 10), 35)
    assert not repeated.applied
    assert repeated.state == update.state


def test_fail_closed_state_and_unreadable_persistence_publish_zero(tmp_path: Path) -> None:
    state = warm_state((0.0,) + (1.0,) * 10)
    state = update_calibration(state, 29, evidence((0.0,) + (1.0,) * 10), 30).state
    failed = publish_fail_closed(state, 31, BlendDecisionReason.TERMINAL_FAILURE)
    assert published_blend_for_generation(failed, 30) == Decimal('0.1')
    assert published_blend_for_generation(failed, 31) == 0

    path = tmp_path / 'state.json'
    path.write_text('{not json', encoding='utf-8')
    unreadable = load_calibration_state_fail_closed(path, CONFIGURATION_SHA256, 40)
    assert unreadable.selected_blend == 0
    assert unreadable.decision_reason == BlendDecisionReason.UNREADABLE_STATE

    save_calibration_state(path, state)
    restarted = load_calibration_state_fail_closed(path, CONFIGURATION_SHA256, 40)
    assert restarted == state
