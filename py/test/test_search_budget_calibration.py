from __future__ import annotations

from pathlib import Path

import pytest
from src.search_budget.calibration import (
    BucketCalibrationState,
    BucketGenerationEvidence,
    CurveCalibrationState,
    CurveDecisionReason,
    CurveGenerationEvidence,
    initial_calibration_state,
    load_calibration_state_fail_closed,
    publish_fail_closed,
    published_curve_for_generation,
    save_calibration_state,
    update_calibration,
)
from src.search_budget.configuration import SearchBudgetConfiguration
from src.search_budget.curve import CURVE_BUCKET_COUNT, flat_curve

CONFIGURATION_SHA256 = 'a' * 64


def generation_evidence(state: CurveCalibrationState, utility: float, gain: float = 1.0) -> CurveGenerationEvidence:
    pending = state.pending_curve
    return CurveGenerationEvidence(
        bucket_evidence=tuple(
            BucketGenerationEvidence(bucket_index=index, sample_count=10, generation_marginal_utility=utility + index)
            for index in range(CURVE_BUCKET_COUNT)
        ),
        validated_curve=pending,
        generation_gain=None if pending is None else gain,
        total_assigned_new_visits=None if pending is None else 6000,
        flat_total_new_visits=6000,
        position_count=10,
    )


def warm_state() -> CurveCalibrationState:
    state = initial_calibration_state(CONFIGURATION_SHA256)
    for generation in range(29):
        state = update_calibration(state, generation, generation_evidence(state, 1.0), generation + 1).state
        assert state.published_curve == flat_curve()
    return state


def test_bucket_ema_initializes_updates_and_retains_empty_bucket() -> None:
    state = initial_calibration_state(CONFIGURATION_SHA256)
    first = update_calibration(state, 0, generation_evidence(state, 1.0), 1).state
    empty_second = CurveGenerationEvidence(
        bucket_evidence=tuple(
            BucketGenerationEvidence(
                bucket_index=index,
                sample_count=0 if index == 0 else 10,
                generation_marginal_utility=None if index == 0 else 0.0,
            )
            for index in range(CURVE_BUCKET_COUNT)
        ),
        validated_curve=first.pending_curve,
        generation_gain=0.0,
        total_assigned_new_visits=6000,
        flat_total_new_visits=6000,
        position_count=10,
    )
    second = update_calibration(first, 1, empty_second, 2).state
    assert first.bucket_states[0].ema_utility == 1.0
    assert second.bucket_states[0].ema_utility == 1.0
    assert second.bucket_states[1].ema_utility == 1.6


def test_pending_curve_is_validated_one_generation_later_and_warmup_is_flat() -> None:
    state = initial_calibration_state(CONFIGURATION_SHA256)
    first = update_calibration(state, 0, generation_evidence(state, 1.0), 1).state
    assert first.pending_source_generation == 0
    assert first.current_validation_gain is None
    second = update_calibration(first, 1, generation_evidence(first, 1.0), 2).state
    assert second.current_validation_gain == 1.0
    assert second.published_curve == flat_curve()


def test_thirtieth_completed_generation_can_publish_positive_validated_curve() -> None:
    state = warm_state()
    updated = update_calibration(state, 29, generation_evidence(state, 1.0), 30).state
    assert updated.decision_reason == CurveDecisionReason.VALIDATED_PENDING
    assert updated.published_curve != flat_curve()


def test_nonpositive_current_or_ema_gain_retreats_immediately_to_flat() -> None:
    state = warm_state()
    activated = update_calibration(state, 29, generation_evidence(state, 1.0), 30).state
    retreated = update_calibration(activated, 30, generation_evidence(activated, 1.0, gain=-10.0), 31).state
    assert retreated.published_curve == flat_curve()
    assert retreated.decision_reason == CurveDecisionReason.NO_ELIGIBLE_PENDING


def test_finalization_is_idempotent_and_publication_applies_only_at_named_generation() -> None:
    state = warm_state()
    update = update_calibration(state, 29, generation_evidence(state, 1.0), 35)
    assert update.applied
    assert published_curve_for_generation(update.state, 34) == flat_curve()
    assert published_curve_for_generation(update.state, 35) == update.state.published_curve
    repeated = update_calibration(update.state, 29, generation_evidence(state, 1.0), 35)
    assert not repeated.applied
    assert repeated.state == update.state


def test_fail_closed_state_and_unreadable_persistence_publish_flat(tmp_path: Path) -> None:
    state = warm_state()
    state = update_calibration(state, 29, generation_evidence(state, 1.0), 30).state
    failed = publish_fail_closed(state, 31, CurveDecisionReason.TERMINAL_FAILURE)
    assert published_curve_for_generation(failed, 30) == state.published_curve
    assert published_curve_for_generation(failed, 31) == flat_curve()

    path = tmp_path / 'state.json'
    path.write_text('{not json', encoding='utf-8')
    unreadable = load_calibration_state_fail_closed(path, CONFIGURATION_SHA256, 40)
    assert unreadable.published_curve == flat_curve()
    assert unreadable.decision_reason == CurveDecisionReason.UNREADABLE_STATE

    save_calibration_state(path, state)
    restarted = load_calibration_state_fail_closed(path, CONFIGURATION_SHA256, 40)
    assert restarted == state


def test_legacy_blend_state_migrates_fail_closed_and_defaults_are_resolved(tmp_path: Path) -> None:
    configuration = SearchBudgetConfiguration()
    assert configuration.curve_version == 'live_ema_ten_bucket_v1'
    assert configuration.calibration.bucket_count == CURVE_BUCKET_COUNT
    assert configuration.calibration.initializer_version == 'analytic_q5_v1'
    assert configuration.calibration.warmup_completed_source_generations == 30
    assert configuration.labeling.parallel_searches == 2

    path = tmp_path / 'legacy-state.json'
    path.write_text(
        '{"schema_version":1,"configuration_sha256":"'
        + CONFIGURATION_SHA256
        + '","finalized_source_generations":[],"previous_blend":"0",'
        '"selected_blend":"0","application_generation":0,"candidate_states":[],"decision_reason":"initial"}',
        encoding='utf-8',
    )
    migrated = load_calibration_state_fail_closed(path, CONFIGURATION_SHA256, 12)
    assert migrated.published_curve == flat_curve()
    assert migrated.application_generation == 12
    assert migrated.decision_reason == CurveDecisionReason.UNREADABLE_STATE


def test_persisted_calibration_rejects_nonfinite_aggregate_state() -> None:
    with pytest.raises(ValueError, match='finite'):
        BucketCalibrationState(
            bucket_index=0,
            sample_count=1,
            current_generation_utility=1.0,
            ema_utility=1.0,
            raw_log_update=float('nan'),
            projection_adjustment=0.0,
        )
