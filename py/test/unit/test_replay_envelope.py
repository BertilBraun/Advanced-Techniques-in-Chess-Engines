from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from src.az.replay.envelope import (
    GameTermination,
    ReplayEnvelope,
    SearchBudgetClass,
    SearchStopReason,
    SearchStrategy,
    derive_self_play_seed_lineage,
)
from test.unit.go_stage5_helpers import envelope


def test_envelope_round_trip_is_stable() -> None:
    original = envelope()
    encoded = original.model_dump_json()

    decoded = ReplayEnvelope.model_validate_json(encoded)

    assert decoded == original
    assert decoded.model_dump_json() == encoded


def test_envelope_rejects_non_utc_timestamp() -> None:
    values = envelope().model_dump()
    values['created_at'] = datetime(2026, 7, 30)

    with pytest.raises(ValidationError, match='timezone-aware UTC'):
        ReplayEnvelope.model_validate(values)


@pytest.mark.parametrize(
    ('updates', 'message'),
    [
        ({'actual_simulations': 17}, 'cannot exceed'),
        ({'policy_target_weight': 0.0}, 'Policy eligibility'),
        ({'value_target_weight': 0.0}, 'Value eligibility'),
    ],
)
def test_envelope_rejects_inconsistent_accounting(updates: dict[str, int | float], message: str) -> None:
    values = envelope().model_dump()
    values.update(updates)

    with pytest.raises(ValidationError, match=message):
        ReplayEnvelope.model_validate(values)


def test_censored_envelope_requires_zero_value_eligibility() -> None:
    censored = envelope(termination=GameTermination.SAFETY_PLY_CAP)

    assert not censored.value_target_eligible
    assert censored.value_target_weight == 0


def test_envelope_rejects_tampered_seed_lineage() -> None:
    values = envelope().model_dump()
    values['seed_lineage']['action_sampling_seed'] += 1

    with pytest.raises(ValidationError, match='self-play seed lineage'):
        ReplayEnvelope.model_validate(values)


def test_self_play_lineage_coordinates_change_only_owned_derivation_levels() -> None:
    base = derive_self_play_seed_lineage(123, 1, 2, 3, 4)
    changed_ply = derive_self_play_seed_lineage(123, 1, 2, 3, 5)
    changed_worker = derive_self_play_seed_lineage(123, 1, 4, 3, 4)

    assert changed_ply.process_seed == base.process_seed
    assert changed_ply.worker_seed == base.worker_seed
    assert changed_ply.game_seed == base.game_seed
    assert changed_ply.search_seed != base.search_seed
    assert changed_ply.root_noise_seed != base.root_noise_seed
    assert changed_ply.action_sampling_seed != base.action_sampling_seed
    assert changed_worker.process_seed == base.process_seed
    assert changed_worker.worker_seed != base.worker_seed
    assert changed_worker.game_seed != base.game_seed
    assert (
        len(
            {
                base.search_seed,
                base.root_noise_seed,
                base.action_sampling_seed,
            }
        )
        == 3
    )


def test_self_play_lineage_rejects_generic_seed_coordinates() -> None:
    values = envelope().model_dump()
    values['seed_lineage']['coordinates'] = {
        'purpose': 'replay_sampling',
        'trainer_rank': 0,
        'optimizer_step': 0,
    }

    with pytest.raises(ValidationError, match='Extra inputs'):
        ReplayEnvelope.model_validate(values)


@pytest.mark.parametrize(
    ('mutate', 'message'),
    [
        ('root-visits', 'Root visit count'),
        ('full-budget-short', 'Full-budget'),
        ('terminal-policy', 'Terminal-root'),
        ('fixed-progressive-class', 'Fixed search'),
        ('adaptive-reason-on-fixed', 'Adaptive-confidence'),
        ('int32-cap', 'less than or equal'),
        ('lineage-ply', 'must match its self-play seed lineage'),
    ],
)
def test_envelope_rejects_tampered_search_provenance(mutate: str, message: str) -> None:
    values = envelope().model_dump()
    match mutate:
        case 'root-visits':
            values['root_diagnostics']['visit_count'] = 15
        case 'full-budget-short':
            values['actual_simulations'] = 15
            values['root_diagnostics']['visit_count'] = 15
        case 'terminal-policy':
            values['stop_reason'] = SearchStopReason.TERMINAL_ROOT
            values['actual_simulations'] = 0
            values['root_diagnostics']['visit_count'] = 0
        case 'fixed-progressive-class':
            values['budget_class'] = SearchBudgetClass.PROGRESSIVE_STAGE
        case 'adaptive-reason-on-fixed':
            values['stop_reason'] = SearchStopReason.ADAPTIVE_CONFIDENCE
        case 'int32-cap':
            values['configured_simulation_cap'] = 2**31
        case 'lineage-ply':
            values['ply'] = 2
        case _:
            raise AssertionError('Unhandled test mutation.')

    with pytest.raises(ValidationError, match=message):
        ReplayEnvelope.model_validate(values)


def test_valid_adaptive_early_stop_is_bounded() -> None:
    values = envelope().model_dump()
    values['search_strategy'] = SearchStrategy.ADAPTIVE
    values['budget_class'] = SearchBudgetClass.ADAPTIVE
    values['stop_reason'] = SearchStopReason.ADAPTIVE_CONFIDENCE
    values['actual_simulations'] = 8
    values['root_diagnostics']['visit_count'] = 8

    adaptive = ReplayEnvelope.model_validate(values)

    assert adaptive.actual_simulations == 8
