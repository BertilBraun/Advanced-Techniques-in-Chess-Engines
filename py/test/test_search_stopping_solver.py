from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest
from src.search_stopping.configuration import SearchStoppingConfiguration
from src.search_stopping.solver import (
    AuditWindowArrays,
    solve_noise_floor_anchored_eps,
    solve_thresholds,
    uncertain_labels,
)


def _configuration(**overrides: object) -> SearchStoppingConfiguration:
    values: dict[str, object] = {
        'audit_sample_fraction': Decimal('0.01'),
        'paired_audit_fraction': Decimal('0.1'),
        'noise_floor_multiple': 1.0,
        'anchor_fraction': Decimal('0.05'),
        'anchor_visit_multiple': 4.0,
        'checkpoint_multiples': (0.5, 1.0),
        'cap_multiple': 2.0,
        'eps_pi_minimum': 0.01,
        'eps_pi_maximum': 0.5,
        'eps_v': 0.3,
        'movement_guard_epsilon': 0.05,
        'false_stop_rate_ceiling': 0.05,
        'minimum_evidence_trigger_count': 5,
        'confidence_level': 0.5,
        'first_production_generation': 10,
        'maximum_realized_mean_spend': 1.3,
        'window_generations': 10,
        'maximum_unstarted_generation_lag': 2,
    }
    values.update(overrides)
    return SearchStoppingConfiguration(**values)


def _arrays(
    kl: np.ndarray,
    guard: np.ndarray | None = None,
    probability: np.ndarray | None = None,
) -> AuditWindowArrays:
    return AuditWindowArrays(
        kl_to_final=kl.astype(np.float64),
        value_gap=np.zeros_like(kl, dtype=np.float64),
        guard_movement=np.zeros_like(kl, dtype=np.float64) if guard is None else guard.astype(np.float64),
        stop_probability=np.zeros_like(kl, dtype=np.float64) if probability is None else probability.astype(np.float64),
    )


def test_uncertain_labels_are_instantaneous_exceedance() -> None:
    kl = np.array([[0.005, 0.2], [0.2, 0.005], [0.005, 0.005]])
    labels = uncertain_labels(_arrays(kl), eps_pi=0.1, eps_v=0.3)
    assert labels.tolist() == [[False, True], [True, False], [False, False]]


def test_threshold_solve_separates_certain_from_uncertain_positions() -> None:
    count = 40
    kl = np.zeros((count, 2))
    kl[: count // 2, :] = 0.5  # first half truly uncertain at every checkpoint
    probability = np.full((count, 2), 0.9)
    probability[count // 2 :, :] = 0.1  # predictor is confident exactly on the certain half
    solution = solve_thresholds(
        _arrays(kl, probability=probability),
        uncertain_labels(_arrays(kl), 0.1, 0.3),
        _configuration(),
    )
    assert not solution.checkpoints[0].attenuated
    assert solution.checkpoints[0].false_stop_count == 0
    assert solution.simulated_mean_spend == pytest.approx(0.5 * 0.5 + 0.5 * 2.0)


def test_threshold_solve_attenuates_without_minimum_evidence() -> None:
    kl = np.zeros((3, 2))
    solution = solve_thresholds(_arrays(kl), uncertain_labels(_arrays(kl), 0.1, 0.3), _configuration())
    assert all(checkpoint.attenuated for checkpoint in solution.checkpoints)
    assert not solution.any_checkpoint_open
    assert solution.simulated_mean_spend == pytest.approx(2.0)


def test_threshold_solve_respects_the_false_stop_ceiling() -> None:
    count = 100
    kl = np.zeros((count, 2))
    kl[::2, :] = 0.5  # half uncertain, interleaved
    probability = np.full((count, 2), 0.5)  # predictor cannot separate them
    solution = solve_thresholds(
        _arrays(kl, probability=probability),
        uncertain_labels(_arrays(kl), 0.1, 0.3),
        _configuration(false_stop_rate_ceiling=0.01, confidence_level=0.95),
    )
    assert all(checkpoint.attenuated for checkpoint in solution.checkpoints)


def test_guard_failure_excludes_positions_from_stopping() -> None:
    count = 20
    kl = np.zeros((count, 2))
    guard = np.full((count, 2), 1.0)  # movement above the guard epsilon everywhere
    probability = np.zeros((count, 2))
    solution = solve_thresholds(
        _arrays(kl, guard=guard, probability=probability),
        uncertain_labels(_arrays(kl), 0.1, 0.3),
        _configuration(),
    )
    assert solution.simulated_mean_spend == pytest.approx(2.0)
    assert all(fraction == 0.0 for fraction in solution.simulated_stop_fraction)


def test_stopped_positions_leave_the_survivor_population() -> None:
    count = 20
    kl = np.zeros((count, 2))
    probability = np.zeros((count, 2))
    solution = solve_thresholds(
        _arrays(kl, probability=probability),
        uncertain_labels(_arrays(kl), 0.1, 0.3),
        _configuration(),
    )
    assert solution.simulated_stop_fraction[0] == pytest.approx(1.0)
    assert solution.simulated_stop_fraction[1] == pytest.approx(0.0)
    assert solution.simulated_mean_spend == pytest.approx(0.5)


def test_eps_anchors_to_the_measured_noise_floor() -> None:
    kl = np.zeros((100, 2))
    floors = np.full(500, 0.12)
    solution = solve_noise_floor_anchored_eps(_arrays(kl), floors, _configuration())
    assert solution.eps_pi == pytest.approx(0.12)
    assert solution.measured_noise_floor == pytest.approx(0.12)
    assert not solution.clamped


def test_eps_anchor_clamps_at_both_ends() -> None:
    kl = np.zeros((100, 2))
    low = solve_noise_floor_anchored_eps(_arrays(kl), np.full(9, 0.001), _configuration())
    high = solve_noise_floor_anchored_eps(_arrays(kl), np.full(9, 5.0), _configuration())
    assert low.eps_pi == pytest.approx(0.01) and low.clamped
    assert high.eps_pi == pytest.approx(0.5) and high.clamped


def test_eps_anchor_scales_with_the_configured_multiple() -> None:
    kl = np.zeros((100, 2))
    solution = solve_noise_floor_anchored_eps(_arrays(kl), np.full(9, 0.2), _configuration(noise_floor_multiple=0.75))
    assert solution.eps_pi == pytest.approx(0.15)


def test_eps_anchor_requires_paired_measurements() -> None:
    kl = np.zeros((100, 2))
    with pytest.raises(ValueError, match='paired-audit'):
        solve_noise_floor_anchored_eps(_arrays(kl), np.array([]), _configuration())


def test_simulated_spend_is_monotone_nonincreasing_in_eps() -> None:
    generator = np.random.default_rng(3)
    kl = generator.uniform(0.0, 0.4, size=(500, 2))
    probability = np.clip(kl * 2.0, 0.0, 1.0)
    arrays = _arrays(kl, probability=probability)
    configuration = _configuration(minimum_evidence_trigger_count=10)
    spends = [
        solve_thresholds(arrays, uncertain_labels(arrays, eps, 0.3), configuration).simulated_mean_spend
        for eps in (0.02, 0.05, 0.1, 0.2, 0.4)
    ]
    assert spends == sorted(spends, reverse=True)
