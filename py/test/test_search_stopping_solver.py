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
        'excess_cost_ceiling': 0.25,
        'catastrophic_excess_multiple': 5.0,
        'catastrophic_stop_ceiling': 0.05,
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
    solution = solve_thresholds(_arrays(kl, probability=probability), _configuration(), eps_pi=0.1)
    assert not solution.checkpoints[0].attenuated
    assert solution.checkpoints[0].false_stop_count == 0
    assert solution.checkpoints[0].mean_excess_cost == pytest.approx(0.0)
    assert solution.simulated_mean_spend == pytest.approx(0.5 * 0.5 + 0.5 * 2.0)


def test_threshold_solve_attenuates_without_minimum_evidence() -> None:
    kl = np.zeros((3, 2))
    solution = solve_thresholds(_arrays(kl), _configuration(), eps_pi=0.1)
    assert all(checkpoint.attenuated for checkpoint in solution.checkpoints)
    assert not solution.any_checkpoint_open
    assert solution.simulated_mean_spend == pytest.approx(2.0)


def test_threshold_solve_attenuates_when_the_cost_bound_exceeds_the_ceiling() -> None:
    count = 100
    kl = np.zeros((count, 2))
    kl[::2, :] = 0.5  # half badly wrong, interleaved
    probability = np.full((count, 2), 0.5)  # predictor cannot separate them
    solution = solve_thresholds(
        _arrays(kl, probability=probability),
        _configuration(confidence_level=0.95, catastrophic_stop_ceiling=0.9),
        eps_pi=0.1,
    )
    assert all(checkpoint.attenuated for checkpoint in solution.checkpoints)


def test_cost_criterion_admits_cheap_false_stops_the_rate_criterion_would_reject() -> None:
    count = 200
    kl = np.full((count, 2), 0.11)  # every stop is a false stop, but barely: excess 0.01 = 0.1 eps
    probability = np.zeros((count, 2))
    solution = solve_thresholds(
        _arrays(kl, probability=probability),
        _configuration(confidence_level=0.95),
        eps_pi=0.1,
    )
    assert not solution.checkpoints[0].attenuated
    assert solution.checkpoints[0].false_stop_count == solution.checkpoints[0].trigger_count
    assert solution.simulated_mean_spend == pytest.approx(0.5)


def test_admission_uses_the_upper_bound_not_the_sample_mean() -> None:
    generator = np.random.default_rng(11)
    count = 400
    kl = np.zeros((count, 2))
    # Mean excess sits exactly at the budget, so any positive z-score must reject.
    kl[:, :] = 0.1 + generator.uniform(0.0, 0.05, size=(count, 2))
    probability = np.zeros((count, 2))
    arrays = _arrays(kl, probability=probability)
    budget = float(np.maximum(0.0, kl[:, 0] - 0.1).mean()) * 1.000001 / 0.1
    tolerant = solve_thresholds(arrays, _configuration(excess_cost_ceiling=budget, confidence_level=0.5), eps_pi=0.1)
    strict = solve_thresholds(arrays, _configuration(excess_cost_ceiling=budget, confidence_level=0.99), eps_pi=0.1)
    assert not tolerant.checkpoints[0].attenuated
    assert strict.checkpoints[0].attenuated
    assert tolerant.checkpoints[0].excess_cost_upper_bound >= tolerant.checkpoints[0].mean_excess_cost


def test_tail_guard_rejects_rare_catastrophic_stops_the_mean_bound_admits() -> None:
    count = 1000
    kl = np.zeros((count, 2))
    kl[::50, :] = 4.0  # 2% of stops are catastrophic (40x eps) yet the mean excess is only 0.8 eps
    probability = np.zeros((count, 2))
    arrays = _arrays(kl, probability=probability)
    guarded = solve_thresholds(
        arrays,
        _configuration(excess_cost_ceiling=1.5, catastrophic_stop_ceiling=0.01, confidence_level=0.95),
        eps_pi=0.1,
    )
    unguarded = solve_thresholds(
        arrays,
        _configuration(excess_cost_ceiling=1.5, catastrophic_stop_ceiling=0.5, confidence_level=0.95),
        eps_pi=0.1,
    )
    assert guarded.checkpoints[0].attenuated
    assert not unguarded.checkpoints[0].attenuated


def test_guard_failure_excludes_positions_from_stopping() -> None:
    count = 20
    kl = np.zeros((count, 2))
    guard = np.full((count, 2), 1.0)  # movement above the guard epsilon everywhere
    probability = np.zeros((count, 2))
    solution = solve_thresholds(_arrays(kl, guard=guard, probability=probability), _configuration(), eps_pi=0.1)
    assert solution.simulated_mean_spend == pytest.approx(2.0)
    assert all(fraction == 0.0 for fraction in solution.simulated_stop_fraction)


def test_stopped_positions_leave_the_survivor_population() -> None:
    count = 20
    kl = np.zeros((count, 2))
    probability = np.zeros((count, 2))
    solution = solve_thresholds(_arrays(kl, probability=probability), _configuration(), eps_pi=0.1)
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
        solve_thresholds(arrays, configuration, eps_pi=eps).simulated_mean_spend for eps in (0.02, 0.05, 0.1, 0.2, 0.4)
    ]
    assert spends == sorted(spends, reverse=True)
