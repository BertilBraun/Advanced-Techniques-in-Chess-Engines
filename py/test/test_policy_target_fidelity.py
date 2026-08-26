from __future__ import annotations

import pytest
from tools.measure_policy_target_fidelity import (
    AdaptiveStoppingRule,
    EqualComputeComparison,
    RuleFidelity,
    _adaptive_stop,
    _Checkpoint,
    _equal_compute_comparisons,
    _fixed_stop,
    _kullback_leibler,
    _probabilities,
    _total_variation,
)

_REFERENCE_INTERVAL = 50


def _checkpoint(
    visits: int,
    leader_action_id: int,
    top_visit_share: float = 0.9,
    top_two_margin: float = 0.8,
    root_value: float = 0.5,
) -> _Checkpoint:
    return _Checkpoint(
        visits=visits,
        leader_action_id=leader_action_id,
        most_visited_action_id=leader_action_id,
        top_visit_share=top_visit_share,
        top_two_margin=top_two_margin,
        root_value=root_value,
        policy=((leader_action_id, visits),),
    )


def _trace(leaders: tuple[int, ...], **overrides: float) -> tuple[_Checkpoint, ...]:
    return tuple(
        _checkpoint((index + 1) * _REFERENCE_INTERVAL, leader, **overrides) for index, leader in enumerate(leaders)
    )


def _rule(**overrides: object) -> AdaptiveStoppingRule:
    fields: dict[str, object] = {
        'label': 'rule',
        'minimum_visits': 200,
        'maximum_visits': 600,
        'observation_interval': 100,
        'leader_stability_window': 200,
        'root_value_tolerance': 0.04,
        'initial_top_visit_share': 0.7,
        'final_top_visit_share': 0.5,
        'initial_top_two_margin': 0.45,
        'final_top_two_margin': 0.15,
        'threshold_relaxation_visits': 1200,
    }
    return AdaptiveStoppingRule(**{**fields, **overrides})


def test_fixed_stop_returns_first_checkpoint_reaching_the_budget() -> None:
    trace = _trace((7,) * 12)
    assert _fixed_stop(trace, 300).visits == 300


def test_fixed_stop_clamps_to_the_final_checkpoint_when_the_budget_exceeds_the_trace() -> None:
    trace = _trace((7,) * 4)
    assert _fixed_stop(trace, 10_000).visits == 200


def test_adaptive_stop_waits_for_the_minimum_visit_count() -> None:
    trace = _trace((7,) * 12)
    assert _adaptive_stop(trace, _rule(minimum_visits=400), _REFERENCE_INTERVAL).visits == 400


def test_adaptive_stop_requires_a_stable_leader_across_the_window() -> None:
    # The leader alternates at every checkpoint the rule observes, so no window is ever stable.
    trace = _trace(tuple((index // 2) % 2 for index in range(12)))
    assert _adaptive_stop(trace, _rule(), _REFERENCE_INTERVAL).visits == 600


def test_adaptive_stop_holds_on_while_the_root_value_still_moves() -> None:
    trace = tuple(_checkpoint((index + 1) * _REFERENCE_INTERVAL, 7, root_value=0.1 * index) for index in range(12))
    assert _adaptive_stop(trace, _rule(root_value_tolerance=0.0), _REFERENCE_INTERVAL).visits == 600


def test_adaptive_stop_returns_the_maximum_when_confidence_never_arrives() -> None:
    trace = _trace((7,) * 12, top_visit_share=0.1, top_two_margin=0.05)
    assert _adaptive_stop(trace, _rule(), _REFERENCE_INTERVAL).visits == 600


def test_adaptive_stop_accepts_a_top_two_margin_without_a_top_share() -> None:
    trace = _trace((7,) * 12, top_visit_share=0.1, top_two_margin=0.5)
    assert _adaptive_stop(trace, _rule(), _REFERENCE_INTERVAL).visits == 300


def test_adaptive_stop_observes_only_its_own_interval() -> None:
    trace = _trace((7,) * 12)
    assert _adaptive_stop(trace, _rule(observation_interval=200, leader_stability_window=400), 50).visits == 600


def test_adaptive_stop_rejects_an_interval_finer_than_the_reference_trace() -> None:
    with pytest.raises(ValueError, match='cannot reproduce'):
        _adaptive_stop(_trace((7,) * 12), _rule(observation_interval=25), _REFERENCE_INTERVAL)


def test_adaptive_stop_never_exceeds_its_maximum_visits() -> None:
    trace = _trace((7,) * 40, top_visit_share=0.1, top_two_margin=0.05)
    assert _adaptive_stop(trace, _rule(maximum_visits=400), _REFERENCE_INTERVAL).visits == 400


def test_identical_policies_have_no_divergence() -> None:
    policy = _probabilities(((1, 30), (2, 10)))
    assert _kullback_leibler(policy, policy) == pytest.approx(0.0)
    assert _total_variation(policy, policy) == pytest.approx(0.0)


def test_total_variation_counts_mass_the_candidate_never_visited() -> None:
    reference = _probabilities(((1, 50), (2, 50)))
    candidate = _probabilities(((1, 100),))
    assert _total_variation(reference, candidate) == pytest.approx(0.5)


def test_divergence_stays_finite_when_the_candidate_missed_a_reference_action() -> None:
    reference = _probabilities(((1, 50), (2, 50)))
    candidate = _probabilities(((1, 100),))
    assert 0.0 < _kullback_leibler(reference, candidate) < 10.0


def _fidelity(label: str, kind: str, visits: float, divergence: float) -> RuleFidelity:
    return RuleFidelity(
        label=label,
        kind=kind,
        positions=10,
        mean_stop_visits=visits,
        median_stop_visits=visits,
        maximum_reached_fraction=0.0,
        policy_leader_agreement=1.0,
        most_visited_agreement=1.0,
        mean_policy_kullback_leibler=divergence,
        mean_policy_total_variation=0.0,
        mean_root_value_absolute_error=0.0,
    )


def test_equal_compute_comparison_credits_an_adaptive_rule_that_beats_the_fixed_curve() -> None:
    rules = (
        _fidelity('fixed-200', 'fixed', 200.0, 0.20),
        _fidelity('fixed-400', 'fixed', 400.0, 0.10),
        _fidelity('fixed-600', 'fixed', 600.0, 0.06),
        _fidelity('adaptive', 'adaptive', 300.0, 0.10),
    )
    comparison = _equal_compute_comparisons(rules)[0]
    assert comparison.fixed_kullback_leibler_at_equal_compute == pytest.approx(0.15)
    assert comparison.kullback_leibler_advantage == pytest.approx(0.05)
    assert comparison.equivalent_fixed_visits == pytest.approx(400.0)
    assert comparison.visit_saving == pytest.approx(100.0)


def test_equal_compute_comparison_declines_to_extrapolate_beyond_the_fixed_curve() -> None:
    rules = (
        _fidelity('fixed-200', 'fixed', 200.0, 0.20),
        _fidelity('fixed-400', 'fixed', 400.0, 0.10),
        _fidelity('adaptive', 'adaptive', 900.0, 0.01),
    )
    comparison = _equal_compute_comparisons(rules)[0]
    assert isinstance(comparison, EqualComputeComparison)
    assert comparison.fixed_kullback_leibler_at_equal_compute is None
    assert comparison.equivalent_fixed_visits is None
