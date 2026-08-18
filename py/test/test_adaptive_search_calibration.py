import pytest
from src.self_play.parameters import AdaptiveFullSearchBudget
from tools.calibrate_adaptive_search import PolicyEntry, PositionAudit, SearchSnapshot, candidate_metrics


def _snapshot(
    visits: int,
    correction: float,
    first_probability: float,
    selected_action_id: int = 0,
) -> SearchSnapshot:
    return SearchSnapshot(
        visits=visits,
        selected_action_id=selected_action_id,
        policy_leader_action_id=0,
        root_value=0.25,
        predicted_search_correction=correction,
        policy=(
            PolicyEntry(action_id=0, probability=first_probability),
            PolicyEntry(action_id=1, probability=1.0 - first_probability),
        ),
    )


def test_candidate_metrics_applies_midpoint_correction_gate() -> None:
    audit = PositionAudit(
        fen='8/8/8/8/8/8/4K3/6k1 w - - 0 1',
        final_policy_correction=0.3,
        final_value_correction=0.1,
        final_search_correction_target=0.3,
        snapshots=tuple(
            _snapshot(visits, correction=0.4, first_probability=0.55 + visits / 100_000)
            for visits in (*range(400, 900, 100), 3200)
        ),
    )
    adaptive = AdaptiveFullSearchBudget(
        kind='adaptive',
        minimum_visits=400,
        maximum_visits=800,
        observation_interval=100,
        leader_stability_window=200,
        root_value_tolerance=0.04,
        initial_top_visit_share=1.0,
        final_top_visit_share=1.0,
        initial_top_two_margin=1.0,
        final_top_two_margin=1.0,
        threshold_relaxation_visits=1200,
        minimum_search_correction_to_unlock_tail=0.5,
    )

    denied = candidate_metrics((audit,), adaptive, maximum_visits=800, threshold=0.5)
    unlocked = candidate_metrics((audit,), adaptive, maximum_visits=800, threshold=0.3)

    assert denied.mean_visits == 600
    assert denied.maximum_visits_reached_fraction == 0.0
    assert denied.selected_move_agreement == 1.0
    assert denied.mean_policy_total_variation == pytest.approx(0.026)
    assert denied.mean_final_search_correction_target == pytest.approx(0.3)
    assert denied.correction_calibration_by_stop_visits[0].mean_absolute_error == pytest.approx(0.1)
    assert denied.unstable_at_gate_positions == 0
    assert denied.unstable_at_gate_recall is None
    assert unlocked.mean_visits == 800
    assert unlocked.maximum_visits_reached_fraction == 1.0


def test_candidate_metrics_reports_unstable_gate_recall() -> None:
    snapshots = tuple(
        _snapshot(
            visits,
            correction=0.7,
            first_probability=0.55 if visits < 3200 else 0.9,
            selected_action_id=1 if visits == 600 else 0,
        )
        for visits in (*range(400, 900, 100), 3200)
    )
    audit = PositionAudit(
        fen='8/8/8/8/8/8/4K3/6k1 w - - 0 1',
        final_policy_correction=0.4,
        final_value_correction=0.1,
        final_search_correction_target=0.4,
        snapshots=snapshots,
    )
    adaptive = AdaptiveFullSearchBudget(
        kind='adaptive',
        minimum_visits=400,
        maximum_visits=800,
        observation_interval=100,
        leader_stability_window=200,
        root_value_tolerance=0.04,
        initial_top_visit_share=1.0,
        final_top_visit_share=1.0,
        initial_top_two_margin=1.0,
        final_top_two_margin=1.0,
        threshold_relaxation_visits=1200,
        minimum_search_correction_to_unlock_tail=0.5,
    )

    metrics = candidate_metrics((audit,), adaptive, maximum_visits=800, threshold=0.5)

    assert metrics.unstable_at_gate_positions == 1
    assert metrics.unstable_at_gate_recall == 1.0


def test_adaptive_budget_rejects_nonpositive_relaxation_window() -> None:
    with pytest.raises(ValueError, match='observation cadence'):
        AdaptiveFullSearchBudget(
            kind='adaptive',
            minimum_visits=400,
            maximum_visits=800,
            observation_interval=100,
            leader_stability_window=200,
            root_value_tolerance=0.04,
            initial_top_visit_share=0.78,
            final_top_visit_share=0.68,
            initial_top_two_margin=0.32,
            final_top_two_margin=0.18,
            threshold_relaxation_visits=0,
            minimum_search_correction_to_unlock_tail=0.5,
        )
