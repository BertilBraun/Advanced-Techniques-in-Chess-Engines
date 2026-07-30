from __future__ import annotations

import pytest

from src.az.config.search import NATIVE_SIMULATION_COUNT_MAXIMUM

try:
    import az_go_native as native
except ImportError:
    native = None

pytestmark = pytest.mark.skipif(
    native is None,
    reason='focused native Go extension has not been built',
)


class DeterministicEvaluator:
    def __init__(self) -> None:
        self.request_count = 0

    def __call__(self, request: native.GoInferenceRequest) -> native.InferenceResult:
        self.request_count += 1
        policy = [float(action + 1) for action in range(request.action_count)]
        value = 0.125 if self.request_count % 2 == 1 else -0.25
        return native.InferenceResult(request.request_id, policy, value)


def fixed_configuration(
    seed: int = 619,
    simulation_cap: int = 12,
) -> native.FixedPuctConfiguration:
    return native.FixedPuctConfiguration(
        simulation_cap=simulation_cap,
        exploration_constant=1.5,
        backup_discount=1.0,
        no_visited_child_value=0.0,
        action_temperature=0.0,
        root_noise_seed=seed,
        action_sampling_seed=seed + 1,
        root_noise=native.RootNoiseConfiguration(
            enabled=False,
            alpha=0.3,
            fraction=0.25,
        ),
        tree_reuse=False,
    )


def test_native_and_python_simulation_limits_match() -> None:
    assert native.MAXIMUM_SIMULATION_COUNT == NATIVE_SIMULATION_COUNT_MAXIMUM


def test_opt_in_prefix_trace_exposes_exact_snapshots_without_changing_result() -> None:
    rules = native.GoRules(5, 1, 80, 2)
    state = native.GoState(rules)
    baseline_evaluator = DeterministicEvaluator()
    traced_evaluator = DeterministicEvaluator()
    baseline = native.search_go_fixed(state, baseline_evaluator, fixed_configuration(simulation_cap=8))
    traced_configuration = native.FixedPuctConfiguration(
        8,
        1.5,
        1.0,
        0.0,
        0.0,
        619,
        620,
        native.RootNoiseConfiguration(False, 0.3, 0.25),
        False,
        native.FpuPolicy.VISITED_CHILD_MEAN,
        0.0,
        native.AdaptiveStoppingConfiguration(False, 1, 1, 1.0, 1.0),
        native.SearchBudgetClass.FIXED,
        1.0,
        native.PrefixTraceConfiguration(True, [2, 4, 6]),
    )
    traced = native.search_go_fixed(state, traced_evaluator, traced_configuration)

    assert traced.selected_action == baseline.selected_action
    assert traced.root_visits == baseline.root_visits
    assert traced.root_value == baseline.root_value
    assert [snapshot.simulations for snapshot in traced.prefix_trace] == [2, 4, 6]
    assert all(sum(snapshot.root_visits) == snapshot.simulations for snapshot in traced.prefix_trace)
    assert baseline.prefix_trace == []


def test_fixed_search_runs_through_typed_go_binding_repeatably() -> None:
    rules = native.GoRules(
        board_size=7,
        komi_half_points=15,
        safety_ply_cap=180,
        history_length=4,
    )
    first_evaluator = DeterministicEvaluator()
    second_evaluator = DeterministicEvaluator()

    first = native.search_go_fixed(
        native.GoState(rules),
        first_evaluator,
        fixed_configuration(),
    )
    second = native.search_go_fixed(
        native.GoState(rules),
        second_evaluator,
        fixed_configuration(),
    )

    assert first.selected_action == second.selected_action
    assert first.root_visits == second.root_visits
    assert first.root_policy == second.root_policy
    assert first.root_value == second.root_value
    assert first.telemetry.actual_simulations == 12
    assert first.telemetry.root_visit_count == 12
    assert first.telemetry.root_inference_requests == 1
    assert first.telemetry.total_inference_requests == first_evaluator.request_count
    assert first.telemetry.leaf_inference_requests == first.telemetry.total_inference_requests - 1
    assert sum(first.root_visits) == 12
    assert len(first.root_visits) == 50
    assert len(first.root_children) == 50
    assert first.telemetry.budget_class == native.SearchBudgetClass.FIXED
    assert first.telemetry.stop_reason == native.SearchStopReason.FULL_BUDGET
    assert first.telemetry.policy_target_eligible
    assert first.telemetry.policy_target_weight == 1.0
    assert first_evaluator.request_count == second_evaluator.request_count


def test_terminal_root_does_not_call_python_evaluator() -> None:
    rules = native.GoRules(7, 0, 180, 4)
    state = native.GoState(rules)
    state.apply(state.pass_action)
    state.apply(state.pass_action)
    evaluator = DeterministicEvaluator()

    result = native.search_go_fixed(state, evaluator, fixed_configuration())

    assert result.selected_action is None
    assert result.root_value == 0.0
    assert evaluator.request_count == 0
    assert result.telemetry.actual_simulations == 0
    assert result.telemetry.total_inference_requests == 0
    assert result.telemetry.stop_reason == native.SearchStopReason.TERMINAL_ROOT
    assert not result.telemetry.policy_target_eligible


def state_one_ply_before_cap() -> native.GoState:
    state = native.GoState(native.GoRules(7, 15, 49, 4))
    for _ in range(48):
        placement = next(action for action in state.legal_actions() if action != state.pass_action)
        state.apply(placement)
    return state


def test_censored_terminal_root_has_no_synthetic_value() -> None:
    state = state_one_ply_before_cap()
    state.apply(state.pass_action)
    evaluator = DeterministicEvaluator()

    result = native.search_go_fixed(state, evaluator, fixed_configuration())

    assert state.termination_reason == native.TerminationReason.SAFETY_PLY_CAP
    assert result.root_value is None
    assert evaluator.request_count == 0
    assert result.telemetry.total_inference_requests == 0


def test_censored_terminal_leaf_uses_network_value() -> None:
    state = state_one_ply_before_cap()
    evaluator = DeterministicEvaluator()

    result = native.search_go_fixed(
        state,
        evaluator,
        fixed_configuration(simulation_cap=1),
    )

    assert result.selected_action == state.pass_action
    assert result.root_value == pytest.approx(0.25)
    assert evaluator.request_count == 2
    assert result.telemetry.actual_simulations == 1
    assert result.telemetry.root_inference_requests == 1
    assert result.telemetry.leaf_inference_requests == 1
