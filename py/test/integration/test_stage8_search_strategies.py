from __future__ import annotations

from collections.abc import Callable
from pathlib import PurePosixPath
from uuid import UUID

import pytest

native = pytest.importorskip('az_go_native', reason='focused native Go extension has not been built')

from src.az.config.artifacts import CalibrationArtifactReference
from src.az.config.search import (
    ConstantTemperature,
    DisabledRootExploration,
    FixedSearchBudget,
    FullBudgetStopping,
    MixedSearchBudget,
    ProgressiveSearchBudget,
    SearchBudgetStage,
    VisitMarginAdaptiveRule,
    VisitedChildMeanFpu,
)
from src.az.replay.envelope import SearchBudgetClass, SearchStrategy
from src.az.self_play.configuration import NativeSearchSpecification
from src.az.self_play.worker import _select_search


def _search_specification(
    budget: FixedSearchBudget | ProgressiveSearchBudget | MixedSearchBudget,
) -> NativeSearchSpecification:
    return NativeSearchSpecification(
        budget=budget,
        stopping=FullBudgetStopping(kind='full_budget'),
        fpu=VisitedChildMeanFpu(kind='visited_child_mean', no_visited_child_value=0),
        exploration_constant=1.5,
        backup_discount=1,
        temperature=ConstantTemperature(kind='constant', temperature=0),
        root_exploration=DisabledRootExploration(kind='disabled'),
    )


def _calibration(identity: int) -> CalibrationArtifactReference:
    return CalibrationArtifactReference(
        artifact_root='reference_artifacts',
        artifact_id=UUID(int=identity),
        path=PurePosixPath(f'calibration/{identity}.json'),
        sha256=f'{identity:064x}',
    )


@pytest.mark.parametrize(
    ('elapsed_seconds', 'expected_cap'),
    ((0, 4), (9, 4), (10, 8), (19, 8), (20, 16)),
)
def test_progressive_budget_uses_exact_shared_epoch_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    elapsed_seconds: int,
    expected_cap: int,
) -> None:
    epoch = 7_000_000_000
    monkeypatch.setattr(
        'src.az.self_play.worker.time.monotonic_ns',
        lambda: epoch + elapsed_seconds * 1_000_000_000,
    )
    specification = _search_specification(
        ProgressiveSearchBudget(
            kind='progressive',
            stages=(
                SearchBudgetStage(start_elapsed_seconds=0, simulations=4),
                SearchBudgetStage(start_elapsed_seconds=10, simulations=8),
                SearchBudgetStage(start_elapsed_seconds=20, simulations=16),
            ),
        )
    )

    selected = _select_search(specification, 41, epoch)

    assert selected.strategy is SearchStrategy.PROGRESSIVE
    assert selected.budget_class is SearchBudgetClass.PROGRESSIVE_STAGE
    assert selected.simulation_cap == expected_cap


def test_mixed_budget_decision_is_position_deterministic_and_weighted() -> None:
    specification = _search_specification(
        MixedSearchBudget(
            kind='mixed',
            cheap_simulations=2,
            full_simulations=12,
            full_search_probability=0.25,
            cheap_policy_target_weight=0,
            full_policy_target_weight=1,
        )
    )

    first = _select_search(specification, 91, 0)
    repeated = _select_search(specification, 91, 0)
    population = tuple(_select_search(specification, seed, 0) for seed in range(0, 2**63, 2**57))

    assert first == repeated
    assert {selection.budget_class for selection in population} == {
        SearchBudgetClass.MIXED_FAST,
        SearchBudgetClass.MIXED_FULL,
    }
    for selection in population:
        if selection.budget_class is SearchBudgetClass.MIXED_FAST:
            assert (selection.simulation_cap, selection.policy_target_weight) == (2, 0)
        else:
            assert (selection.simulation_cap, selection.policy_target_weight) == (12, 1)


def test_adaptive_selection_carries_calibration_identity() -> None:
    base = _search_specification(FixedSearchBudget(kind='fixed', simulations=32))
    specification = base.model_copy(
        update={
            'stopping': VisitMarginAdaptiveRule(
                kind='visit_margin',
                minimum_simulations=8,
                check_interval_simulations=4,
                required_top_visit_fraction=0.75,
                required_top_two_margin=0.5,
                calibration=_calibration(1),
            )
        }
    )

    selected = _select_search(specification, 1, 0)

    assert selected.strategy is SearchStrategy.ADAPTIVE
    assert selected.budget_class is SearchBudgetClass.FIXED
    assert selected.search_calibration.artifact_id == UUID(int=1)


def test_progressive_adaptive_preserves_progressive_budget_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    epoch = 1_000_000_000
    monkeypatch.setattr('src.az.self_play.worker.time.monotonic_ns', lambda: epoch + 10_000_000_000)
    base = _search_specification(
        ProgressiveSearchBudget(
            kind='progressive',
            stages=(
                SearchBudgetStage(start_elapsed_seconds=0, simulations=8),
                SearchBudgetStage(start_elapsed_seconds=10, simulations=16),
            ),
        )
    )
    specification = base.model_copy(
        update={
            'stopping': VisitMarginAdaptiveRule(
                kind='visit_margin',
                minimum_simulations=4,
                check_interval_simulations=2,
                required_top_visit_fraction=0.75,
                required_top_two_margin=0.5,
                calibration=_calibration(2),
            )
        }
    )

    selected = _select_search(specification, 1, epoch)

    assert selected.strategy is SearchStrategy.ADAPTIVE
    assert selected.budget_class is SearchBudgetClass.PROGRESSIVE_STAGE
    assert selected.simulation_cap == 16


def _native_configuration(
    *,
    fpu_policy: native.FpuPolicy,
    adaptive: native.AdaptiveStoppingConfiguration,
    budget_class: native.SearchBudgetClass,
    policy_weight: float,
) -> native.FixedPuctConfiguration:
    return native.FixedPuctConfiguration(
        simulation_cap=16,
        exploration_constant=0.01,
        backup_discount=1,
        no_visited_child_value=0,
        action_temperature=0,
        root_noise_seed=1,
        action_sampling_seed=2,
        root_noise=native.RootNoiseConfiguration(False, 0.3, 0.25),
        tree_reuse=False,
        fpu_policy=fpu_policy,
        fpu_reduction=0.5,
        adaptive_stopping=adaptive,
        budget_class=budget_class,
        policy_target_weight=policy_weight,
    )


def _evaluator(value: float) -> Callable[[native.GoInferenceRequest], native.InferenceResult]:
    def evaluate(request: native.GoInferenceRequest) -> native.InferenceResult:
        policy = [0.0] * request.action_count
        policy[0] = 1.0
        return native.InferenceResult(request.request_id, policy, value)

    return evaluate


def test_binding_executes_fpu_and_adaptive_stop_with_typed_metadata() -> None:
    state = native.GoState(native.GoRules(3, 1, 40, 2))
    configuration = _native_configuration(
        fpu_policy=native.FpuPolicy.REDUCED_PARENT_VALUE,
        adaptive=native.AdaptiveStoppingConfiguration(True, 4, 2, 0.75, 0.5),
        budget_class=native.SearchBudgetClass.FIXED,
        policy_weight=1,
    )

    result = native.search_go_fixed(state, _evaluator(0.6), configuration)

    assert result.telemetry.actual_simulations == 4
    assert result.telemetry.stop_reason == native.SearchStopReason.ADAPTIVE_CONFIDENCE
    assert result.telemetry.budget_class == native.SearchBudgetClass.FIXED
    assert result.telemetry.initial_root_fpu == pytest.approx(0.6)


def test_binding_preserves_zero_policy_weight_for_fast_search() -> None:
    state = native.GoState(native.GoRules(3, 1, 40, 2))
    configuration = _native_configuration(
        fpu_policy=native.FpuPolicy.VISITED_CHILD_MEAN,
        adaptive=native.AdaptiveStoppingConfiguration(False, 1, 1, 1, 1),
        budget_class=native.SearchBudgetClass.MIXED_FAST,
        policy_weight=0,
    )

    result = native.search_go_fixed(state, _evaluator(0), configuration)

    assert result.telemetry.actual_simulations == 16
    assert result.telemetry.stop_reason == native.SearchStopReason.FULL_BUDGET
    assert not result.telemetry.policy_target_eligible
    assert result.telemetry.policy_target_weight == 0
