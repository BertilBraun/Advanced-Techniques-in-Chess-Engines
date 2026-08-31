from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip('AlphaZeroCpp')
from AlphaZeroCpp import FirstPlayUrgencyKind, FirstPlayUrgencyParameters, TreeSearchParameters
from src.experiment.configuration import load_experiment_configuration
from src.games.chess.configuration import ChessExperimentConfiguration
from src.games.chess.training import ChessImplementation
from src.search_budget.policy import BUDGET_CURVE_MULTIPLES, BUDGET_CURVE_POINTS, SearchBudgetPolicy
from test_helpers.configuration_paths import TEST_CONFIG_DIRECTORY


def _zero_first_play_urgency() -> FirstPlayUrgencyParameters:
    return FirstPlayUrgencyParameters(FirstPlayUrgencyKind.ZERO)


def test_tree_search_parameters_default_virtual_loss_weight_is_a_full_loss() -> None:
    parameters = TreeSearchParameters(
        exploration_constant=1.5,
        first_play_urgency=_zero_first_play_urgency(),
        forced_playout_coefficient=0.0,
        value_discount_per_ply=1.0,
    )

    assert parameters.virtual_loss_weight == 1.0


def test_tree_search_parameters_accept_a_fractional_virtual_loss_weight() -> None:
    parameters = TreeSearchParameters(
        exploration_constant=1.5,
        first_play_urgency=_zero_first_play_urgency(),
        forced_playout_coefficient=0.0,
        value_discount_per_ply=1.0,
        virtual_loss_weight=0.5,
    )

    assert parameters.virtual_loss_weight == 0.5


@pytest.mark.parametrize('invalid_weight', (-0.5, 1.5, float('nan')))
def test_tree_search_parameters_reject_invalid_virtual_loss_weights(invalid_weight: float) -> None:
    with pytest.raises(ValueError, match='Virtual-loss weight'):
        TreeSearchParameters(
            exploration_constant=1.5,
            first_play_urgency=_zero_first_play_urgency(),
            forced_playout_coefficient=0.0,
            value_discount_per_ply=1.0,
            virtual_loss_weight=invalid_weight,
        )


def _exported_corrector(tmp_path: Path) -> tuple[Path, str]:
    import numpy as np
    from src.search_budget.corrector import export_corrector, fit_curve_corrector
    from test_helpers.search_budget_records import analysis_records

    random = np.random.default_rng(20260831)
    count = 1024
    predicted = random.normal(-2.0, 1.0, size=(count, BUDGET_CURVE_POINTS))
    top_visit_share = random.uniform(0.2, 1.0, size=count)
    policy_entropy = random.uniform(0.0, 3.0, size=count)
    ply = random.integers(0, 200, size=count)
    residual = 0.5 + 0.3 * predicted - 1.2 * top_visit_share[:, None] + 0.2 * policy_entropy[:, None]
    records = analysis_records(predicted, predicted + residual, top_visit_share, policy_entropy, ply)
    fit = fit_curve_corrector(records)
    assert fit.network is not None
    path = tmp_path / 'corrector.jit.pt'
    sha256 = export_corrector(fit.network, path)
    return path, sha256


def test_resolved_virtual_loss_weight_reaches_the_native_search_parameters(tmp_path: Path) -> None:
    configuration = load_experiment_configuration(TEST_CONFIG_DIRECTORY / 'chess-experiment.yaml')
    assert isinstance(configuration, ChessExperimentConfiguration)
    implementation = ChessImplementation(configuration)
    corrector_path, corrector_sha256 = _exported_corrector(tmp_path)
    policy = SearchBudgetPolicy(
        lagrange_multiplier=0.4,
        corrector_path=corrector_path,
        corrector_sha256=corrector_sha256,
        apply_learned=True,
    )
    resolved = replace(implementation.self_play_parameters_at(0, policy), virtual_loss_weight=0.25)

    native_parameters = implementation.native_search_parameters(resolved)

    assert native_parameters.tree_search.virtual_loss_weight == pytest.approx(0.25)
    assert native_parameters.baseline_visits == resolved.baseline_visits
    native_policy = native_parameters.search_budget_policy
    assert tuple(native_policy.multiples) == pytest.approx(BUDGET_CURVE_MULTIPLES)
    assert native_policy.lagrange_multiplier == pytest.approx(0.4)
    assert native_policy.has_corrector is True
    assert native_policy.apply_learned is True


def test_an_identity_policy_reaches_native_without_a_corrector() -> None:
    from AlphaZeroCpp import SearchBudgetPolicy as NativeSearchBudgetPolicy

    native_policy = NativeSearchBudgetPolicy(list(BUDGET_CURVE_MULTIPLES), 0.3, '', True)
    assert native_policy.has_corrector is False


def _float32(value: float) -> float:
    import numpy as np

    return float(np.float32(value))


def test_native_and_python_budget_selection_agree_on_identical_inputs(tmp_path: Path) -> None:
    import random

    from AlphaZeroCpp import SearchBudgetPolicy as NativeSearchBudgetPolicy
    from AlphaZeroCpp import SearchBudgetSelectionFeatures
    from AlphaZeroCpp import correct_budget_curve as native_correct
    from AlphaZeroCpp import select_budget_index as native_select
    from src.search_budget.corrector import LoadedCurveCorrector
    from src.search_budget.policy import (
        BudgetSelectionFeatures,
        corrected_curve,
        identity_correction,
        select_budget_index,
    )

    corrector_path, corrector_sha256 = _exported_corrector(tmp_path)
    loaded_corrector = LoadedCurveCorrector.load(corrector_path, corrector_sha256)
    generator = random.Random(20260830)
    for _ in range(250):
        use_corrector = generator.random() < 0.5
        # The native curve is float32, so feed values that are exact in both precisions.
        prediction = tuple(_float32(generator.uniform(-8.0, 4.0)) for _ in range(BUDGET_CURVE_POINTS))
        lagrange_multiplier = generator.uniform(0.0, 2.0)
        python_policy = SearchBudgetPolicy(
            lagrange_multiplier=lagrange_multiplier,
            corrector_path=corrector_path if use_corrector else None,
            corrector_sha256=corrector_sha256 if use_corrector else None,
            apply_learned=True,
        )
        native_policy = NativeSearchBudgetPolicy(
            list(BUDGET_CURVE_MULTIPLES),
            lagrange_multiplier,
            str(corrector_path) if use_corrector else '',
            True,
        )
        ply = generator.randrange(0, 300)
        baseline_visits = generator.choice((300, 400, 600))
        source_generation = generator.randrange(0, 900)
        python_features = BudgetSelectionFeatures(
            top_visit_share=_float32(generator.uniform(0.05, 1.0)),
            policy_entropy=_float32(generator.uniform(0.0, 4.0)),
            ply=ply,
            baseline_visits=baseline_visits,
            source_generation=source_generation,
        )
        native_features = SearchBudgetSelectionFeatures(
            top_visit_share=python_features.top_visit_share,
            policy_entropy=python_features.policy_entropy,
            ply=float(ply),
            baseline_visits=float(baseline_visits),
            source_generation=float(source_generation),
        )
        correction = loaded_corrector if use_corrector else identity_correction
        assert native_select(native_policy, list(prediction), native_features) == select_budget_index(
            prediction, python_policy, python_features, correction
        )
        native_curve = native_correct(native_policy, list(prediction), native_features)
        assert tuple(native_curve) == pytest.approx(corrected_curve(prediction, python_features, correction), abs=1e-5)
