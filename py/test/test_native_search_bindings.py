from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('AlphaZeroCpp')
from AlphaZeroCpp import FirstPlayUrgencyKind, FirstPlayUrgencyParameters, TreeSearchParameters
from src.experiment.configuration import load_experiment_configuration
from src.games.chess.configuration import ChessExperimentConfiguration
from src.games.chess.training import ChessImplementation
from src.search_stopping.features import STOP_PREDICTOR_FEATURE_COUNT
from src.search_stopping.policy import SearchStopPolicy, flat_stop_policy
from src.search_stopping.predictor import LoadedStopPredictor, export_stop_predictor, fit_stop_predictor
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


def _exported_stop_predictor(tmp_path: Path) -> tuple[Path, str]:
    random = np.random.default_rng(20260901)
    count = 600
    features = random.normal(size=(count, STOP_PREDICTOR_FEATURE_COUNT)).astype(np.float32)
    labels = (features[:, 0] > 0.0).astype(np.float32)
    groups = (np.arange(count) // 6).astype(np.uint64)
    fit = fit_stop_predictor(features, labels, groups)
    assert fit.network is not None
    path = tmp_path / 'stop-predictor.jit.pt'
    sha256 = export_stop_predictor(fit.network, path)
    return path, sha256


def _open_policy(tmp_path: Path) -> SearchStopPolicy:
    predictor_path, predictor_sha256 = _exported_stop_predictor(tmp_path)
    return SearchStopPolicy(
        checkpoint_multiples=(1.0 / 3.0, 0.5, 2.0 / 3.0, 1.0, 1.5),
        thresholds=(0.1, 0.1, 0.2, 0.2, 0.3),
        movement_guard_epsilon=0.05,
        cap_multiple=2.0,
        predictor_path=predictor_path,
        predictor_sha256=predictor_sha256,
        apply_learned=True,
    )


def test_the_feature_count_is_a_shared_binding_constant() -> None:
    import AlphaZeroCpp

    assert AlphaZeroCpp.STOP_PREDICTOR_FEATURE_COUNT == STOP_PREDICTOR_FEATURE_COUNT


def test_resolved_stop_policy_reaches_the_native_search_parameters(tmp_path: Path) -> None:
    configuration = load_experiment_configuration(TEST_CONFIG_DIRECTORY / 'chess-experiment.yaml')
    assert isinstance(configuration, ChessExperimentConfiguration)
    implementation = ChessImplementation(configuration)
    policy = _open_policy(tmp_path)

    parameters = implementation.self_play_parameters_at(0, policy)
    native_parameters = implementation.native_search_parameters(parameters)

    native_policy = native_parameters.search_stop_policy
    assert tuple(native_policy.checkpoint_multiples) == pytest.approx(policy.checkpoint_multiples)
    assert tuple(native_policy.thresholds) == pytest.approx(policy.thresholds)
    assert native_policy.movement_guard_epsilon == pytest.approx(policy.movement_guard_epsilon)
    assert native_policy.cap_multiple == pytest.approx(policy.cap_multiple)
    assert native_policy.has_predictor
    assert native_policy.apply_learned
    implementation.close()


def test_a_closed_policy_reaches_native_without_a_predictor() -> None:
    from AlphaZeroCpp import SearchStopPolicy as NativeSearchStopPolicy

    policy = flat_stop_policy()
    native_policy = NativeSearchStopPolicy(
        list(policy.checkpoint_multiples),
        list(policy.thresholds),
        policy.movement_guard_epsilon,
        policy.cap_multiple,
        '',
        policy.apply_learned,
    )

    assert not native_policy.has_predictor
    assert not native_policy.apply_learned


def test_an_applied_native_policy_requires_a_predictor() -> None:
    from AlphaZeroCpp import SearchStopPolicy as NativeSearchStopPolicy

    with pytest.raises(ValueError, match='predictor'):
        NativeSearchStopPolicy([0.5, 1.0], [0.1, 0.1], 0.05, 2.0, '', True)


def test_native_and_python_stop_predictors_agree_on_identical_inputs(tmp_path: Path) -> None:
    from AlphaZeroCpp import SearchStopPolicy as NativeSearchStopPolicy
    from AlphaZeroCpp import evaluate_stop_rule

    predictor_path, predictor_sha256 = _exported_stop_predictor(tmp_path)
    loaded = LoadedStopPredictor.load(predictor_path, predictor_sha256)
    native_policy = NativeSearchStopPolicy([0.5, 1.0], [0.5, 0.5], 0.05, 2.0, str(predictor_path), True)

    generator = np.random.default_rng(7)
    for _ in range(32):
        features = tuple(float(value) for value in generator.normal(size=STOP_PREDICTOR_FEATURE_COUNT))
        movement = float(generator.uniform(0.0, 0.1))
        features = (*features[:4], movement, *features[5:])
        evaluation = evaluate_stop_rule(native_policy, 0, features)
        assert evaluation.guard_movement == pytest.approx(movement)
        assert evaluation.guard_passed == (movement < 0.05)
        if evaluation.guard_passed:
            assert evaluation.predictor_evaluated
            assert evaluation.uncertainty == pytest.approx(loaded(features), abs=1e-6)
            assert evaluation.would_stop == (evaluation.uncertainty < 0.5)
        else:
            assert not evaluation.predictor_evaluated
            assert not evaluation.would_stop


def test_the_stop_rule_rejects_an_out_of_range_checkpoint_index(tmp_path: Path) -> None:
    from AlphaZeroCpp import SearchStopPolicy as NativeSearchStopPolicy
    from AlphaZeroCpp import evaluate_stop_rule

    predictor_path, _ = _exported_stop_predictor(tmp_path)
    native_policy = NativeSearchStopPolicy([0.5], [0.1], 0.05, 2.0, str(predictor_path), True)
    with pytest.raises(ValueError, match='checkpoint index'):
        evaluate_stop_rule(native_policy, 1, tuple(0.0 for _ in range(STOP_PREDICTOR_FEATURE_COUNT)))
