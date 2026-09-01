from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from src.search_stopping.features import STOP_PREDICTOR_FEATURE_COUNT
from src.search_stopping.predictor import (
    LoadedStopPredictor,
    export_stop_predictor,
    fit_stop_predictor,
)


def _separable_window(count: int = 600) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(11)
    features = generator.normal(size=(count, STOP_PREDICTOR_FEATURE_COUNT)).astype(np.float32)
    labels = (features[:, 0] > 0.0).astype(np.float32)
    groups = (np.arange(count) // 6).astype(np.uint64)  # six sibling rows per game
    return features, labels, groups


def test_fit_learns_a_separable_window() -> None:
    features, labels, groups = _separable_window()
    fit = fit_stop_predictor(features, labels, groups)
    assert fit.applied and fit.holdout_bce < fit.base_rate_bce


def test_fit_rejects_an_unlearnable_window() -> None:
    features, labels, groups = _separable_window()
    generator = np.random.default_rng(5)
    shuffled = labels.copy()
    generator.shuffle(shuffled)
    fit = fit_stop_predictor(features, shuffled, groups)
    assert not fit.applied and fit.rejection_reason is not None


def test_holdout_split_keeps_sibling_groups_on_one_side() -> None:
    features, labels, groups = _separable_window()
    holdout_groups = set(groups[groups % 5 == 0].tolist())
    training_groups = set(groups[groups % 5 != 0].tolist())
    assert holdout_groups.isdisjoint(training_groups)


def test_fit_rejects_non_finite_features() -> None:
    features, labels, groups = _separable_window()
    features[0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        fit_stop_predictor(features, labels, groups)


def test_fit_rejects_non_binary_labels() -> None:
    features, labels, groups = _separable_window()
    labels[0] = 0.5
    with pytest.raises(ValueError, match='binary'):
        fit_stop_predictor(features, labels, groups)


def test_fit_requires_groups_on_both_holdout_sides() -> None:
    features, labels, _ = _separable_window(60)
    with pytest.raises(ValueError, match='both sides'):
        fit_stop_predictor(features, labels, np.ones(60, dtype=np.uint64))


def test_exported_predictor_round_trips_with_matching_probabilities(tmp_path: Path) -> None:
    features, labels, groups = _separable_window()
    fit = fit_stop_predictor(features, labels, groups)
    assert fit.network is not None
    path = tmp_path / 'stop-predictor.jit.pt'
    digest = export_stop_predictor(fit.network, path)
    loaded = LoadedStopPredictor.load(path, expected_sha256=digest)
    probe = tuple(float(value) for value in features[0])
    with torch.no_grad():
        expected = float(fit.network(torch.tensor([probe]))[0, 0])
    assert loaded(probe) == pytest.approx(expected, abs=1e-6)


def test_loading_rejects_a_digest_mismatch(tmp_path: Path) -> None:
    features, labels, groups = _separable_window()
    fit = fit_stop_predictor(features, labels, groups)
    assert fit.network is not None
    path = tmp_path / 'stop-predictor.jit.pt'
    export_stop_predictor(fit.network, path)
    with pytest.raises(ValueError, match='digest'):
        LoadedStopPredictor.load(path, expected_sha256='d' * 64)
