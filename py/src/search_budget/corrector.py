from __future__ import annotations

import hashlib
import io
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from src.search_budget.analysis_log import ANALYSIS_RECORD_DTYPE
from src.search_budget.policy import (
    BUDGET_CURVE_POINTS,
    CORRECTOR_INPUT_FEATURES,
    LOG_KL_EPSILON,
    BudgetSelectionFeatures,
)
from src.util.atomic_file import write_bytes_atomically

CORRECTOR_HIDDEN_WIDTH = 64
CORRECTOR_TRAINING_EPOCHS = 30
CORRECTOR_BATCH_SIZE = 4096
CORRECTOR_LEARNING_RATE = 1e-3
CORRECTOR_HOLDOUT_STRIDE = 5
CORRECTOR_MAXIMUM_CORRECTION = 2.0
_MINIMUM_FEATURE_SCALE = 1e-6
_RELATIVE_FEATURE_SCALE_FLOOR = 0.05


class CurveCorrectorNetwork(torch.nn.Module):
    """Additive log-KL curve correction; standardisation is folded in so native applies it raw."""

    __constants__ = ['maximum_correction']

    def __init__(self) -> None:
        super().__init__()
        self.maximum_correction = CORRECTOR_MAXIMUM_CORRECTION
        self.register_buffer('feature_mean', torch.zeros(CORRECTOR_INPUT_FEATURES))
        self.register_buffer('feature_scale', torch.ones(CORRECTOR_INPUT_FEATURES))
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(CORRECTOR_INPUT_FEATURES, CORRECTOR_HIDDEN_WIDTH),
            torch.nn.ReLU(),
            torch.nn.Linear(CORRECTOR_HIDDEN_WIDTH, CORRECTOR_HIDDEN_WIDTH),
            torch.nn.ReLU(),
            torch.nn.Linear(CORRECTOR_HIDDEN_WIDTH, BUDGET_CURVE_POINTS),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        standardised = (features - self.feature_mean) / self.feature_scale
        return self.layers(standardised).clamp(-self.maximum_correction, self.maximum_correction)


@dataclass(frozen=True)
class CorrectorFit:
    network: CurveCorrectorNetwork | None
    applied: bool
    rejection_reason: str | None
    uncorrected_holdout_residual: float
    corrected_holdout_residual: float


def _feature_scale(training_features: torch.Tensor) -> torch.Tensor:
    """A feature that is constant across the window must not standardise to a near-zero divisor.

    `baseline_visits` and `source_generation` are constant within a fitting window, so an absolute
    floor turns the first out-of-window value into a z-score of ~1e8 and the correction explodes.
    """
    scale = training_features.std(dim=0, unbiased=False)
    relative_floor = training_features.abs().mean(dim=0) * _RELATIVE_FEATURE_SCALE_FLOOR
    return torch.maximum(scale, relative_floor).clamp_min(_MINIMUM_FEATURE_SCALE)


def corrector_input_features(records: npt.NDArray[np.void]) -> npt.NDArray[np.float32]:
    if records.dtype != ANALYSIS_RECORD_DTYPE:
        raise ValueError('Corrector features require the fixed analysis record dtype.')
    return np.concatenate(
        [
            np.asarray(records['predicted_curve'], dtype=np.float32),
            np.asarray(records['top_visit_share'], dtype=np.float32)[:, None],
            np.asarray(records['policy_entropy'], dtype=np.float32)[:, None],
            np.asarray(records['ply'], dtype=np.float32)[:, None],
            np.asarray(records['baseline_visits'], dtype=np.float32)[:, None],
            np.asarray(records['source_generation'], dtype=np.float32)[:, None],
        ],
        axis=1,
    )


def corrector_residual_targets(records: npt.NDArray[np.void]) -> npt.NDArray[np.float32]:
    if records.dtype != ANALYSIS_RECORD_DTYPE:
        raise ValueError('Corrector targets require the fixed analysis record dtype.')
    targets = np.log(np.asarray(records['policy_kl'], dtype=np.float64) + LOG_KL_EPSILON)
    return (targets - np.asarray(records['predicted_curve'], dtype=np.float64)).astype(np.float32)


def selection_feature_vector(
    predicted_curve: tuple[float, ...],
    features: BudgetSelectionFeatures,
) -> tuple[float, ...]:
    if len(predicted_curve) != BUDGET_CURVE_POINTS:
        raise ValueError('A corrector input requires one prediction per grid point.')
    return (
        *predicted_curve,
        features.top_visit_share,
        features.policy_entropy,
        float(features.ply),
        float(features.baseline_visits),
        float(features.source_generation),
    )


def fit_curve_corrector(records: npt.NDArray[np.void], random_seed: int = 0) -> CorrectorFit:
    """Fit the joint-curve MLP corrector on a trailing analysis window.

    The holdout split is a fixed stride over the concatenated window, so the guard measures the
    correction on positions the optimizer never saw while spanning every window generation.
    """
    if records.dtype != ANALYSIS_RECORD_DTYPE:
        raise ValueError('Corrector fitting requires the fixed analysis record dtype.')
    if records.shape[0] < 2 * CORRECTOR_HOLDOUT_STRIDE:
        raise ValueError('Corrector fitting requires enough positions for a train/holdout split.')

    features = corrector_input_features(records)
    residuals = corrector_residual_targets(records)
    if not np.isfinite(features).all() or not np.isfinite(residuals).all():
        raise ValueError('Corrector fitting requires finite features and residuals.')
    holdout_mask = np.arange(records.shape[0]) % CORRECTOR_HOLDOUT_STRIDE == 0
    training_features = torch.from_numpy(features[~holdout_mask])
    training_residuals = torch.from_numpy(residuals[~holdout_mask])
    holdout_features = torch.from_numpy(features[holdout_mask])
    holdout_residuals = torch.from_numpy(residuals[holdout_mask])

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(random_seed)
        network = CurveCorrectorNetwork()
    with torch.no_grad():
        mean = training_features.mean(dim=0)
        network.feature_mean.copy_(mean)
        network.feature_scale.copy_(_feature_scale(training_features))

    generator = torch.Generator().manual_seed(random_seed)
    optimizer = torch.optim.Adam(network.parameters(), lr=CORRECTOR_LEARNING_RATE)
    loss_function = torch.nn.SmoothL1Loss()
    row_count = training_features.shape[0]
    for _ in range(CORRECTOR_TRAINING_EPOCHS):
        permutation = torch.randperm(row_count, generator=generator)
        for start in range(0, row_count, CORRECTOR_BATCH_SIZE):
            batch = permutation[start : start + CORRECTOR_BATCH_SIZE]
            optimizer.zero_grad()
            loss = loss_function(network(training_features[batch]), training_residuals[batch])
            loss.backward()
            optimizer.step()

    network.eval()
    with torch.no_grad():
        holdout_correction = network(holdout_features)
        uncorrected = float(holdout_residuals.square().mean())
        corrected = float((holdout_residuals - holdout_correction).square().mean())
    rejection_reason = _rejection_reason(network, uncorrected, corrected)
    return CorrectorFit(
        network=None if rejection_reason is not None else network,
        applied=rejection_reason is None,
        rejection_reason=rejection_reason,
        uncorrected_holdout_residual=uncorrected,
        corrected_holdout_residual=corrected,
    )


def _rejection_reason(network: CurveCorrectorNetwork, uncorrected: float, corrected: float) -> str | None:
    """A corrector that makes predictions worse must not reach production, so fail toward identity."""
    if not _has_finite_tensors(network):
        return 'the fitted corrector has a non-finite parameter'
    if not math.isfinite(corrected) or not math.isfinite(uncorrected) or corrected >= uncorrected:
        return 'the fitted corrector does not improve the held-out residual'
    return None


def _has_finite_tensors(network: CurveCorrectorNetwork) -> bool:
    return all(bool(torch.isfinite(tensor).all()) for tensor in (*network.parameters(), *network.buffers()))


def export_corrector(network: CurveCorrectorNetwork, path: Path) -> str:
    # The output clamp would mask a non-finite parameter from the probe below, so check the tensors.
    if not _has_finite_tensors(network):
        raise ValueError('A curve corrector must have finite parameters and buffers.')
    network.eval()
    scripted = torch.jit.script(network)
    buffer = io.BytesIO()
    torch.jit.save(scripted, buffer)
    content = buffer.getvalue()
    _validate_corrector_module(torch.jit.load(io.BytesIO(content), map_location='cpu'))
    write_bytes_atomically(path, content)
    return hashlib.sha256(content).hexdigest()


def _validate_corrector_module(module: torch.jit.ScriptModule) -> None:
    with torch.no_grad():
        probe = module(torch.zeros((1, CORRECTOR_INPUT_FEATURES), dtype=torch.float32))
    if probe.shape != (1, BUDGET_CURVE_POINTS) or not torch.isfinite(probe).all():
        raise ValueError('A curve corrector must map the feature vector to one finite correction per grid point.')


class LoadedCurveCorrector:
    """Python-side evaluation of a published corrector; callable as a CurveCorrection."""

    def __init__(self, module: torch.jit.ScriptModule) -> None:
        _validate_corrector_module(module)
        self._module = module

    @classmethod
    def load(cls, path: Path, expected_sha256: str | None = None) -> LoadedCurveCorrector:
        content = path.read_bytes()
        if expected_sha256 is not None and hashlib.sha256(content).hexdigest() != expected_sha256:
            raise ValueError(f'Corrector artifact does not match its published digest: {path}')
        return cls(torch.jit.load(io.BytesIO(content), map_location='cpu'))

    def __call__(self, predicted_curve: tuple[float, ...], features: BudgetSelectionFeatures) -> tuple[float, ...]:
        inputs = torch.tensor([selection_feature_vector(predicted_curve, features)], dtype=torch.float32)
        with torch.no_grad():
            correction = self._module(inputs)[0]
        return tuple(float(predicted) + float(delta) for predicted, delta in zip(predicted_curve, correction))
