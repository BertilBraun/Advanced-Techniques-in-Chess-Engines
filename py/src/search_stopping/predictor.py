from __future__ import annotations

import hashlib
import io
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from src.search_stopping.features import STOP_PREDICTOR_FEATURE_COUNT
from src.util.atomic_file import write_bytes_atomically

STOP_PREDICTOR_HIDDEN_WIDTH = 64
STOP_PREDICTOR_TRAINING_EPOCHS = 30
STOP_PREDICTOR_BATCH_SIZE = 4096
STOP_PREDICTOR_LEARNING_RATE = 1e-3
STOP_PREDICTOR_HOLDOUT_GROUP_MODULUS = 5
_MINIMUM_FEATURE_SCALE = 1e-6
_RELATIVE_FEATURE_SCALE_FLOOR = 0.05
_PROBABILITY_FLOOR = 1e-6


class StopPredictorNetwork(torch.nn.Module):
    """Uncertainty probability u for one checkpoint; standardisation is folded in so native applies it raw."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer('feature_mean', torch.zeros(STOP_PREDICTOR_FEATURE_COUNT))
        self.register_buffer('feature_scale', torch.ones(STOP_PREDICTOR_FEATURE_COUNT))
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(STOP_PREDICTOR_FEATURE_COUNT, STOP_PREDICTOR_HIDDEN_WIDTH),
            torch.nn.ReLU(),
            torch.nn.Linear(STOP_PREDICTOR_HIDDEN_WIDTH, STOP_PREDICTOR_HIDDEN_WIDTH),
            torch.nn.ReLU(),
            torch.nn.Linear(STOP_PREDICTOR_HIDDEN_WIDTH, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        standardised = (features - self.feature_mean) / self.feature_scale
        return torch.sigmoid(self.layers(standardised))


@dataclass(frozen=True)
class StopPredictorFit:
    network: StopPredictorNetwork | None
    applied: bool
    rejection_reason: str | None
    holdout_bce: float
    base_rate_bce: float


def _feature_scale(training_features: torch.Tensor) -> torch.Tensor:
    """A feature that is constant across the window must not standardise to a near-zero divisor."""
    scale = training_features.std(dim=0, unbiased=False)
    relative_floor = training_features.abs().mean(dim=0) * _RELATIVE_FEATURE_SCALE_FLOOR
    return torch.maximum(scale, relative_floor).clamp_min(_MINIMUM_FEATURE_SCALE)


def _binary_cross_entropy(probabilities: torch.Tensor, labels: torch.Tensor) -> float:
    clamped = probabilities.clamp(_PROBABILITY_FLOOR, 1.0 - _PROBABILITY_FLOOR)
    return float(torch.nn.functional.binary_cross_entropy(clamped, labels))


def fit_stop_predictor(
    features: npt.NDArray[np.float32],
    uncertain_labels: npt.NDArray[np.float32],
    group_keys: npt.NDArray[np.uint64],
    random_seed: int = 0,
) -> StopPredictorFit:
    """Fit the stop-predictor MLP on a trailing audit window.

    The holdout is split by group key (game identity hash), never by row: one audit search yields
    sibling examples with deterministically nested labels, and a row-wise split would leak them
    across the boundary and inflate the holdout gate.
    """
    if features.ndim != 2 or features.shape[1] != STOP_PREDICTOR_FEATURE_COUNT:
        raise ValueError('Stop-predictor fitting requires one fixed-width feature row per example.')
    if uncertain_labels.shape != (features.shape[0],) or group_keys.shape != (features.shape[0],):
        raise ValueError('Stop-predictor fitting requires one label and one group key per feature row.')
    if not np.isfinite(features).all() or not np.isfinite(uncertain_labels).all():
        raise ValueError('Stop-predictor fitting requires finite features and labels.')
    if not np.isin(uncertain_labels, (0.0, 1.0)).all():
        raise ValueError('Stop-predictor labels must be binary.')

    holdout_mask = group_keys.astype(np.uint64) % STOP_PREDICTOR_HOLDOUT_GROUP_MODULUS == 0
    if not holdout_mask.any() or holdout_mask.all():
        raise ValueError('Stop-predictor fitting requires groups on both sides of the holdout split.')
    training_features = torch.from_numpy(features[~holdout_mask])
    training_labels = torch.from_numpy(uncertain_labels[~holdout_mask])[:, None]
    holdout_features = torch.from_numpy(features[holdout_mask])
    holdout_labels = torch.from_numpy(uncertain_labels[holdout_mask])[:, None]

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(random_seed)
        network = StopPredictorNetwork()
    with torch.no_grad():
        network.feature_mean.copy_(training_features.mean(dim=0))
        network.feature_scale.copy_(_feature_scale(training_features))

    generator = torch.Generator().manual_seed(random_seed)
    optimizer = torch.optim.Adam(network.parameters(), lr=STOP_PREDICTOR_LEARNING_RATE)
    row_count = training_features.shape[0]
    for _ in range(STOP_PREDICTOR_TRAINING_EPOCHS):
        permutation = torch.randperm(row_count, generator=generator)
        for start in range(0, row_count, STOP_PREDICTOR_BATCH_SIZE):
            batch = permutation[start : start + STOP_PREDICTOR_BATCH_SIZE]
            optimizer.zero_grad()
            probabilities = network(training_features[batch]).clamp(_PROBABILITY_FLOOR, 1.0 - _PROBABILITY_FLOOR)
            loss = torch.nn.functional.binary_cross_entropy(probabilities, training_labels[batch])
            loss.backward()
            optimizer.step()

    network.eval()
    with torch.no_grad():
        holdout_bce = _binary_cross_entropy(network(holdout_features), holdout_labels)
        base_rate = holdout_labels.mean().clamp(_PROBABILITY_FLOOR, 1.0 - _PROBABILITY_FLOOR)
        base_rate_bce = _binary_cross_entropy(base_rate.expand_as(holdout_labels), holdout_labels)
    rejection_reason = _rejection_reason(network, holdout_bce, base_rate_bce)
    return StopPredictorFit(
        network=None if rejection_reason is not None else network,
        applied=rejection_reason is None,
        rejection_reason=rejection_reason,
        holdout_bce=holdout_bce,
        base_rate_bce=base_rate_bce,
    )


def _rejection_reason(network: StopPredictorNetwork, holdout_bce: float, base_rate_bce: float) -> str | None:
    """A predictor that cannot beat the window base rate must not reach production."""
    if not _has_finite_tensors(network):
        return 'the fitted stop predictor has a non-finite parameter'
    if not math.isfinite(holdout_bce) or not math.isfinite(base_rate_bce) or holdout_bce >= base_rate_bce:
        return 'the fitted stop predictor does not improve on the held-out base rate'
    return None


def _has_finite_tensors(network: StopPredictorNetwork) -> bool:
    return all(bool(torch.isfinite(tensor).all()) for tensor in (*network.parameters(), *network.buffers()))


def export_stop_predictor(network: StopPredictorNetwork, path: Path) -> str:
    if not _has_finite_tensors(network):
        raise ValueError('A stop predictor must have finite parameters and buffers.')
    network.eval()
    scripted = torch.jit.script(network)
    buffer = io.BytesIO()
    torch.jit.save(scripted, buffer)
    content = buffer.getvalue()
    _validate_stop_predictor_module(torch.jit.load(io.BytesIO(content), map_location='cpu'))
    write_bytes_atomically(path, content)
    return hashlib.sha256(content).hexdigest()


def _validate_stop_predictor_module(module: torch.jit.ScriptModule) -> None:
    with torch.no_grad():
        probe = module(torch.zeros((1, STOP_PREDICTOR_FEATURE_COUNT), dtype=torch.float32))
    if (
        probe.shape != (1, 1)
        or not torch.isfinite(probe).all()
        or not bool((0.0 <= probe).all())
        or not bool((probe <= 1.0).all())
    ):
        raise ValueError('A stop predictor must map the feature vector to one probability.')


class LoadedStopPredictor:
    """Python-side evaluation of a published stop predictor."""

    def __init__(self, module: torch.jit.ScriptModule) -> None:
        _validate_stop_predictor_module(module)
        self._module = module

    @classmethod
    def load(cls, path: Path, expected_sha256: str | None = None) -> LoadedStopPredictor:
        content = path.read_bytes()
        if expected_sha256 is not None and hashlib.sha256(content).hexdigest() != expected_sha256:
            raise ValueError(f'Stop-predictor artifact does not match its published digest: {path}')
        return cls(torch.jit.load(io.BytesIO(content), map_location='cpu'))

    def __call__(self, features: tuple[float, ...]) -> float:
        inputs = torch.tensor([features], dtype=torch.float32)
        with torch.no_grad():
            return float(self._module(inputs)[0, 0])
