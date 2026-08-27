from __future__ import annotations

import argparse
import gzip
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import torch
from pydantic import Field
from src.games.chess.contract import ChessStateContract
from src.games.representation import decode_packed_planes_batch
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel
from src.util.hashing import file_sha256
from src.util.provenance import read_source_revision
from tools.measure_policy_target_fidelity import PerPositionReport, PositionRecord
from torch import Tensor, nn

_LAGRANGE_STEPS = 4_000
_LAGRANGE_MINIMUM_EXPONENT = -7.0
_LAGRANGE_EXPONENT_STEP = 0.002


@dataclass(frozen=True)
class Arguments:
    model: Path
    per_position: Path
    output: Path
    device: torch.device
    baseline_visits: int
    depth_visits: int
    folds: int
    epochs: int
    batch_size: int
    feature_batch_size: int
    learning_rate: float
    weight_decay: float
    bootstrap_samples: int
    random_orderings: int
    seed: int


@dataclass(frozen=True)
class AllocationData:
    budgets: npt.NDArray[np.float64]
    monotone_divergences: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.budgets.ndim != 1 or len(self.budgets) < 2:
            raise ValueError('Allocation budgets must be a one-dimensional array with at least two entries.')
        if self.monotone_divergences.ndim != 2 or self.monotone_divergences.shape[1] != len(self.budgets):
            raise ValueError('Allocation divergences must have one column per budget.')
        if len(self.monotone_divergences) < 2:
            raise ValueError('Allocation scoring needs at least two positions.')


@dataclass(frozen=True)
class AllocationResult:
    mean_visits: float
    per_position_divergence: npt.NDArray[np.float64]
    assigned_budgets: npt.NDArray[np.float64]


class FoldResult(FrozenModel):
    fold: int = Field(ge=0)
    training_positions: int = Field(gt=0)
    held_out_positions: int = Field(gt=0)
    final_training_loss: float = Field(ge=0.0)
    held_out_mean_squared_error: float = Field(ge=0.0)
    held_out_spearman: float = Field(ge=-1.0, le=1.0)


class GainCapture(FrozenModel):
    mean_visits: float = Field(gt=0.0)
    mean_kullback_leibler: float = Field(ge=0.0)
    oracle_gain_fraction: float


class BootstrapInterval(FrozenModel):
    samples: int = Field(gt=0)
    lower: float
    median: float
    upper: float


class RandomControl(FrozenModel):
    orderings: int = Field(gt=0)
    mean_oracle_gain_fraction: float
    minimum_oracle_gain_fraction: float
    maximum_oracle_gain_fraction: float


class ProbeReport(FrozenModel):
    schema_version: Literal[1] = 1
    source_revision: str = Field(min_length=40, max_length=40)
    tool_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    model_path: Path
    model_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    per_position_path: Path
    per_position_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    source_generation: int = Field(ge=0)
    positions: int = Field(gt=0)
    baseline_visits: int = Field(gt=0)
    depth_visits: int = Field(gt=0)
    feature_shape: tuple[int, int, int]
    folds: tuple[FoldResult, ...] = Field(min_length=2)
    label_spearman: float = Field(ge=-1.0, le=1.0)
    flat_mean_kullback_leibler: float = Field(ge=0.0)
    oracle: GainCapture
    deep_label: GainCapture
    predictor: GainCapture
    predictor_capture_interval: BootstrapInterval
    predictor_beats_flat_at_95_percent: bool
    random_control: RandomControl


class FrozenScalarHead(nn.Module):
    def __init__(self, input_channels: int, rows: int, columns: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(input_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.output = nn.Linear(rows * columns, 1)

    def forward(self, features: Tensor) -> Tensor:
        projected = self.projection(features).flatten(start_dim=1)
        return torch.sigmoid(self.output(projected)).flatten()


def _read_position_report(path: Path) -> PerPositionReport:
    if path.suffix == '.gz':
        with gzip.open(path, mode='rt', encoding='utf-8') as handle:
            return PerPositionReport.model_validate_json(handle.read())
    return PerPositionReport.model_validate_json(path.read_text(encoding='utf-8'))


def quantile_ranks(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    if values.ndim != 1 or len(values) < 2:
        raise ValueError('Quantile ranking needs at least two scalar values.')
    order = np.argsort(values, kind='stable')
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks / (len(values) - 1)


def _spearman(left: npt.NDArray[np.float64], right: npt.NDArray[np.float64]) -> float:
    if left.shape != right.shape:
        raise ValueError('Spearman inputs must have the same shape.')
    left_ranks = quantile_ranks(left)
    right_ranks = quantile_ranks(right)
    left_centered = left_ranks - np.mean(left_ranks)
    right_centered = right_ranks - np.mean(right_ranks)
    denominator = math.sqrt(float(np.sum(left_centered**2) * np.sum(right_centered**2)))
    if denominator == 0.0:
        return 0.0
    return float(np.sum(left_centered * right_centered) / denominator)


def _label_values(
    report: PerPositionReport,
    baseline_visits: int,
    depth_visits: int,
) -> npt.NDArray[np.float64]:
    labels: list[float] = []
    for record in report.records:
        matches = tuple(
            candidate
            for candidate in record.label_candidates
            if candidate.baseline_visits == baseline_visits and candidate.depth_visits == depth_visits
        )
        if len(matches) != 1:
            raise ValueError(
                f'Every position must have exactly one {baseline_visits}->{depth_visits} KL label; found {len(matches)}.'
            )
        labels.append(matches[0].kullback_leibler)
    return np.asarray(labels, dtype=np.float64)


def _allocation_data(report: PerPositionReport) -> AllocationData:
    budgets = np.asarray([budget.visits for budget in report.records[0].budgets], dtype=np.float64)
    if np.any(np.diff(budgets) <= 0):
        raise ValueError('Per-position budgets must be strictly increasing.')
    rows: list[list[float]] = []
    expected_budgets = tuple(int(value) for value in budgets)
    for record in report.records:
        record_budgets = tuple(budget.visits for budget in record.budgets)
        if record_budgets != expected_budgets:
            raise ValueError('Every position must use the same ordered budget menu.')
        rows.append([budget.kullback_leibler for budget in record.budgets])
    divergences = np.asarray(rows, dtype=np.float64)
    # More search can move a finite-depth target away from the reference by chance. A useful allocator cannot
    # predict that noise, so score against the same non-increasing correction used by the design measurement.
    monotone = np.maximum.accumulate(divergences[:, ::-1], axis=1)[:, ::-1]
    return AllocationData(budgets=budgets, monotone_divergences=monotone)


def _oracle_allocation(data: AllocationData, target_mean_visits: float) -> AllocationResult:
    best_distance = math.inf
    best_indices: npt.NDArray[np.int64] | None = None
    for step in range(_LAGRANGE_STEPS):
        multiplier = 10.0 ** (_LAGRANGE_MINIMUM_EXPONENT + step * _LAGRANGE_EXPONENT_STEP)
        indices = np.argmin(data.monotone_divergences + multiplier * data.budgets[None, :], axis=1)
        mean_visits = float(np.mean(data.budgets[indices]))
        distance = abs(mean_visits - target_mean_visits)
        if distance < best_distance:
            best_distance = distance
            best_indices = indices.astype(np.int64, copy=True)
    assert best_indices is not None
    positions = np.arange(len(data.monotone_divergences))
    return AllocationResult(
        mean_visits=float(np.mean(data.budgets[best_indices])),
        per_position_divergence=data.monotone_divergences[positions, best_indices],
        assigned_budgets=data.budgets[best_indices],
    )


def _ranked_allocation(
    data: AllocationData,
    oracle_budgets: npt.NDArray[np.float64],
    signal: npt.NDArray[np.float64],
) -> AllocationResult:
    if signal.shape != (len(data.monotone_divergences),):
        raise ValueError('Allocation signal must contain one scalar per position.')
    ordered_positions = np.argsort(-signal, kind='stable')
    ordered_budgets = np.sort(oracle_budgets)[::-1]
    assigned_budgets = np.empty_like(ordered_budgets)
    assigned_budgets[ordered_positions] = ordered_budgets
    budget_indices = np.searchsorted(data.budgets, assigned_budgets).astype(np.int64)
    if np.any(data.budgets[budget_indices] != assigned_budgets):
        raise ValueError('Assigned budgets must come from the measured budget menu.')
    positions = np.arange(len(data.monotone_divergences))
    return AllocationResult(
        mean_visits=float(np.mean(assigned_budgets)),
        per_position_divergence=data.monotone_divergences[positions, budget_indices],
        assigned_budgets=assigned_budgets,
    )


def _gain_fraction(flat: npt.NDArray[np.float64], oracle: AllocationResult, candidate: AllocationResult) -> float:
    available_gain = float(np.mean(flat) - np.mean(oracle.per_position_divergence))
    if available_gain <= 0.0:
        raise ValueError('Oracle allocation must improve on the flat baseline.')
    return float((np.mean(flat) - np.mean(candidate.per_position_divergence)) / available_gain)


def _gain_capture(
    flat: npt.NDArray[np.float64],
    oracle: AllocationResult,
    candidate: AllocationResult,
) -> GainCapture:
    return GainCapture(
        mean_visits=candidate.mean_visits,
        mean_kullback_leibler=float(np.mean(candidate.per_position_divergence)),
        oracle_gain_fraction=_gain_fraction(flat, oracle, candidate),
    )


def _bootstrap_interval(
    flat: npt.NDArray[np.float64],
    oracle: AllocationResult,
    candidate: AllocationResult,
    samples: int,
    seed: int,
) -> BootstrapInterval:
    generator = np.random.default_rng(seed)
    captures = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        indices = generator.integers(0, len(flat), size=len(flat))
        available_gain = float(np.mean(flat[indices]) - np.mean(oracle.per_position_divergence[indices]))
        captures[sample] = (
            float(np.mean(flat[indices]) - np.mean(candidate.per_position_divergence[indices])) / available_gain
        )
    lower, median, upper = np.quantile(captures, (0.025, 0.5, 0.975))
    return BootstrapInterval(samples=samples, lower=float(lower), median=float(median), upper=float(upper))


def _random_control(
    data: AllocationData,
    flat: npt.NDArray[np.float64],
    oracle: AllocationResult,
    orderings: int,
    seed: int,
) -> RandomControl:
    generator = np.random.default_rng(seed)
    captures = []
    for _ in range(orderings):
        signal = generator.random(len(flat))
        allocation = _ranked_allocation(data, oracle.assigned_budgets, signal)
        captures.append(_gain_fraction(flat, oracle, allocation))
    return RandomControl(
        orderings=orderings,
        mean_oracle_gain_fraction=float(np.mean(captures)),
        minimum_oracle_gain_fraction=float(np.min(captures)),
        maximum_oracle_gain_fraction=float(np.max(captures)),
    )


def _flat_divergence(data: AllocationData, baseline_visits: int) -> npt.NDArray[np.float64]:
    matches = np.flatnonzero(data.budgets == baseline_visits)
    if len(matches) != 1:
        raise ValueError(f'Baseline budget {baseline_visits} must occur exactly once in the measured menu.')
    return data.monotone_divergences[:, int(matches[0])]


def _extract_features(
    model: torch.jit.ScriptModule,
    states: Tensor,
) -> Tensor:
    features = model.start_block(states)
    for block in model.backbone.children():
        features = block(features)
    return model.finish_block(features)


def extract_frozen_features(
    model_path: Path,
    records: tuple[PositionRecord, ...],
    device: torch.device,
    batch_size: int,
) -> Tensor:
    from AlphaZeroCpp import ChessPosition

    state = ChessStateContract()
    encoded_states = tuple(state.encode_network_input(ChessPosition(record.fen)) for record in records)
    decoded_states = decode_packed_planes_batch(
        encoded_states,
        state.packed_plane_layout,
        state.representation.binary_channels,
        state.representation.scalar_channels,
    )
    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()
    feature_batches: list[Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(decoded_states), batch_size):
            inputs = torch.from_numpy(decoded_states[start : start + batch_size]).to(
                device=device,
                dtype=torch.float32,
            )
            feature_batches.append(_extract_features(model, inputs).float().cpu())
    features = torch.cat(feature_batches)
    if features.ndim != 4 or features.shape[0] != len(records):
        raise ValueError(f'Frozen trunk returned unexpected feature shape {tuple(features.shape)}.')
    return features


def _train_fold(
    features: Tensor,
    labels: Tensor,
    training_indices: Tensor,
    held_out_indices: Tensor,
    arguments: Arguments,
    fold: int,
) -> tuple[Tensor, FoldResult]:
    torch.manual_seed(arguments.seed + fold)
    head = FrozenScalarHead(features.shape[1], features.shape[2], features.shape[3]).to(arguments.device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
    )
    generator = torch.Generator().manual_seed(arguments.seed + fold)
    final_loss = math.inf
    head.train()
    for _ in range(arguments.epochs):
        permutation = training_indices[torch.randperm(len(training_indices), generator=generator)]
        epoch_loss = 0.0
        for start in range(0, len(permutation), arguments.batch_size):
            indices = permutation[start : start + arguments.batch_size]
            batch_features = features.index_select(0, indices).to(arguments.device)
            batch_labels = labels.index_select(0, indices).to(arguments.device)
            optimizer.zero_grad(set_to_none=True)
            predictions = head(batch_features)
            loss = torch.mean((predictions - batch_labels) ** 2)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(indices)
        final_loss = epoch_loss / len(training_indices)
    head.eval()
    with torch.inference_mode():
        predictions = head(features.index_select(0, held_out_indices).to(arguments.device)).cpu()
    held_out_labels = labels.index_select(0, held_out_indices)
    mean_squared_error = float(torch.mean((predictions - held_out_labels) ** 2))
    return predictions, FoldResult(
        fold=fold,
        training_positions=len(training_indices),
        held_out_positions=len(held_out_indices),
        final_training_loss=final_loss,
        held_out_mean_squared_error=mean_squared_error,
        held_out_spearman=_spearman(predictions.numpy().astype(np.float64), held_out_labels.numpy().astype(np.float64)),
    )


def fit_out_of_fold_predictions(
    features: Tensor,
    labels: npt.NDArray[np.float64],
    arguments: Arguments,
) -> tuple[npt.NDArray[np.float64], tuple[FoldResult, ...]]:
    if arguments.folds > len(features) // 2:
        raise ValueError('Every cross-validation fold must contain at least two held-out positions.')
    generator = torch.Generator().manual_seed(arguments.seed)
    shuffled = torch.randperm(len(features), generator=generator)
    fold_indices = tuple(torch.tensor_split(shuffled, arguments.folds))
    label_tensor = torch.from_numpy(labels.astype(np.float32))
    predictions = torch.empty(len(features), dtype=torch.float32)
    results: list[FoldResult] = []
    for fold, held_out_indices in enumerate(fold_indices):
        training_indices = torch.cat(tuple(indices for index, indices in enumerate(fold_indices) if index != fold))
        held_out_predictions, result = _train_fold(
            features,
            label_tensor,
            training_indices,
            held_out_indices,
            arguments,
            fold,
        )
        predictions[held_out_indices] = held_out_predictions
        results.append(result)
    return predictions.numpy().astype(np.float64), tuple(results)


def build_report(
    arguments: Arguments,
    position_report: PerPositionReport,
    features: Tensor,
) -> ProbeReport:
    raw_labels = _label_values(position_report, arguments.baseline_visits, arguments.depth_visits)
    labels = quantile_ranks(raw_labels)
    predictions, folds = fit_out_of_fold_predictions(features, labels, arguments)
    allocation_data = _allocation_data(position_report)
    flat = _flat_divergence(allocation_data, arguments.baseline_visits)
    oracle = _oracle_allocation(allocation_data, float(arguments.baseline_visits))
    label_allocation = _ranked_allocation(allocation_data, oracle.assigned_budgets, labels)
    predicted_allocation = _ranked_allocation(allocation_data, oracle.assigned_budgets, predictions)
    interval = _bootstrap_interval(
        flat,
        oracle,
        predicted_allocation,
        arguments.bootstrap_samples,
        arguments.seed + 1,
    )
    return ProbeReport(
        source_revision=read_source_revision().commit,
        tool_sha256=file_sha256(Path(__file__)),
        model_path=arguments.model.resolve(),
        model_sha256=file_sha256(arguments.model),
        per_position_path=arguments.per_position.resolve(),
        per_position_sha256=file_sha256(arguments.per_position),
        source_generation=position_report.generation,
        positions=len(position_report.records),
        baseline_visits=arguments.baseline_visits,
        depth_visits=arguments.depth_visits,
        feature_shape=(features.shape[1], features.shape[2], features.shape[3]),
        folds=folds,
        label_spearman=_spearman(predictions, labels),
        flat_mean_kullback_leibler=float(np.mean(flat)),
        oracle=GainCapture(
            mean_visits=oracle.mean_visits,
            mean_kullback_leibler=float(np.mean(oracle.per_position_divergence)),
            oracle_gain_fraction=1.0,
        ),
        deep_label=_gain_capture(flat, oracle, label_allocation),
        predictor=_gain_capture(flat, oracle, predicted_allocation),
        predictor_capture_interval=interval,
        predictor_beats_flat_at_95_percent=interval.lower > 0.0,
        random_control=_random_control(
            allocation_data,
            flat,
            oracle,
            arguments.random_orderings,
            arguments.seed + 2,
        ),
    )


def run_probe(arguments: Arguments) -> ProbeReport:
    position_report = _read_position_report(arguments.per_position)
    model_hash = file_sha256(arguments.model)
    if model_hash != position_report.model_sha256:
        raise ValueError(
            f'Frozen model hash {model_hash} does not match per-position dataset model {position_report.model_sha256}.'
        )
    features = extract_frozen_features(
        arguments.model,
        position_report.records,
        arguments.device,
        arguments.feature_batch_size,
    )
    return build_report(arguments, position_report, features)


def parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser(
        description='Fit the adaptive-budget label from frozen trunk features and score equal-compute gain capture.'
    )
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--per-position', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--baseline-visits', type=int, default=600)
    parser.add_argument('--depth-visits', type=int, default=5_000)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--feature-batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--bootstrap-samples', type=int, default=2_000)
    parser.add_argument('--random-orderings', type=int, default=100)
    parser.add_argument('--seed', type=int, default=20260827)
    parsed = parser.parse_args()
    positive_integers = (
        ('baseline-visits', parsed.baseline_visits),
        ('depth-visits', parsed.depth_visits),
        ('folds', parsed.folds),
        ('epochs', parsed.epochs),
        ('batch-size', parsed.batch_size),
        ('feature-batch-size', parsed.feature_batch_size),
        ('bootstrap-samples', parsed.bootstrap_samples),
        ('random-orderings', parsed.random_orderings),
    )
    for name, value in positive_integers:
        if value <= 0:
            parser.error(f'--{name} must be positive')
    if parsed.folds < 2:
        parser.error('--folds must be at least two')
    if parsed.depth_visits <= parsed.baseline_visits:
        parser.error('--depth-visits must exceed --baseline-visits')
    if parsed.learning_rate <= 0.0:
        parser.error('--learning-rate must be positive')
    if parsed.weight_decay < 0.0:
        parser.error('--weight-decay must be nonnegative')
    return Arguments(
        model=parsed.model,
        per_position=parsed.per_position,
        output=parsed.output,
        device=torch.device(parsed.device),
        baseline_visits=parsed.baseline_visits,
        depth_visits=parsed.depth_visits,
        folds=parsed.folds,
        epochs=parsed.epochs,
        batch_size=parsed.batch_size,
        feature_batch_size=parsed.feature_batch_size,
        learning_rate=parsed.learning_rate,
        weight_decay=parsed.weight_decay,
        bootstrap_samples=parsed.bootstrap_samples,
        random_orderings=parsed.random_orderings,
        seed=parsed.seed,
    )


def main() -> None:
    arguments = parse_arguments()
    report = run_probe(arguments)
    write_text_atomically(arguments.output, report.model_dump_json(indent=2) + '\n')
    print(arguments.output)


if __name__ == '__main__':
    main()
