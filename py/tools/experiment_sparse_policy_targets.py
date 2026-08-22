from __future__ import annotations

import argparse
import copy
import platform
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import torch
from src.training.objective import mask_policy_logits
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel
from torch import nn
from torch.nn import functional

ACTION_SIZE = 4_864
MAXIMUM_LEGAL_ACTIONS = 218
MAXIMUM_POLICY_ENTRIES = 60
NEXT_POLICY_WEIGHT = 0.7


class Timing(FrozenModel):
    median_seconds: float
    median_iterations_per_second: float
    trial_seconds: tuple[float, ...]


class UnavailableMetric(FrozenModel):
    available: Literal[False] = False
    reason: str


class EquivalenceResult(FrozenModel):
    dtype: str
    dense_loss: float
    sparse_loss: float
    loss_absolute_error: float
    logits_gradient_max_absolute_error: float
    parameter_gradient_max_absolute_error: float
    tolerance: float
    passed: bool


class SparsePolicyExperimentReport(FrozenModel):
    evidence_scope: str
    conclusion: str
    python_version: str
    torch_version: str
    cpu: str
    action_size: int
    maximum_legal_actions: int
    maximum_policy_entries: int
    batch_size: int
    eligible_next_policy_rows: int
    iterations_per_trial: int
    repeats: int
    float64_equivalence: EquivalenceResult
    float32_equivalence: EquivalenceResult
    dense_cpu_target_construction: Timing
    sparse_cpu_target_construction: Timing
    dense_loss_forward_backward: Timing
    sparse_loss_forward_backward: Timing
    dense_total_iteration: Timing
    sparse_total_iteration: Timing
    dense_host_target_bytes: int
    sparse_host_target_bytes: int
    host_target_byte_reduction_ratio: float
    dense_h2d_target_bytes: int
    sparse_h2d_target_bytes: int
    h2d_target_byte_reduction_ratio: float
    gpu_loss_time: UnavailableMetric
    peak_device_memory: UnavailableMetric
    end_to_end_cuda_throughput: UnavailableMetric


@dataclass(frozen=True)
class RawPolicyBatch:
    primary_action_ids: npt.NDArray[np.uint16]
    primary_visit_counts: npt.NDArray[np.uint32]
    primary_entry_count: npt.NDArray[np.uint16]
    primary_legal_action_ids: npt.NDArray[np.uint16]
    primary_legal_count: npt.NDArray[np.uint16]
    next_action_ids: npt.NDArray[np.uint16]
    next_visit_counts: npt.NDArray[np.uint32]
    next_entry_count: npt.NDArray[np.uint16]
    next_legal_action_ids: npt.NDArray[np.uint16]
    next_legal_count: npt.NDArray[np.uint16]
    next_eligible: npt.NDArray[np.bool_]
    sample_weights: npt.NDArray[np.float32]


@dataclass(frozen=True)
class DensePolicyBatch:
    primary_probabilities: npt.NDArray[np.float32]
    primary_legal_action_ids: npt.NDArray[np.int64]
    next_probabilities: npt.NDArray[np.float32]
    next_legal_action_ids: npt.NDArray[np.int64]
    next_eligible: npt.NDArray[np.bool_]
    sample_weights: npt.NDArray[np.float32]


@dataclass(frozen=True)
class SparsePolicyTarget:
    action_ids: npt.NDArray[np.int64]
    probabilities: npt.NDArray[np.float32]
    valid: npt.NDArray[np.bool_]
    legal_action_ids: npt.NDArray[np.int64]
    legal_valid: npt.NDArray[np.bool_]


@dataclass(frozen=True)
class SparsePolicyBatch:
    primary: SparsePolicyTarget
    next_policy: SparsePolicyTarget
    next_eligible: npt.NDArray[np.bool_]
    sample_weights: npt.NDArray[np.float32]


@dataclass(frozen=True)
class TorchDensePolicyBatch:
    primary_probabilities: torch.Tensor
    primary_legal_action_ids: torch.Tensor
    next_probabilities: torch.Tensor
    next_legal_action_ids: torch.Tensor
    next_eligible: torch.Tensor
    sample_weights: torch.Tensor


@dataclass(frozen=True)
class TorchSparsePolicyTarget:
    action_ids: torch.Tensor
    probabilities: torch.Tensor
    valid: torch.Tensor
    legal_action_ids: torch.Tensor
    legal_valid: torch.Tensor


@dataclass(frozen=True)
class TorchSparsePolicyBatch:
    primary: TorchSparsePolicyTarget
    next_policy: TorchSparsePolicyTarget
    next_eligible: torch.Tensor
    sample_weights: torch.Tensor


class _TwoPolicyHeads(nn.Module):
    def __init__(self, feature_count: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.primary = nn.Linear(feature_count, ACTION_SIZE, dtype=dtype)
        self.next_policy = nn.Linear(feature_count, ACTION_SIZE, dtype=dtype)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.primary(features), self.next_policy(features)


def create_raw_policy_batch(batch_size: int, seed: int) -> RawPolicyBatch:
    generator = np.random.default_rng(seed)
    primary_actions = np.zeros((batch_size, MAXIMUM_POLICY_ENTRIES), dtype=np.uint16)
    primary_visits = np.zeros((batch_size, MAXIMUM_POLICY_ENTRIES), dtype=np.uint32)
    primary_counts = generator.integers(1, MAXIMUM_POLICY_ENTRIES + 1, size=batch_size, dtype=np.uint16)
    primary_legal = np.empty((batch_size, MAXIMUM_LEGAL_ACTIONS), dtype=np.uint16)
    next_actions = np.zeros_like(primary_actions)
    next_visits = np.zeros_like(primary_visits)
    next_counts = generator.integers(1, MAXIMUM_POLICY_ENTRIES + 1, size=batch_size, dtype=np.uint16)
    next_legal = np.zeros_like(primary_legal)
    next_eligible = np.arange(batch_size) % 3 != 0
    next_legal_counts = np.where(next_eligible, MAXIMUM_LEGAL_ACTIONS, 0).astype(np.uint16)
    for row in range(batch_size):
        legal = generator.choice(ACTION_SIZE, size=MAXIMUM_LEGAL_ACTIONS, replace=False).astype(np.uint16)
        primary_legal[row] = legal
        primary_count = int(primary_counts[row])
        primary_actions[row, :primary_count] = legal[:primary_count]
        primary_visits[row, :primary_count] = generator.integers(1, 1_001, size=primary_count, dtype=np.uint32)
        if next_eligible[row]:
            next_row_legal = generator.choice(
                ACTION_SIZE,
                size=MAXIMUM_LEGAL_ACTIONS,
                replace=False,
            ).astype(np.uint16)
            next_legal[row] = next_row_legal
            next_count = int(next_counts[row])
            next_actions[row, :next_count] = next_row_legal[:next_count]
            next_visits[row, :next_count] = generator.integers(1, 1_001, size=next_count, dtype=np.uint32)
        else:
            next_counts[row] = 0
    return RawPolicyBatch(
        primary_action_ids=primary_actions,
        primary_visit_counts=primary_visits,
        primary_entry_count=primary_counts,
        primary_legal_action_ids=primary_legal,
        primary_legal_count=np.full(batch_size, MAXIMUM_LEGAL_ACTIONS, dtype=np.uint16),
        next_action_ids=next_actions,
        next_visit_counts=next_visits,
        next_entry_count=next_counts,
        next_legal_action_ids=next_legal,
        next_legal_count=next_legal_counts,
        next_eligible=next_eligible,
        sample_weights=generator.uniform(0.25, 2.0, size=batch_size).astype(np.float32),
    )


def _probabilities(
    visit_counts: npt.NDArray[np.uint32],
    entry_count: npt.NDArray[np.uint16],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    valid = np.arange(visit_counts.shape[1])[np.newaxis, :] < entry_count[:, np.newaxis]
    visits = np.where(valid, visit_counts, 0).astype(np.float32)
    totals = visits.sum(axis=1, keepdims=True)
    probabilities = np.divide(visits, totals, out=np.zeros_like(visits), where=totals > 0)
    return probabilities, valid


def _padded_legal(
    legal_action_ids: npt.NDArray[np.uint16],
    legal_count: npt.NDArray[np.uint16],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.bool_]]:
    valid = np.arange(legal_action_ids.shape[1])[np.newaxis, :] < legal_count[:, np.newaxis]
    return np.where(valid, legal_action_ids, 0).astype(np.int64), valid


def build_dense_policy_batch(raw: RawPolicyBatch) -> DensePolicyBatch:
    primary_probabilities, primary_valid = _probabilities(raw.primary_visit_counts, raw.primary_entry_count)
    next_probabilities, next_valid = _probabilities(raw.next_visit_counts, raw.next_entry_count)
    row_count = len(raw.primary_entry_count)
    primary_dense = np.zeros((row_count, ACTION_SIZE), dtype=np.float32)
    primary_rows, primary_entries = np.nonzero(primary_valid)
    primary_dense[primary_rows, raw.primary_action_ids[primary_rows, primary_entries]] = primary_probabilities[
        primary_rows, primary_entries
    ]
    next_dense = np.zeros_like(primary_dense)
    next_rows, next_entries = np.nonzero(next_valid)
    next_dense[next_rows, raw.next_action_ids[next_rows, next_entries]] = next_probabilities[next_rows, next_entries]
    primary_legal, primary_legal_valid = _padded_legal(raw.primary_legal_action_ids, raw.primary_legal_count)
    next_legal, next_legal_valid = _padded_legal(raw.next_legal_action_ids, raw.next_legal_count)
    return DensePolicyBatch(
        primary_probabilities=primary_dense,
        primary_legal_action_ids=np.where(primary_legal_valid, primary_legal, -1),
        next_probabilities=next_dense,
        next_legal_action_ids=np.where(next_legal_valid, next_legal, -1),
        next_eligible=raw.next_eligible.copy(),
        sample_weights=raw.sample_weights.copy(),
    )


def build_sparse_policy_batch(raw: RawPolicyBatch) -> SparsePolicyBatch:
    primary_probabilities, primary_valid = _probabilities(raw.primary_visit_counts, raw.primary_entry_count)
    next_probabilities, next_valid = _probabilities(raw.next_visit_counts, raw.next_entry_count)
    primary_legal, primary_legal_valid = _padded_legal(raw.primary_legal_action_ids, raw.primary_legal_count)
    next_legal, next_legal_valid = _padded_legal(raw.next_legal_action_ids, raw.next_legal_count)
    return SparsePolicyBatch(
        primary=SparsePolicyTarget(
            action_ids=raw.primary_action_ids.astype(np.int64),
            probabilities=primary_probabilities,
            valid=primary_valid,
            legal_action_ids=primary_legal,
            legal_valid=primary_legal_valid,
        ),
        next_policy=SparsePolicyTarget(
            action_ids=raw.next_action_ids.astype(np.int64),
            probabilities=next_probabilities,
            valid=next_valid,
            legal_action_ids=next_legal,
            legal_valid=next_legal_valid,
        ),
        next_eligible=raw.next_eligible.copy(),
        sample_weights=raw.sample_weights.copy(),
    )


def _torch_dense(batch: DensePolicyBatch, dtype: torch.dtype) -> TorchDensePolicyBatch:
    return TorchDensePolicyBatch(
        primary_probabilities=torch.from_numpy(batch.primary_probabilities).to(dtype=dtype),
        primary_legal_action_ids=torch.from_numpy(batch.primary_legal_action_ids),
        next_probabilities=torch.from_numpy(batch.next_probabilities).to(dtype=dtype),
        next_legal_action_ids=torch.from_numpy(batch.next_legal_action_ids),
        next_eligible=torch.from_numpy(batch.next_eligible),
        sample_weights=torch.from_numpy(batch.sample_weights).to(dtype=dtype),
    )


def _torch_sparse_target(target: SparsePolicyTarget, dtype: torch.dtype) -> TorchSparsePolicyTarget:
    return TorchSparsePolicyTarget(
        action_ids=torch.from_numpy(target.action_ids),
        probabilities=torch.from_numpy(target.probabilities).to(dtype=dtype),
        valid=torch.from_numpy(target.valid),
        legal_action_ids=torch.from_numpy(target.legal_action_ids),
        legal_valid=torch.from_numpy(target.legal_valid),
    )


def _torch_sparse(batch: SparsePolicyBatch, dtype: torch.dtype) -> TorchSparsePolicyBatch:
    return TorchSparsePolicyBatch(
        primary=_torch_sparse_target(batch.primary, dtype),
        next_policy=_torch_sparse_target(batch.next_policy, dtype),
        next_eligible=torch.from_numpy(batch.next_eligible),
        sample_weights=torch.from_numpy(batch.sample_weights).to(dtype=dtype),
    )


def _dense_loss(
    primary_logits: torch.Tensor,
    next_logits: torch.Tensor,
    batch: TorchDensePolicyBatch,
) -> torch.Tensor:
    sample_weights = batch.sample_weights / batch.sample_weights.mean()
    primary_rows = functional.cross_entropy(
        mask_policy_logits(primary_logits, batch.primary_legal_action_ids),
        batch.primary_probabilities,
        reduction='none',
    )
    eligible = batch.next_eligible.to(dtype=torch.bool)
    next_rows = torch.zeros(len(eligible), dtype=next_logits.dtype)
    next_rows[eligible] = functional.cross_entropy(
        mask_policy_logits(next_logits[eligible], batch.next_legal_action_ids[eligible]),
        batch.next_probabilities[eligible],
        reduction='none',
    )
    eligible_weights = eligible.to(dtype=sample_weights.dtype) * sample_weights
    next_loss = (next_rows * eligible_weights).sum() / eligible_weights.sum().clamp_min(1.0)
    return (primary_rows * sample_weights).mean() + NEXT_POLICY_WEIGHT * next_loss


def _sparse_rows(logits: torch.Tensor, target: TorchSparsePolicyTarget) -> torch.Tensor:
    legal_logits = logits.gather(1, target.legal_action_ids)
    legal_logits = legal_logits.masked_fill(~target.legal_valid, torch.finfo(logits.dtype).min)
    log_normalizer = torch.logsumexp(legal_logits, dim=1)
    target_logits = logits.gather(1, target.action_ids)
    return -(
        target.probabilities * target.valid.to(dtype=logits.dtype) * (target_logits - log_normalizer[:, np.newaxis])
    ).sum(dim=1)


def _sparse_loss(
    primary_logits: torch.Tensor,
    next_logits: torch.Tensor,
    batch: TorchSparsePolicyBatch,
) -> torch.Tensor:
    sample_weights = batch.sample_weights / batch.sample_weights.mean()
    primary_rows = _sparse_rows(primary_logits, batch.primary)
    eligible = batch.next_eligible.to(dtype=torch.bool)
    next_target = TorchSparsePolicyTarget(
        action_ids=batch.next_policy.action_ids[eligible],
        probabilities=batch.next_policy.probabilities[eligible],
        valid=batch.next_policy.valid[eligible],
        legal_action_ids=batch.next_policy.legal_action_ids[eligible],
        legal_valid=batch.next_policy.legal_valid[eligible],
    )
    next_rows = torch.zeros(len(eligible), dtype=next_logits.dtype)
    next_rows[eligible] = _sparse_rows(next_logits[eligible], next_target)
    eligible_weights = eligible.to(dtype=sample_weights.dtype) * sample_weights
    next_loss = (next_rows * eligible_weights).sum() / eligible_weights.sum().clamp_min(1.0)
    return (primary_rows * sample_weights).mean() + NEXT_POLICY_WEIGHT * next_loss


def _gradient_equivalence(raw: RawPolicyBatch, dtype: torch.dtype, tolerance: float) -> EquivalenceResult:
    dense = _torch_dense(build_dense_policy_batch(raw), dtype)
    sparse = _torch_sparse(build_sparse_policy_batch(raw), dtype)
    generator = torch.Generator().manual_seed(91)
    primary_values = torch.randn((len(raw.primary_entry_count), ACTION_SIZE), generator=generator, dtype=dtype)
    next_values = torch.randn((len(raw.primary_entry_count), ACTION_SIZE), generator=generator, dtype=dtype)
    dense_primary = primary_values.clone().requires_grad_(True)
    dense_next = next_values.clone().requires_grad_(True)
    sparse_primary = primary_values.clone().requires_grad_(True)
    sparse_next = next_values.clone().requires_grad_(True)
    dense_loss = _dense_loss(dense_primary, dense_next, dense)
    sparse_loss = _sparse_loss(sparse_primary, sparse_next, sparse)
    dense_loss.backward()
    sparse_loss.backward()
    logits_error = max(
        float(torch.max(torch.abs(dense_primary.grad - sparse_primary.grad))),
        float(torch.max(torch.abs(dense_next.grad - sparse_next.grad))),
    )

    feature_count = 11
    features = torch.randn((len(raw.primary_entry_count), feature_count), generator=generator, dtype=dtype)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(901)
        dense_model = _TwoPolicyHeads(feature_count, dtype)
    sparse_model = copy.deepcopy(dense_model)
    dense_output = dense_model(features)
    sparse_output = sparse_model(features)
    _dense_loss(*dense_output, dense).backward()
    _sparse_loss(*sparse_output, sparse).backward()
    parameter_error = max(
        float(torch.max(torch.abs(dense_parameter.grad - sparse_parameter.grad)))
        for dense_parameter, sparse_parameter in zip(
            dense_model.parameters(),
            sparse_model.parameters(),
            strict=True,
        )
    )
    loss_error = float(torch.abs(dense_loss - sparse_loss).detach())
    return EquivalenceResult(
        dtype=str(dtype),
        dense_loss=float(dense_loss.detach()),
        sparse_loss=float(sparse_loss.detach()),
        loss_absolute_error=loss_error,
        logits_gradient_max_absolute_error=logits_error,
        parameter_gradient_max_absolute_error=parameter_error,
        tolerance=tolerance,
        passed=max(loss_error, logits_error, parameter_error) <= tolerance,
    )


def _array_bytes(values: tuple[npt.NDArray[np.generic], ...]) -> int:
    return sum(value.nbytes for value in values)


def _dense_bytes(batch: DensePolicyBatch) -> int:
    return _array_bytes(
        (
            batch.primary_probabilities,
            batch.primary_legal_action_ids,
            batch.next_probabilities,
            batch.next_legal_action_ids,
            batch.next_eligible,
            batch.sample_weights,
        )
    )


def _sparse_bytes(batch: SparsePolicyBatch) -> int:
    return _array_bytes(
        (
            batch.primary.action_ids,
            batch.primary.probabilities,
            batch.primary.valid,
            batch.primary.legal_action_ids,
            batch.primary.legal_valid,
            batch.next_policy.action_ids,
            batch.next_policy.probabilities,
            batch.next_policy.valid,
            batch.next_policy.legal_action_ids,
            batch.next_policy.legal_valid,
            batch.next_eligible,
            batch.sample_weights,
        )
    )


def _elapsed(operation: Callable[[], object]) -> float:
    started = time.perf_counter()
    operation()
    return time.perf_counter() - started


def _timing(trials: list[float], iterations: int) -> Timing:
    median = statistics.median(trials)
    return Timing(
        median_seconds=median,
        median_iterations_per_second=iterations / median,
        trial_seconds=tuple(trials),
    )


def _repeat(operation: Callable[[], object], iterations: int) -> None:
    for _ in range(iterations):
        operation()


def _dense_backward(logits: tuple[torch.Tensor, torch.Tensor], batch: TorchDensePolicyBatch) -> float:
    primary = logits[0].clone().requires_grad_(True)
    next_policy = logits[1].clone().requires_grad_(True)
    loss = _dense_loss(primary, next_policy, batch)
    loss.backward()
    return float(loss.detach())


def _sparse_backward(logits: tuple[torch.Tensor, torch.Tensor], batch: TorchSparsePolicyBatch) -> float:
    primary = logits[0].clone().requires_grad_(True)
    next_policy = logits[1].clone().requires_grad_(True)
    loss = _sparse_loss(primary, next_policy, batch)
    loss.backward()
    return float(loss.detach())


def _dense_total(raw: RawPolicyBatch, logits: tuple[torch.Tensor, torch.Tensor]) -> float:
    return _dense_backward(logits, _torch_dense(build_dense_policy_batch(raw), torch.float32))


def _sparse_total(raw: RawPolicyBatch, logits: tuple[torch.Tensor, torch.Tensor]) -> float:
    return _sparse_backward(logits, _torch_sparse(build_sparse_policy_batch(raw), torch.float32))


def run_experiment(
    output: Path,
    batch_size: int = 128,
    iterations: int = 5,
    repeats: int = 5,
    seed: int = 20_260_822,
) -> SparsePolicyExperimentReport:
    if batch_size <= 1 or iterations <= 0 or repeats <= 0:
        raise ValueError('Batch size must exceed one and iterations and repeats must be positive.')
    raw = create_raw_policy_batch(batch_size, seed)
    dense = build_dense_policy_batch(raw)
    sparse = build_sparse_policy_batch(raw)
    float64 = _gradient_equivalence(raw, torch.float64, 1e-12)
    float32 = _gradient_equivalence(raw, torch.float32, 2e-6)
    if not float64.passed or not float32.passed:
        raise RuntimeError('Sparse policy loss failed dense forward/backward equivalence.')
    dense_torch = _torch_dense(dense, torch.float32)
    sparse_torch = _torch_sparse(sparse, torch.float32)
    generator = torch.Generator().manual_seed(seed)
    logits = (
        torch.randn((batch_size, ACTION_SIZE), generator=generator),
        torch.randn((batch_size, ACTION_SIZE), generator=generator),
    )
    trials: dict[str, list[float]] = {
        name: [] for name in ('dense_build', 'sparse_build', 'dense_loss', 'sparse_loss', 'dense_total', 'sparse_total')
    }
    _dense_backward(logits, dense_torch)
    _sparse_backward(logits, sparse_torch)
    for trial in range(repeats):
        operations: tuple[tuple[str, Callable[[], object]], ...] = (
            ('dense_build', lambda: _repeat(lambda: build_dense_policy_batch(raw), iterations)),
            ('sparse_build', lambda: _repeat(lambda: build_sparse_policy_batch(raw), iterations)),
            ('dense_loss', lambda: _repeat(lambda: _dense_backward(logits, dense_torch), iterations)),
            ('sparse_loss', lambda: _repeat(lambda: _sparse_backward(logits, sparse_torch), iterations)),
            ('dense_total', lambda: _repeat(lambda: _dense_total(raw, logits), iterations)),
            ('sparse_total', lambda: _repeat(lambda: _sparse_total(raw, logits), iterations)),
        )
        if trial % 2:
            operations = tuple(reversed(operations))
        for name, operation in operations:
            trials[name].append(_elapsed(operation))
    dense_bytes = _dense_bytes(dense)
    sparse_bytes = _sparse_bytes(sparse)
    unavailable = UnavailableMetric(reason='CUDA is unavailable in this controlled CPU-only environment.')
    report = SparsePolicyExperimentReport(
        evidence_scope=(
            'Synthetic CPU-only isolated primary/next-policy experiment; not CUDA, DDP, live-run, '
            'or end-to-end acceptance evidence.'
        ),
        conclusion=(
            'Keep the dense path authoritative: mathematical equivalence and CPU/byte measurements alone do not '
            'demonstrate an end-to-end CUDA gain.'
        ),
        python_version=platform.python_version(),
        torch_version=str(torch.__version__),
        cpu=platform.processor() or platform.machine(),
        action_size=ACTION_SIZE,
        maximum_legal_actions=MAXIMUM_LEGAL_ACTIONS,
        maximum_policy_entries=MAXIMUM_POLICY_ENTRIES,
        batch_size=batch_size,
        eligible_next_policy_rows=int(raw.next_eligible.sum()),
        iterations_per_trial=iterations,
        repeats=repeats,
        float64_equivalence=float64,
        float32_equivalence=float32,
        dense_cpu_target_construction=_timing(trials['dense_build'], iterations),
        sparse_cpu_target_construction=_timing(trials['sparse_build'], iterations),
        dense_loss_forward_backward=_timing(trials['dense_loss'], iterations),
        sparse_loss_forward_backward=_timing(trials['sparse_loss'], iterations),
        dense_total_iteration=_timing(trials['dense_total'], iterations),
        sparse_total_iteration=_timing(trials['sparse_total'], iterations),
        dense_host_target_bytes=dense_bytes,
        sparse_host_target_bytes=sparse_bytes,
        host_target_byte_reduction_ratio=1.0 - sparse_bytes / dense_bytes,
        dense_h2d_target_bytes=dense_bytes,
        sparse_h2d_target_bytes=sparse_bytes,
        h2d_target_byte_reduction_ratio=1.0 - sparse_bytes / dense_bytes,
        gpu_loss_time=unavailable,
        peak_device_memory=unavailable,
        end_to_end_cuda_throughput=unavailable,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomically(output, report.model_dump_json(indent=2) + '\n')
    return report


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError('value must be positive')
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description='Run the bounded synthetic sparse-policy target experiment.')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--batch-size', type=_positive_int, default=128)
    parser.add_argument('--iterations', type=_positive_int, default=5)
    parser.add_argument('--repeats', type=_positive_int, default=5)
    parser.add_argument('--seed', type=int, default=20_260_822)
    arguments = parser.parse_args()
    run_experiment(arguments.output, arguments.batch_size, arguments.iterations, arguments.repeats, arguments.seed)


if __name__ == '__main__':
    main()
