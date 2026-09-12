from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

import torch
from pydantic import Field
from src.util.frozen_model import FrozenModel
from torch import Tensor


@dataclass(frozen=True)
class ModelOutputs:
    policy_logits: Tensor
    wdl_probabilities: Tensor


class TimingDistribution(FrozenModel):
    repetitions: int = Field(gt=0)
    iterations_per_repetition: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    repetition_seconds: tuple[float, ...] = Field(min_length=1)
    minimum_batch_milliseconds: float = Field(gt=0.0)
    median_batch_milliseconds: float = Field(gt=0.0)
    p95_batch_milliseconds: float = Field(gt=0.0)
    maximum_batch_milliseconds: float = Field(gt=0.0)
    median_positions_per_second: float = Field(gt=0.0)


class FidelityLimits(FrozenModel):
    minimum_policy_top1_agreement: float = Field(ge=0.0, le=1.0)
    maximum_mean_policy_kl_divergence: float = Field(ge=0.0)
    maximum_wdl_mean_absolute_error: float = Field(ge=0.0)
    maximum_expected_value_mean_absolute_error: float = Field(ge=0.0)


class FidelityMetrics(FrozenModel):
    positions: int = Field(gt=0)
    policy_top1_agreement: float = Field(ge=0.0, le=1.0)
    mean_policy_kl_divergence: float = Field(ge=0.0)
    maximum_policy_kl_divergence: float = Field(ge=0.0)
    wdl_mean_absolute_error: float = Field(ge=0.0)
    wdl_maximum_absolute_error: float = Field(ge=0.0)
    expected_value_mean_absolute_error: float = Field(ge=0.0)
    expected_value_maximum_absolute_error: float = Field(ge=0.0)


def _percentile(values: tuple[float, ...], fraction: float) -> float:
    if not values:
        raise ValueError('A percentile requires at least one value.')
    if not 0.0 <= fraction <= 1.0:
        raise ValueError('A percentile fraction must be between zero and one.')
    ordered = sorted(values)
    offset = (len(ordered) - 1) * fraction
    lower_index = math.floor(offset)
    upper_index = math.ceil(offset)
    if lower_index == upper_index:
        return ordered[lower_index]
    upper_weight = offset - lower_index
    return ordered[lower_index] * (1.0 - upper_weight) + ordered[upper_index] * upper_weight


def summarize_timings(
    repetition_seconds: tuple[float, ...],
    iterations_per_repetition: int,
    batch_size: int,
) -> TimingDistribution:
    if not repetition_seconds or any(seconds <= 0.0 for seconds in repetition_seconds):
        raise ValueError('Timing repetitions must contain positive durations.')
    if iterations_per_repetition <= 0 or batch_size <= 0:
        raise ValueError('Timing iteration and batch counts must be positive.')
    batch_seconds = tuple(seconds / iterations_per_repetition for seconds in repetition_seconds)
    median_batch_seconds = statistics.median(batch_seconds)
    return TimingDistribution(
        repetitions=len(repetition_seconds),
        iterations_per_repetition=iterations_per_repetition,
        batch_size=batch_size,
        repetition_seconds=repetition_seconds,
        minimum_batch_milliseconds=min(batch_seconds) * 1000.0,
        median_batch_milliseconds=median_batch_seconds * 1000.0,
        p95_batch_milliseconds=_percentile(batch_seconds, 0.95) * 1000.0,
        maximum_batch_milliseconds=max(batch_seconds) * 1000.0,
        median_positions_per_second=batch_size / median_batch_seconds,
    )


def _legal_log_probabilities(policy_logits: Tensor, legal_action_mask: Tensor) -> Tensor:
    if policy_logits.ndim != 2 or legal_action_mask.shape != policy_logits.shape:
        raise ValueError('Policy logits and legal-action mask must have the same two-dimensional shape.')
    if legal_action_mask.dtype != torch.bool:
        raise ValueError('The legal-action mask must be boolean.')
    if not torch.all(legal_action_mask.any(dim=1)):
        raise ValueError('Every benchmark position must have at least one legal action.')
    masked_logits = policy_logits.to(torch.float64).masked_fill(~legal_action_mask, float('-inf'))
    return torch.log_softmax(masked_logits, dim=1)


def _expected_values(wdl_probabilities: Tensor) -> Tensor:
    if wdl_probabilities.ndim != 2 or wdl_probabilities.shape[1] != 3:
        raise ValueError('WDL outputs must have shape [positions, 3].')
    return wdl_probabilities.to(torch.float64)[:, 0] - wdl_probabilities.to(torch.float64)[:, 2]


def measure_fidelity(
    reference: ModelOutputs,
    candidate: ModelOutputs,
    legal_action_mask: Tensor,
) -> FidelityMetrics:
    if reference.policy_logits.shape != candidate.policy_logits.shape:
        raise ValueError('Reference and candidate policy outputs have different shapes.')
    if reference.wdl_probabilities.shape != candidate.wdl_probabilities.shape:
        raise ValueError('Reference and candidate WDL outputs have different shapes.')
    if reference.policy_logits.shape[0] != reference.wdl_probabilities.shape[0]:
        raise ValueError('Policy and WDL outputs contain different position counts.')
    tensors = (
        reference.policy_logits,
        reference.wdl_probabilities,
        candidate.policy_logits,
        candidate.wdl_probabilities,
    )
    if any(not torch.isfinite(tensor).all() for tensor in tensors):
        raise ValueError('Inference outputs must be finite.')

    reference_log_probabilities = _legal_log_probabilities(reference.policy_logits, legal_action_mask)
    candidate_log_probabilities = _legal_log_probabilities(candidate.policy_logits, legal_action_mask)
    reference_probabilities = reference_log_probabilities.exp()
    divergence_terms = reference_probabilities * (reference_log_probabilities - candidate_log_probabilities)
    policy_kl = (
        torch.where(reference_probabilities > 0.0, divergence_terms, torch.zeros_like(divergence_terms))
        .sum(dim=1)
        .clamp_min(0.0)
    )
    policy_top1_agreement = (reference_log_probabilities.argmax(dim=1) == candidate_log_probabilities.argmax(dim=1)).to(
        torch.float64
    )
    wdl_error = (reference.wdl_probabilities.to(torch.float64) - candidate.wdl_probabilities.to(torch.float64)).abs()
    expected_value_error = (
        _expected_values(reference.wdl_probabilities) - _expected_values(candidate.wdl_probabilities)
    ).abs()
    return FidelityMetrics(
        positions=reference.policy_logits.shape[0],
        policy_top1_agreement=float(policy_top1_agreement.mean()),
        mean_policy_kl_divergence=float(policy_kl.mean()),
        maximum_policy_kl_divergence=float(policy_kl.max()),
        wdl_mean_absolute_error=float(wdl_error.mean()),
        wdl_maximum_absolute_error=float(wdl_error.max()),
        expected_value_mean_absolute_error=float(expected_value_error.mean()),
        expected_value_maximum_absolute_error=float(expected_value_error.max()),
    )


def fidelity_failures(metrics: FidelityMetrics, limits: FidelityLimits) -> tuple[str, ...]:
    failures: list[str] = []
    if metrics.policy_top1_agreement < limits.minimum_policy_top1_agreement:
        failures.append(
            f'policy top-1 agreement {metrics.policy_top1_agreement:.6g} is below '
            f'{limits.minimum_policy_top1_agreement:.6g}'
        )
    if metrics.mean_policy_kl_divergence > limits.maximum_mean_policy_kl_divergence:
        failures.append(
            f'mean policy KL {metrics.mean_policy_kl_divergence:.6g} exceeds '
            f'{limits.maximum_mean_policy_kl_divergence:.6g}'
        )
    if metrics.wdl_mean_absolute_error > limits.maximum_wdl_mean_absolute_error:
        failures.append(
            f'WDL mean absolute error {metrics.wdl_mean_absolute_error:.6g} exceeds '
            f'{limits.maximum_wdl_mean_absolute_error:.6g}'
        )
    if metrics.expected_value_mean_absolute_error > limits.maximum_expected_value_mean_absolute_error:
        failures.append(
            f'expected-value mean absolute error {metrics.expected_value_mean_absolute_error:.6g} exceeds '
            f'{limits.maximum_expected_value_mean_absolute_error:.6g}'
        )
    return tuple(failures)


def validate_fidelity(backend_name: str, metrics: FidelityMetrics, limits: FidelityLimits) -> None:
    failures = fidelity_failures(metrics, limits)
    if failures:
        raise ValueError(f'{backend_name} failed fidelity limits: {"; ".join(failures)}.')
