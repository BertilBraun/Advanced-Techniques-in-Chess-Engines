from __future__ import annotations

import pytest
import torch
from tools.tensorrt_benchmark_metrics import (
    FidelityLimits,
    ModelOutputs,
    fidelity_failures,
    measure_fidelity,
    summarize_timings,
    validate_fidelity,
)


def _limits() -> FidelityLimits:
    return FidelityLimits(
        minimum_policy_top1_agreement=0.98,
        maximum_mean_policy_kl_divergence=0.005,
        maximum_wdl_mean_absolute_error=0.01,
        maximum_expected_value_mean_absolute_error=0.015,
    )


def test_timing_summary_reports_distribution_and_positions_per_second() -> None:
    summary = summarize_timings((1.0, 1.2, 0.8), iterations_per_repetition=100, batch_size=320)

    assert summary.minimum_batch_milliseconds == pytest.approx(8.0)
    assert summary.median_batch_milliseconds == pytest.approx(10.0)
    assert summary.p95_batch_milliseconds == pytest.approx(11.8)
    assert summary.maximum_batch_milliseconds == pytest.approx(12.0)
    assert summary.median_positions_per_second == pytest.approx(32_000.0)


def test_fidelity_uses_only_legal_policy_actions_and_wdl_expected_value() -> None:
    reference = ModelOutputs(
        policy_logits=torch.tensor(((2.0, 100.0, 1.0), (0.0, 2.0, 1.0))),
        wdl_probabilities=torch.tensor(((0.7, 0.2, 0.1), (0.2, 0.5, 0.3))),
    )
    candidate = ModelOutputs(
        policy_logits=torch.tensor(((2.0, -100.0, 1.0), (0.0, 1.0, 2.0))),
        wdl_probabilities=torch.tensor(((0.6, 0.3, 0.1), (0.3, 0.4, 0.3))),
    )
    legal_action_mask = torch.tensor(((True, False, True), (True, True, True)))

    metrics = measure_fidelity(reference, candidate, legal_action_mask)

    assert metrics.policy_top1_agreement == 0.5
    assert metrics.mean_policy_kl_divergence > 0.0
    assert metrics.wdl_mean_absolute_error == pytest.approx(1.0 / 15.0)
    assert metrics.expected_value_mean_absolute_error == pytest.approx(0.1)
    assert metrics.wdl_maximum_absolute_error == pytest.approx(0.1)


def test_identical_outputs_pass_fidelity_limits() -> None:
    outputs = ModelOutputs(
        policy_logits=torch.tensor(((1.0, 2.0), (3.0, 2.0))),
        wdl_probabilities=torch.tensor(((0.2, 0.5, 0.3), (0.6, 0.3, 0.1))),
    )
    legal_action_mask = torch.ones_like(outputs.policy_logits, dtype=torch.bool)
    metrics = measure_fidelity(outputs, outputs, legal_action_mask)

    assert fidelity_failures(metrics, _limits()) == ()
    validate_fidelity('identical', metrics, _limits())


def test_fidelity_gate_reports_every_excessive_error() -> None:
    reference = ModelOutputs(
        policy_logits=torch.tensor(((10.0, 0.0),)),
        wdl_probabilities=torch.tensor(((1.0, 0.0, 0.0),)),
    )
    candidate = ModelOutputs(
        policy_logits=torch.tensor(((0.0, 10.0),)),
        wdl_probabilities=torch.tensor(((0.0, 0.0, 1.0),)),
    )
    legal_action_mask = torch.ones_like(reference.policy_logits, dtype=torch.bool)
    metrics = measure_fidelity(reference, candidate, legal_action_mask)

    failures = fidelity_failures(metrics, _limits())
    assert len(failures) == 4
    with pytest.raises(ValueError, match='failed fidelity limits'):
        validate_fidelity('int8', metrics, _limits())


@pytest.mark.parametrize(
    'repetition_seconds,iterations,batch_size',
    (
        ((), 1, 1),
        ((0.0,), 1, 1),
        ((1.0,), 0, 1),
        ((1.0,), 1, 0),
    ),
)
def test_timing_summary_rejects_invalid_measurements(
    repetition_seconds: tuple[float, ...], iterations: int, batch_size: int
) -> None:
    with pytest.raises(ValueError):
        summarize_timings(repetition_seconds, iterations, batch_size)
