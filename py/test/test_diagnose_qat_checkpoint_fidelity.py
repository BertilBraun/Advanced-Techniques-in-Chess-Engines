from __future__ import annotations

import torch
from tools.diagnose_qat_checkpoint_fidelity import (
    FailedFidelityComparison,
    FiniteOutputSummary,
    NonFiniteOutputSummary,
    SuccessfulFidelityComparison,
    _compare_outputs,
    _output_summary,
)
from tools.tensorrt_benchmark_metrics import ModelOutputs


def _legal_mask() -> torch.Tensor:
    return torch.tensor([[True, True, False], [True, False, True]])


def test_output_summary_reports_finite_policy_shape() -> None:
    outputs = ModelOutputs(
        policy_logits=torch.tensor([[1.0, 0.0, -2.0], [0.5, -1.0, 0.25]]),
        wdl_probabilities=torch.tensor([[0.5, 0.25, 0.25], [0.2, 0.3, 0.5]]),
    )

    summary = _output_summary(outputs, _legal_mask())

    assert isinstance(summary, FiniteOutputSummary)
    assert summary.mean_legal_top1_mass > 0.5
    assert summary.mean_legal_top3_mass == 1.0


def test_output_summary_and_comparison_preserve_nonfinite_failure() -> None:
    reference = ModelOutputs(
        policy_logits=torch.tensor([[1.0, 0.0, -2.0], [0.5, -1.0, 0.25]]),
        wdl_probabilities=torch.tensor([[0.5, 0.25, 0.25], [0.2, 0.3, 0.5]]),
    )
    candidate = ModelOutputs(
        policy_logits=torch.tensor([[float('inf'), 0.0, -2.0], [0.5, -1.0, 0.25]]),
        wdl_probabilities=torch.tensor([[0.5, 0.25, 0.25], [float('nan'), 0.3, 0.5]]),
    )

    summary = _output_summary(candidate, _legal_mask())
    comparison = _compare_outputs(reference, candidate, _legal_mask())

    assert isinstance(summary, NonFiniteOutputSummary)
    assert summary.policy_nonfinite_values == 1
    assert summary.wdl_nonfinite_values == 1
    assert isinstance(comparison, FailedFidelityComparison)
    assert comparison.reason == 'candidate outputs are non-finite'


def test_finite_comparison_returns_fidelity_metrics() -> None:
    outputs = ModelOutputs(
        policy_logits=torch.tensor([[1.0, 0.0, -2.0], [0.5, -1.0, 0.25]]),
        wdl_probabilities=torch.tensor([[0.5, 0.25, 0.25], [0.2, 0.3, 0.5]]),
    )

    comparison = _compare_outputs(outputs, outputs, _legal_mask())

    assert isinstance(comparison, SuccessfulFidelityComparison)
    assert comparison.metrics.policy_top1_agreement == 1.0
    assert comparison.metrics.mean_policy_kl_divergence == 0.0
