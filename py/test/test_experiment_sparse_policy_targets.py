from __future__ import annotations

from pathlib import Path

from tools.experiment_sparse_policy_targets import SparsePolicyExperimentReport, run_experiment


def test_sparse_policy_experiment_proves_equivalence_and_reports_unavailable_cuda(tmp_path: Path) -> None:
    output = tmp_path / 'sparse-policy.json'

    report = run_experiment(output, batch_size=8, iterations=1, repeats=2, seed=31)

    persisted = SparsePolicyExperimentReport.model_validate_json(output.read_text(encoding='utf-8'))
    assert persisted == report
    assert report.float64_equivalence.passed
    assert report.float32_equivalence.passed
    assert report.eligible_next_policy_rows == 5
    assert report.sparse_host_target_bytes < report.dense_host_target_bytes
    assert report.sparse_h2d_target_bytes < report.dense_h2d_target_bytes
    assert report.dense_cpu_target_construction.median_seconds > 0.0
    assert report.sparse_cpu_target_construction.median_seconds > 0.0
    assert report.dense_loss_forward_backward.median_seconds > 0.0
    assert report.sparse_loss_forward_backward.median_seconds > 0.0
    assert not report.gpu_loss_time.available
    assert not report.peak_device_memory.available
    assert not report.end_to_end_cuda_throughput.available
    assert report.conclusion.startswith('Keep the dense path authoritative')
