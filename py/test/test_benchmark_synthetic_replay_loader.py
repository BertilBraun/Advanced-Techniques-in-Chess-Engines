from __future__ import annotations

from pathlib import Path

from tools.benchmark_synthetic_replay_loader import SyntheticReplayBenchmarkReport, run_benchmark


def test_synthetic_replay_benchmark_is_bounded_exact_and_writes_report(tmp_path: Path) -> None:
    output = tmp_path / 'synthetic-replay-loader.json'

    report = run_benchmark(
        output,
        maximum_rows=32,
        logical_rows=24,
        batch_size=8,
        iterations=2,
        repeats=2,
        seed=17,
    )

    persisted = SyntheticReplayBenchmarkReport.model_validate_json(output.read_text(encoding='utf-8'))
    assert persisted == report
    assert report.exact_equivalence
    assert report.wrapped_store
    assert report.duplicate_indices_per_batch
    assert report.measured_rows_per_trial == 16
    assert report.semantic_checksum
    assert report.full_object_reference.median_seconds > 0.0
    assert report.full_vectorized.median_seconds > 0.0
    assert report.full_build_speedup > 0.0
    assert 'not CUDA' in report.evidence_scope
