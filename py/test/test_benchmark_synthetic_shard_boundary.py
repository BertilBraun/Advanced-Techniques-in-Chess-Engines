from __future__ import annotations

from pathlib import Path

from tools.benchmark_synthetic_shard_boundary import SyntheticShardBoundaryReport, run_benchmark


def test_synthetic_shard_boundary_benchmark_is_bounded_wrapped_and_exact(tmp_path: Path) -> None:
    output = tmp_path / 'synthetic-shard-boundary.json'

    report = run_benchmark(
        output,
        games=8,
        rows_per_game=2,
        games_per_shard=4,
        repeats=2,
    )

    persisted = SyntheticShardBoundaryReport.model_validate_json(output.read_text(encoding='utf-8'))
    assert persisted == report
    assert report.evidence_scope.startswith('Synthetic CPU-only')
    assert '<3%' in report.evidence_scope
    assert report.exact_final_semantics
    assert report.wrapped_ring
    assert report.old_per_game_file_count == 16
    assert report.columnar_shard_file_count == 4
    assert report.file_count_reduction_factor == 4.0
    assert report.shard_count == 2
    assert report.columnar_boundary_total.median_seconds > 0.0
    assert report.columnar_sequential_append_and_flush.median_rows_per_second > 0.0
    assert report.old_aos_boundary_total.median_seconds > 0.0
