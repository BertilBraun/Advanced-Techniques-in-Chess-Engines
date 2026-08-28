from __future__ import annotations

from pathlib import Path

from src.search_budget.configuration import DeepLabelingConfiguration, LabelArtifactRetention
from src.search_budget.retention import (
    RECEIPT_FILE_NAME,
    CompletedLabelArtifactCleanupReceipt,
    LabelArtifactCleanupEvidence,
    LabelArtifactCleanupTarget,
    LabelArtifactKind,
    PreparedLabelArtifactCleanupReceipt,
    cleanup_completed_generation_artifacts,
    load_label_artifact_cleanup_receipt,
)


def test_diagnostic_label_artifacts_are_retained_by_default() -> None:
    assert DeepLabelingConfiguration().artifact_retention is LabelArtifactRetention.RETAIN_ALL


def _evidence() -> LabelArtifactCleanupEvidence:
    return LabelArtifactCleanupEvidence(
        final_report_path=Path('search-budget-labels/generation-00000003/final-report.json'),
        manager_state_path=Path('search-budget-labels/manager-state.json'),
        calibration_state_path=Path('search-budget-labels/calibration-state.json'),
        replay_writeback_state_path=Path('completed-games/labelled-replay-writebacks.json'),
    )


def test_completed_generation_cleanup_removes_only_bulky_artifacts(tmp_path: Path) -> None:
    job_path = tmp_path / 'search-budget-labels' / 'generation-00000003'
    prediction_path = job_path / 'prediction' / 'shard-00000'
    deep_search_path = job_path / 'deep-search' / 'shard-00000'
    prediction_path.mkdir(parents=True)
    deep_search_path.mkdir(parents=True)
    source_path = job_path / 'source.json'
    prediction_artifact = prediction_path / 'artifact-attempt-1.json'
    deep_search_artifact = deep_search_path / 'artifact-attempt-1.json'
    prediction_manifest = prediction_path / 'attempt-1.json'
    deep_search_manifest = deep_search_path / 'attempt-1.json'
    final_report = job_path / 'final-report.json'
    source_path.write_bytes(b'source')
    prediction_artifact.write_bytes(b'prediction')
    deep_search_artifact.write_bytes(b'deep-search-policy-checkpoints')
    prediction_manifest.write_text('prediction manifest', encoding='utf-8')
    deep_search_manifest.write_text('deep-search manifest', encoding='utf-8')
    final_report.write_text('final report', encoding='utf-8')

    receipt = cleanup_completed_generation_artifacts(job_path, 3, _evidence())

    assert isinstance(receipt, CompletedLabelArtifactCleanupReceipt)
    assert receipt.removed_file_count == 3
    assert receipt.removed_size_bytes == len(b'sourcepredictiondeep-search-policy-checkpoints')
    assert receipt.preserved_manifest_count == 2
    assert not source_path.exists()
    assert not prediction_artifact.exists()
    assert not deep_search_artifact.exists()
    assert prediction_manifest.read_text(encoding='utf-8') == 'prediction manifest'
    assert deep_search_manifest.read_text(encoding='utf-8') == 'deep-search manifest'
    assert final_report.read_text(encoding='utf-8') == 'final report'
    persisted = load_label_artifact_cleanup_receipt(job_path / RECEIPT_FILE_NAME)
    assert persisted == receipt


def test_completed_generation_cleanup_is_idempotent(tmp_path: Path) -> None:
    job_path = tmp_path / 'search-budget-labels' / 'generation-00000003'
    job_path.mkdir(parents=True)
    (job_path / 'source.json').write_bytes(b'source')

    first = cleanup_completed_generation_artifacts(job_path, 3, _evidence())
    second = cleanup_completed_generation_artifacts(job_path, 3, _evidence())

    assert isinstance(first, CompletedLabelArtifactCleanupReceipt)
    assert second == first


def test_prepared_cleanup_receipt_resumes_partial_cleanup(tmp_path: Path) -> None:
    job_path = tmp_path / 'search-budget-labels' / 'generation-00000003'
    job_path.mkdir(parents=True)
    source_path = job_path / 'source.json'
    source_path.write_bytes(b'source')
    prepared = PreparedLabelArtifactCleanupReceipt(
        source_generation=3,
        retention=LabelArtifactRetention.REMOVE_BULKY_AFTER_FINALIZATION,
        targets=(
            LabelArtifactCleanupTarget(
                relative_path=Path('source.json'),
                kind=LabelArtifactKind.GENERATION_SOURCE,
                size_bytes=len(b'source'),
            ),
        ),
        preserved_manifest_count=0,
        evidence=_evidence(),
    )
    (job_path / RECEIPT_FILE_NAME).write_text(prepared.model_dump_json(), encoding='utf-8')

    receipt = cleanup_completed_generation_artifacts(job_path, 3, _evidence())

    assert isinstance(receipt, CompletedLabelArtifactCleanupReceipt)
    assert not source_path.exists()
