from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest

from src.az.calibration.models import (
    InitialTraceModelIdentity,
    SearchTraceCollectionPayload,
    SearchTraceObservation,
    SearchTraceSampleLineage,
    SearchTraceSnapshot,
    publish_trace_collection_artifact,
)
from src.az.config.seeds import (
    SEED_DERIVATION_VERSION,
    SearchTraceSampleSeedCoordinates,
    SeedPurpose,
    derive_seed,
)
from src.az.config.dependency_lock import parse_pinned_dependency_lock
from src.az.config.artifacts import CalibrationArtifactReference
from src.az.config.manifest import (
    DependencyDeclaration,
    build_manifest,
    current_python_build,
    file_sha256,
)
from src.az.config.serialization import write_resolved_configuration
from src.az.config.root import validate_resolved_configuration
from src.az.config.search import FixedSearchBudget, VisitMarginAdaptiveRule
from src.az.experiment.environment import inspect_hardware
from src.az.experiment.lifecycle import (
    ExperimentRunRepository,
    ExperimentPhase,
    ExperimentStatus,
    RunArtifactKind,
    require_exact_artifact_files,
)
from src.az.experiment.commit_journal import ReplayCommitJournal
from src.az.experiment.cli import main
from src.az.experiment.smoke import local_cpu_smoke_configuration
from src.az.replay.envelope import ReplayRecord
from test.unit.go_stage5_helpers import envelope


def _freeze(tmp_path: Path) -> ExperimentRunRepository:
    configuration = local_cpu_smoke_configuration()
    configuration_path = tmp_path / "smoke.json"
    write_resolved_configuration(configuration_path, configuration)
    repository = ExperimentRunRepository((tmp_path / "run").resolve())
    lock_path = Path("requirements-training.lock").resolve()
    manifest = build_manifest(
        configuration=configuration,
        repository_root=Path("..").resolve(),
        build=current_python_build("unit-test"),
        dependencies=DependencyDeclaration(
            lock_file=lock_path,
            lock_file_sha256=file_sha256(lock_path),
            packages=parse_pinned_dependency_lock(lock_path),
        ),
        hardware=inspect_hardware(repository.directory),
    )
    repository.freeze(configuration_path, UUID(int=11), manifest, Path("..").resolve())
    return repository


def test_frozen_run_authenticates_configuration_source_and_stop_resume(
    tmp_path: Path,
) -> None:
    repository = _freeze(tmp_path)
    initial = repository.load()

    assert initial.run_id == UUID(int=11)
    assert initial.source_revision
    request = repository.request_stop()
    stopped = initial.model_copy(
        update={
            "status": ExperimentStatus.STOPPED,
            "stop_requested": True,
        }
    )
    repository.save(initial, stopped)
    resumed = repository.resume()

    assert request.run_id == resumed.run_id
    assert resumed.status is ExperimentStatus.READY
    assert not resumed.stop_requested
    assert not repository.stop_path.exists()


def test_frozen_run_rejects_configuration_tampering(tmp_path: Path) -> None:
    repository = _freeze(tmp_path)
    repository.configuration_path.write_bytes(
        repository.configuration_path.read_bytes() + b" "
    )

    with pytest.raises(ValueError, match="configuration"):
        repository.load()


def test_run_state_rejects_concurrent_overwrite(tmp_path: Path) -> None:
    repository = _freeze(tmp_path)
    initial = repository.load()
    first = initial.model_copy(
        update={"updated_at": initial.updated_at.replace(microsecond=1)}
    )
    second = initial.model_copy(
        update={"updated_at": initial.updated_at.replace(microsecond=2)}
    )

    repository.save(initial, first)
    with pytest.raises(ValueError, match="concurrently"):
        repository.save(initial, second)


def test_explicit_recovery_handles_crash_between_running_state_and_lease(
    tmp_path: Path,
) -> None:
    repository = _freeze(tmp_path)
    ready = repository.load()
    running = ready.model_copy(update={"status": ExperimentStatus.RUNNING})
    repository.save(ready, running)

    recovered = repository.resume(recover_crash=True)

    assert recovered.status is ExperimentStatus.READY
    assert not repository.lease_path.exists()


def test_stop_at_training_deadline_advances_to_resumable_evaluation(
    tmp_path: Path,
) -> None:
    repository = _freeze(tmp_path)
    ready = repository.load()
    running = ready.model_copy(update={"status": ExperimentStatus.RUNNING})
    repository.save(ready, running)
    repository.acquire_lease(running)
    repository.request_stop()

    stopped = repository.complete_training_at_stop(running, (), 12, 12)

    assert stopped.status is ExperimentStatus.STOPPED
    assert stopped.completed_phases == (ExperimentPhase.TRAINING_RUN,)
    assert stopped.next_phase is ExperimentPhase.EVALUATION
    resumed = repository.resume()
    assert resumed.status is ExperimentStatus.READY
    assert resumed.next_phase is ExperimentPhase.EVALUATION


def test_authoritative_directory_rejects_unregistered_evaluation_result(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "evaluation-results"
    directory.mkdir()
    expected = directory / "expected.json"
    expected.write_text("{}", encoding="utf-8")
    injected = directory / "injected.json"
    injected.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="unregistered"):
        require_exact_artifact_files(directory, "*.json", (expected,))


def test_calibration_cli_publishes_reference_from_registered_committed_trace(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = _freeze(tmp_path)
    state = repository.load()
    sample_id = UUID(int=1_001)
    journal = ReplayCommitJournal(
        (repository.directory / "replay-commits.azc").resolve()
    )
    journal.commit(
        (
            ReplayRecord(
                envelope=envelope(1).model_copy(update={"sample_id": sample_id}),
                payload=b"payload",
            ),
        )
    )
    repository.register_artifact(RunArtifactKind.REPLAY_COMMIT_JOURNAL, journal.path)
    trace_seed = derive_seed(
        91,
        SearchTraceSampleSeedCoordinates(
            purpose=SeedPurpose.SEARCH_TRACE_SAMPLE,
            process_index=0,
            worker_index=0,
            game_index=1,
            ply=0,
        ),
    )
    snapshots = tuple(
        SearchTraceSnapshot(
            simulations=simulations,
            root_policy=(1.0, *(0.0 for _ in range(9))),
            root_visits=(simulations, *(0 for _ in range(9))),
            root_value=0.25,
        )
        for simulations in (2, 4, 6, 8)
    )
    trace_path = publish_trace_collection_artifact(
        (repository.directory / "search-traces").resolve(),
        SearchTraceCollectionPayload(
            artifact_id=UUID(int=2_001),
            source_run_id=state.run_id,
            source_configuration_sha256=state.resolved_configuration_sha256,
            source_model=InitialTraceModelIdentity(
                kind="initial_model",
                model_initialization_seed=7,
                model_configuration_sha256="a" * 64,
            ),
            game_id=UUID(int=3_001),
            replay_sample_id=sample_id,
            lifecycle="completed_game_awaiting_replay_commit",
            game_configuration_sha256="b" * 64,
            native_state_hash=1,
            encoding_planes=5,
            encoding_board_size=3,
            canonical_encoding=(0,) * 45,
            legal_actions=tuple(range(10)),
            observation=SearchTraceObservation(
                source_position_id=sample_id,
                prefixes=snapshots[:-1],
                full=snapshots[-1],
            ),
            seed_lineage=SearchTraceSampleLineage(
                derivation_version=SEED_DERIVATION_VERSION,
                root_seed=91,
                process_index=0,
                worker_index=0,
                game_index=1,
                ply=0,
                trace_sample_seed=trace_seed,
            ),
        ),
    )
    repository.register_artifact(RunArtifactKind.SEARCH_TRACE, trace_path)
    request_path = tmp_path / "calibration-request.json"
    request_path.write_text(
        """
{
  "schema_version": 1,
  "artifact_id": "00000000-0000-0000-0000-000000000063",
  "candidates": [{
    "minimum_simulations": 2,
    "check_interval_simulations": 2,
    "required_top_visit_fraction": 0.75,
    "required_top_two_margin": 0.5
  }],
  "acceptance_rule": {
    "kind": "maximum_mean_disagreement",
    "maximum_policy_total_variation": 1.0,
    "maximum_value_absolute_error": 2.0
  }
}
""".strip(),
        encoding="utf-8",
    )

    assert (
        main(
            (
                "calibrate",
                "--run-directory",
                str(repository.directory),
                "--request",
                str(request_path),
            )
        )
        == 0
    )

    output = capsys.readouterr().out
    assert '"artifact_id": "00000000-0000-0000-0000-000000000063"' in output
    reference = CalibrationArtifactReference.model_validate_json(output)
    assert reference.artifact_root == "reference_artifacts"
    calibration_artifacts = tuple(
        artifact
        for artifact in repository.load().artifacts
        if artifact.kind is RunArtifactKind.CALIBRATION
    )
    assert len(calibration_artifacts) == 1

    adaptive = local_cpu_smoke_configuration()
    adaptive = validate_resolved_configuration(
        adaptive.model_copy(
            update={
                "search": adaptive.search.model_copy(
                    update={
                        "budget": FixedSearchBudget(kind="fixed", simulations=8),
                        "stopping": VisitMarginAdaptiveRule(
                            kind="visit_margin",
                            minimum_simulations=2,
                            check_interval_simulations=2,
                            required_top_visit_fraction=0.75,
                            required_top_two_margin=0.5,
                            calibration=reference,
                        ),
                    }
                )
            }
        ).model_dump()
    )
    adaptive_path = tmp_path / "adaptive.json"
    write_resolved_configuration(adaptive_path, adaptive)
    dependency_lock = Path("requirements-training.lock").resolve()
    source_root = Path("..").resolve()
    adaptive_repository = ExperimentRunRepository((tmp_path / "adaptive-run").resolve())
    adaptive_manifest = build_manifest(
        configuration=adaptive,
        repository_root=source_root,
        build=current_python_build("unit-test"),
        dependencies=DependencyDeclaration(
            lock_file=dependency_lock,
            lock_file_sha256=file_sha256(dependency_lock),
            packages=parse_pinned_dependency_lock(dependency_lock),
        ),
        hardware=inspect_hardware(tmp_path.resolve()),
    )
    frozen = adaptive_repository.freeze(
        adaptive_path,
        UUID(int=12),
        adaptive_manifest,
        source_root,
        repository.directory,
    )
    copied = tuple(
        artifact
        for artifact in frozen.artifacts
        if artifact.kind is RunArtifactKind.REFERENCE_ARTIFACT
    )
    assert len(copied) == 1
    assert (
        adaptive_repository.directory / "reference-artifacts" / reference.path
    ).is_file()
