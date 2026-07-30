from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from uuid import UUID

import pytest

from src.az.experiment.lifecycle import (
    ExperimentPhase,
    ExperimentRunRepository,
    ExperimentRunState,
    ExperimentStatus,
)
from src.az.experiment.commit_journal import ReplayCommitJournal
from src.az.config.dependency_lock import parse_pinned_dependency_lock
from src.az.config.manifest import (
    DependencyDeclaration,
    build_manifest,
    current_python_build,
    file_sha256,
)
from src.az.config.serialization import write_resolved_configuration
from src.az.experiment.environment import inspect_hardware
from src.az.experiment.smoke import local_cpu_smoke_configuration

pytest.importorskip('az_go_native', reason='focused native Go extension has not been built')
pytestmark = pytest.mark.integration


def _command(*arguments: str) -> None:
    subprocess.run(
        (sys.executable, '-m', 'src.az.experiment.cli', *arguments),
        check=True,
        capture_output=True,
        text=True,
        timeout=45,
    )


def test_real_cpu_experiment_lifecycle_completes_in_seconds(tmp_path: Path) -> None:
    configuration = tmp_path / 'smoke.resolved.json'
    run_directory = (tmp_path / 'runs' / 'go-local-readiness').resolve()
    started = time.perf_counter()

    resolved = local_cpu_smoke_configuration()
    write_resolved_configuration(configuration, resolved)
    dependency_lock = Path('requirements-training.lock').resolve()
    source_root = Path('..').resolve()
    manifest = build_manifest(
        configuration=resolved,
        repository_root=source_root,
        build=current_python_build('system-smoke'),
        dependencies=DependencyDeclaration(
            lock_file=dependency_lock,
            lock_file_sha256=file_sha256(dependency_lock),
            packages=parse_pinned_dependency_lock(dependency_lock),
        ),
        hardware=inspect_hardware(tmp_path.resolve()),
    )
    ExperimentRunRepository(run_directory).freeze(
        configuration,
        UUID('00000000-0000-0000-0000-000000000711'),
        manifest,
        source_root,
    )
    training_process = subprocess.Popen(
        (
            sys.executable,
            '-m',
            'src.az.experiment.cli',
            'training-run',
            '--run-directory',
            str(run_directory),
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    state_path = run_directory / ExperimentRunRepository.STATE_FILENAME
    stop_deadline = time.monotonic() + 20
    while time.monotonic() < stop_deadline:
        if (
            state_path.is_file()
            and ExperimentRunState.model_validate_json(state_path.read_bytes()).status is ExperimentStatus.RUNNING
        ):
            break
        if training_process.poll() is not None:
            raise AssertionError('Training run exited before the stop request.')
        time.sleep(0.05)
    else:
        training_process.kill()
        raise AssertionError('Training run did not enter its resumable running state in time.')
    time.sleep(0.25)
    _command('stop', '--run-directory', str(run_directory))
    stdout, stderr = training_process.communicate(timeout=15)
    assert training_process.returncode == 0, (stdout, stderr)

    stopped = ExperimentRunRepository(run_directory).load()
    commit_path = run_directory / 'replay-commits.azc'
    committed_before_resume = ReplayCommitJournal(commit_path.resolve())
    sample_ids_before_resume = committed_before_resume.sample_ids
    next_indices_before_resume = committed_before_resume.next_game_indices(1)
    assert stopped.status is ExperimentStatus.STOPPED
    assert stopped.next_phase is ExperimentPhase.TRAINING_RUN
    assert stopped.self_play_elapsed_seconds is not None
    assert 0 < stopped.self_play_elapsed_seconds < 12

    _command('resume', '--recover-crash', '--run-directory', str(run_directory))

    state = ExperimentRunRepository(run_directory).load()
    committed_after_resume = ReplayCommitJournal(commit_path.resolve())
    assert state.status is ExperimentStatus.COMPLETE
    assert state.next_phase is ExperimentPhase.COMPLETE
    assert state.completed_phases == (
        ExperimentPhase.TRAINING_RUN,
        ExperimentPhase.EVALUATION,
        ExperimentPhase.REPORTING,
    )
    assert (run_directory / 'report' / 'report.json').is_file()
    assert (run_directory / 'report' / 'report.md').is_file()
    assert (run_directory / 'report' / 'report.csv').is_file()
    assert any(artifact.kind.value == 'checkpoint_pointer' for artifact in state.artifacts)
    assert state.self_play_elapsed_seconds == 12
    assert committed_after_resume.sample_ids
    assert sample_ids_before_resume < committed_after_resume.sample_ids
    assert committed_after_resume.next_game_indices(1)[0] > next_indices_before_resume[0]
    assert time.perf_counter() - started < 45
