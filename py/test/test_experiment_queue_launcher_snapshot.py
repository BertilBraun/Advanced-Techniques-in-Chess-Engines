from pathlib import Path
from unittest.mock import patch

from src.experiment_queue.configuration import (
    QueueConfiguration,
    QueuedExperiment,
    ResourceRequest,
    ResourceSlot,
    RunnerCommand,
)
from src.experiment_queue.launcher_snapshot import LauncherSnapshot, create_launcher_snapshot
from src.experiment_queue.runner import ExperimentQueueRunner
from src.experiment_queue.validation import ValidatedQueue, ValidatedQueuedExperiment
from src.experiment_queue.workspace import ExperimentWorkspace, ExperimentWorkspaceManager


def test_launcher_snapshot_survives_control_helper_removal(tmp_path: Path) -> None:
    control_source = tmp_path / 'control' / 'experiment_queue'
    control_source.mkdir(parents=True)
    linux_source = control_source / 'linux_child.py'
    worktree_source = control_source / 'worktree_child.py'
    linux_source.write_text("print('stable-linux')\n", encoding='utf-8')
    worktree_source.write_text("print('stable-worktree')\n", encoding='utf-8')

    snapshot = create_launcher_snapshot(tmp_path / 'persistent-runtime', control_source)
    linux_source.unlink()
    worktree_source.unlink()

    assert snapshot.linux_child.read_text(encoding='utf-8') == "print('stable-linux')\n"
    assert snapshot.worktree_child.read_text(encoding='utf-8') == "print('stable-worktree')\n"
    assert snapshot.linux_child.parent.is_relative_to(control_source) is False
    assert snapshot.worktree_child.parent.is_relative_to(control_source) is False

    snapshot.close()
    assert not snapshot.directory.exists()


def test_pending_launch_uses_runner_initialization_snapshot_after_control_helpers_are_removed(
    tmp_path: Path,
) -> None:
    control_source = tmp_path / 'control' / 'experiment_queue'
    control_source.mkdir(parents=True)
    linux_source = control_source / 'linux_child.py'
    worktree_source = control_source / 'worktree_child.py'
    linux_source.write_text("print('initialized-linux')\n", encoding='utf-8')
    worktree_source.write_text("print('initialized-worktree')\n", encoding='utf-8')
    experiment_file = tmp_path / 'repository' / 'experiment.yaml'
    experiment_file.parent.mkdir()
    experiment_file.write_text('experiment', encoding='utf-8')
    experiment = QueuedExperiment(
        experiment_id='pending',
        experiment_file=experiment_file,
        source_revision='1' * 40,
        resources=ResourceRequest(cuda_device_count=0, cpu_core_count=1, ram_limit_bytes=1_000),
    )
    configuration = QueueConfiguration(
        runner=RunnerCommand(command=('python',)),
        repository_directory=experiment_file.parent,
        worktree_root=tmp_path / 'worktrees',
        runtime_directory=tmp_path / 'runtime',
        tensorboard_log_directory=tmp_path / 'tensorboard',
        slots=(
            ResourceSlot(
                slot_id='slot',
                cuda_devices=(),
                cpu_affinity=(0,),
                ram_capacity_bytes=2_000,
                log_directory=tmp_path / 'logs',
            ),
        ),
        experiments=(experiment,),
        summary_path=tmp_path / 'summary.json',
    )
    validated = ValidatedQueuedExperiment(definition=experiment, configuration_sha256='2' * 64)
    queue = ValidatedQueue(configuration=configuration, experiments=(validated,), fingerprint='3' * 64)
    source_worktree = tmp_path / 'worktrees' / 'pending'
    worktree_experiment = source_worktree / 'experiment.yaml'
    source_worktree.mkdir(parents=True)
    worktree_experiment.write_text('experiment', encoding='utf-8')
    workspace = ExperimentWorkspace(
        source_worktree=source_worktree,
        runtime_directory=configuration.runtime_directory,
        tensorboard_log_directory=configuration.tensorboard_log_directory,
        experiment_file=worktree_experiment,
        preserved_configuration_directory=configuration.runtime_directory / 'evidence',
    )

    with patch('src.experiment_queue.runner.experiment_queue_source_directory', return_value=control_source):
        runner = ExperimentQueueRunner(lambda: queue)
    linux_source.unlink()
    worktree_source.unlink()

    def reject_after_snapshot_check(*, launcher_snapshot: LauncherSnapshot, **_: object) -> None:
        assert launcher_snapshot.linux_child.read_text(encoding='utf-8') == "print('initialized-linux')\n"
        assert launcher_snapshot.worktree_child.read_text(encoding='utf-8') == "print('initialized-worktree')\n"
        raise OSError('test launch stop')

    with (
        patch.object(ExperimentWorkspaceManager, 'create', return_value=workspace),
        patch('src.experiment_queue.runner.launch_process', side_effect=reject_after_snapshot_check),
    ):
        runner._launch_ready_experiments()

    assert runner.summary.experiments[0].status == 'preparation_failed'
    runner._launcher_snapshot.close()
