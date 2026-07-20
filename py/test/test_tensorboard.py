from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from src.util.tensorboard import (
    TensorboardWriter,
    configure_tensorboard_run_directory,
    is_tensorboard_writer_active,
    log_scalar,
    log_scalars,
)


def test_tensorboard_uses_run_specific_artifact_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv('TRAINING_TENSORBOARD_LOG_PATH', str(tmp_path))
    monkeypatch.delenv('TRAINING_TENSORBOARD_RUN_DIRECTORY', raising=False)

    writer = TensorboardWriter(run=7, suffix='trainer', postfix_pid=False)

    assert Path(writer.log_folder) == tmp_path / 'run_7' / 'trainer'


def test_tensorboard_explicit_run_directory_combines_process_restarts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv('TRAINING_TENSORBOARD_LOG_PATH', str(tmp_path))
    monkeypatch.delenv('TRAINING_TENSORBOARD_RUN_DIRECTORY', raising=False)
    configure_tensorboard_run_directory('clean-retrain')

    first_process_writer = TensorboardWriter(run=1, suffix='trainer', postfix_pid=False)
    restarted_process_writer = TensorboardWriter(run=2, suffix='trainer', postfix_pid=False)
    with first_process_writer:
        log_scalar('training/loss', 2.0, iteration=1)
        log_scalars('evaluation/vs_random', {'wins': 20}, iteration=1)
    with restarted_process_writer:
        log_scalar('training/loss', 1.0, iteration=2)
        log_scalars('evaluation/vs_random', {'wins': 30}, iteration=2)

    expected_directory = tmp_path / 'clean-retrain' / 'trainer'
    assert Path(first_process_writer.log_folder) == expected_directory
    assert Path(restarted_process_writer.log_folder) == expected_directory
    event_accumulator = EventAccumulator(str(expected_directory)).Reload()
    scalar_events = event_accumulator.Scalars('training/loss')
    assert tuple((event.step, event.value) for event in scalar_events) == ((1, 2.0), (2, 1.0))
    child_event_accumulator = EventAccumulator(str(expected_directory / 'evaluation' / 'vs_random' / 'wins')).Reload()
    child_scalar_events = child_event_accumulator.Scalars('evaluation/vs_random')
    assert tuple((event.step, event.value) for event in child_scalar_events) == ((1, 20.0), (2, 30.0))


@pytest.mark.parametrize('run_directory', ('', '-invalid', '../escape', 'nested/run', 'run name'))
def test_tensorboard_rejects_unsafe_run_directory(run_directory: str) -> None:
    with pytest.raises(ValueError, match='TensorBoard run directory'):
        configure_tensorboard_run_directory(run_directory)


def write_thread_event(run: int) -> Path:
    writer = TensorboardWriter(run=run, suffix='usage', postfix_pid=False)
    with writer:
        log_scalar('utilization', float(run), iteration=0)
    return Path(writer.log_folder)


def test_tensorboard_context_is_isolated_between_threads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv('TRAINING_TENSORBOARD_LOG_PATH', str(tmp_path))
    monkeypatch.delenv('TRAINING_TENSORBOARD_RUN_DIRECTORY', raising=False)

    with ThreadPoolExecutor(max_workers=2) as executor:
        paths = tuple(executor.map(write_thread_event, (1, 2)))

    assert paths == (
        tmp_path / 'run_1' / 'usage',
        tmp_path / 'run_2' / 'usage',
    )


def test_disabled_tensorboard_writer_creates_no_event_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv('TRAINING_TENSORBOARD_LOG_PATH', str(tmp_path))
    monkeypatch.delenv('TRAINING_TENSORBOARD_RUN_DIRECTORY', raising=False)

    writer = TensorboardWriter(run=3, suffix='self_play', postfix_pid=False, enabled=False)
    with writer:
        log_scalar('mcts/average_search_depth', 4.0, iteration=0)

    assert not Path(writer.log_folder).exists()


def test_tensorboard_writer_active_detection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv('TRAINING_TENSORBOARD_LOG_PATH', str(tmp_path))
    monkeypatch.delenv('TRAINING_TENSORBOARD_RUN_DIRECTORY', raising=False)

    assert not is_tensorboard_writer_active()

    with TensorboardWriter(run=4, suffix='self_play', postfix_pid=False):
        assert is_tensorboard_writer_active()

    with TensorboardWriter(run=5, suffix='self_play', postfix_pid=False, enabled=False):
        assert not is_tensorboard_writer_active()

    assert not is_tensorboard_writer_active()
