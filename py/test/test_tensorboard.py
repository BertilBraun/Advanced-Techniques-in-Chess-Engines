from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from src.util.tensorboard import TensorboardWriter, log_scalar


def test_tensorboard_uses_run_specific_artifact_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv('TRAINING_TENSORBOARD_LOG_PATH', str(tmp_path))

    writer = TensorboardWriter(run=7, suffix='trainer', postfix_pid=False)

    assert Path(writer.log_folder) == tmp_path / 'run_7' / 'trainer'


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

    with ThreadPoolExecutor(max_workers=2) as executor:
        paths = tuple(executor.map(write_thread_event, (1, 2)))

    assert paths == (
        tmp_path / 'run_1' / 'usage',
        tmp_path / 'run_2' / 'usage',
    )
