from pathlib import Path

import pytest

from src.util.tensorboard import TensorboardWriter


def test_tensorboard_uses_run_specific_artifact_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv('TRAINING_TENSORBOARD_LOG_PATH', str(tmp_path))

    writer = TensorboardWriter(run=7, suffix='trainer', postfix_pid=False)

    assert Path(writer.log_folder) == tmp_path / 'run_7' / 'trainer'
