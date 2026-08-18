from pathlib import Path
from typing import NoReturn

import pytest
from src.experiment import training_startup
from src.experiment.configuration import ExperimentConfiguration
from src.experiment.run_contract import load_approval_record

CHESS_EXPERIMENT_PATH = Path(__file__).parent / 'configs' / 'chess-experiment.yaml'


def test_missing_approval_creates_no_tensorboard_run_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tensorboard_path = tmp_path / 'tensorboard'
    approval_path = tmp_path / 'missing-approval.json'
    monkeypatch.setenv('TRAINING_TENSORBOARD_LOG_PATH', str(tensorboard_path))

    def reject_missing_approval(
        experiment: ExperimentConfiguration,
        expected_source_revision: str,
        candidate_approval_path: Path,
    ) -> NoReturn:
        del experiment, expected_source_revision
        load_approval_record(candidate_approval_path)
        raise AssertionError('A missing approval must fail to load.')

    monkeypatch.setattr(training_startup, 'prepare_experiment_training_run', reject_missing_approval)

    with pytest.raises(FileNotFoundError):
        training_startup.prepare_training_startup(CHESS_EXPERIMENT_PATH, 'expected-revision', approval_path)

    assert not tensorboard_path.exists()
