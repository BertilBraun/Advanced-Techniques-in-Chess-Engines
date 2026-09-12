from __future__ import annotations

from pathlib import Path
from typing import NoReturn, cast

import pytest
from src.experiment import training_startup
from src.experiment.configuration import ExperimentConfiguration
from src.experiment.run import ExperimentRunManifest
from src.experiment.run_contract import load_approval_record
from test_helpers.configuration_paths import TEST_CONFIG_DIRECTORY

CHESS_EXPERIMENT_PATH = TEST_CONFIG_DIRECTORY / 'chess-experiment.yaml'


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


def test_successful_resume_clears_stale_run_outcome(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    experiment = training_startup.load_experiment_configuration(CHESS_EXPERIMENT_PATH)
    training = experiment.training.model_copy(update={'save_path': str(tmp_path)})
    resumed_experiment = experiment.model_copy(update={'training': training})
    outcome_path = tmp_path / 'run-outcome.json'
    outcome_path.write_text('{"status":"failed"}\n', encoding='utf-8')

    monkeypatch.setattr(training_startup, 'load_experiment_configuration', lambda _: resumed_experiment)
    monkeypatch.setattr(
        training_startup,
        'prepare_experiment_training_run',
        lambda *_: cast(ExperimentRunManifest, None),
    )
    monkeypatch.setattr(training_startup, 'write_resolved_experiment', lambda *_: None)
    monkeypatch.setattr(training_startup, 'configure_tensorboard_run_directory', lambda _: None)
    monkeypatch.setattr(training_startup, 'get_run_id', lambda: 7)

    startup = training_startup.prepare_training_startup(CHESS_EXPERIMENT_PATH, 'revision', tmp_path / 'approval.json')

    assert startup.run_id == 7
    assert not outcome_path.exists()
