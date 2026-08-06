from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from dataclasses import FrozenInstanceError
from pydantic import ValidationError

from src.experiment.chess_experiment import (
    ChessExperimentConfiguration,
    load_chess_experiment_configuration,
    validate_experiment_queue,
    write_resolved_chess_experiment,
)
from src.settings import CHESS_EXPERIMENT, TRAINING_ARGS


DEFAULT_EXPERIMENT_PATH = Path('configs/chess-default-experiment.yaml')


def test_default_chess_experiment_loads_canonical_runtime_configuration() -> None:
    configuration = load_chess_experiment_configuration(DEFAULT_EXPERIMENT_PATH)

    assert isinstance(configuration, ChessExperimentConfiguration)
    assert configuration.chess.game == 'chess'
    assert configuration.training.network.hidden_size == 112
    assert configuration.training.self_play.mcts.c_param == pytest.approx(1.5)
    assert configuration.training.cluster.trainer_ddp_device_ids == (0,)
    assert configuration.training.self_play.mcts.num_parallel_searches == 2
    assert configuration.chess.evaluation.inference_cache_capacity == 50_000
    assert configuration.chess.evaluation.dataset_path is not None
    assert configuration.chess.evaluation.dataset_path.endswith('memory_0_chess_database.hdf5')
    assert configuration.chess.evaluation.stockfish_skill_levels == (0, 1, 2, 3)


def test_settings_exposes_experiment_training_without_an_adapter() -> None:
    assert TRAINING_ARGS is CHESS_EXPERIMENT.training


def test_experiment_queue_validation_loads_multiple_experiments() -> None:
    configurations = validate_experiment_queue((DEFAULT_EXPERIMENT_PATH, DEFAULT_EXPERIMENT_PATH))

    assert len(configurations) == 2
    assert configurations[0] == configurations[1]


def test_resolved_experiment_round_trips_as_canonical_json(tmp_path: Path) -> None:
    configuration = load_chess_experiment_configuration(DEFAULT_EXPERIMENT_PATH)
    resolved_path = tmp_path / 'resolved-chess-experiment.json'

    write_resolved_chess_experiment(resolved_path, configuration)

    assert load_chess_experiment_configuration(resolved_path) == configuration


def test_legacy_run_topology_is_rejected() -> None:
    candidate = yaml.safe_load(DEFAULT_EXPERIMENT_PATH.read_text(encoding='utf-8'))
    candidate['run']['topology'] = {'trainer_device_type': 'cpu'}

    with pytest.raises(ValidationError, match='topology'):
        ChessExperimentConfiguration.model_validate(candidate)


def test_chess_evaluation_fields_are_rejected_from_shared_training() -> None:
    candidate = yaml.safe_load(DEFAULT_EXPERIMENT_PATH.read_text(encoding='utf-8'))
    candidate['training']['evaluation'] = {'opening_suite_path': 'openings.tsv'}

    with pytest.raises(ValidationError, match='evaluation'):
        ChessExperimentConfiguration.model_validate(candidate)


def test_network_rejects_unknown_parameters() -> None:
    candidate = yaml.safe_load(DEFAULT_EXPERIMENT_PATH.read_text(encoding='utf-8'))
    candidate['training']['network']['experimental_width'] = 256

    with pytest.raises(ValidationError, match='experimental_width'):
        ChessExperimentConfiguration.model_validate(candidate)


def test_training_configuration_is_frozen() -> None:
    configuration = load_chess_experiment_configuration(DEFAULT_EXPERIMENT_PATH)

    with pytest.raises(FrozenInstanceError):
        configuration.training.save_path = 'different-path'
