from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.experiment.chess_experiment import (
    ChessExperimentConfiguration,
    build_chess_training_args,
    load_chess_experiment_configuration,
    validate_experiment_queue,
    write_resolved_chess_experiment,
)


DEFAULT_EXPERIMENT_PATH = Path('configs/chess-default-experiment.yaml')


def test_default_chess_experiment_loads_from_yaml() -> None:
    configuration = load_chess_experiment_configuration(DEFAULT_EXPERIMENT_PATH)

    assert isinstance(configuration, ChessExperimentConfiguration)
    assert configuration.chess.game == 'chess'
    assert configuration.chess.network.hidden_size == 112
    assert configuration.chess.self_play.exploration_constant == pytest.approx(1.5)
    assert configuration.run.topology.training.ddp_device_ids == (0,)
    assert configuration.run.topology.self_play.search.parallel_searches == 2
    assert configuration.run.topology.evaluation.inference.cache_capacity_per_process == 50_000
    assert configuration.chess.evaluation.dataset is not None
    assert configuration.chess.evaluation.dataset.path.endswith('memory_0_chess_database.hdf5')
    assert configuration.chess.evaluation.stockfish is not None
    assert configuration.chess.evaluation.stockfish.skill_levels == (0, 1, 2, 3)


def test_chess_training_args_are_built_from_experiment_settings() -> None:
    configuration = load_chess_experiment_configuration(DEFAULT_EXPERIMENT_PATH)

    training_args = build_chess_training_args(configuration)

    assert training_args.network.hidden_size == configuration.chess.network.hidden_size
    assert training_args.network is configuration.chess.network
    assert training_args.self_play.mcts.c_param == pytest.approx(configuration.chess.self_play.exploration_constant)
    assert training_args.self_play.starting_temperature == pytest.approx(
        configuration.chess.self_play.starting_temperature
    )
    assert training_args.evaluation is not None
    assert training_args.evaluation.search_exploration_constant == pytest.approx(
        configuration.chess.evaluation.exploration_constant
    )
    assert training_args.cluster.trainer_ddp_device_ids == configuration.run.topology.training.ddp_device_ids
    assert training_args.evaluation.dataset_path == configuration.chess.evaluation.dataset.path
    assert training_args.evaluation.stockfish_skill_levels == configuration.chess.evaluation.stockfish.skill_levels


def test_experiment_queue_validation_loads_multiple_experiments() -> None:
    configurations = validate_experiment_queue((DEFAULT_EXPERIMENT_PATH, DEFAULT_EXPERIMENT_PATH))

    assert len(configurations) == 2
    assert configurations[0] == configurations[1]


def test_resolved_experiment_round_trips_as_canonical_json(tmp_path: Path) -> None:
    configuration = load_chess_experiment_configuration(DEFAULT_EXPERIMENT_PATH)
    resolved_path = tmp_path / 'resolved-chess-experiment.json'

    write_resolved_chess_experiment(resolved_path, configuration)

    assert load_chess_experiment_configuration(resolved_path) == configuration


def test_flat_topology_fields_are_rejected() -> None:
    candidate = yaml.safe_load(DEFAULT_EXPERIMENT_PATH.read_text(encoding='utf-8'))
    candidate['run']['topology']['trainer_device_type'] = 'cpu'

    with pytest.raises(ValidationError, match='trainer_device_type'):
        ChessExperimentConfiguration.model_validate(candidate)


def test_chess_evaluation_fields_are_rejected_from_shared_evaluation() -> None:
    candidate = yaml.safe_load(DEFAULT_EXPERIMENT_PATH.read_text(encoding='utf-8'))
    candidate['run']['evaluation']['opening_suite_path'] = 'openings.tsv'

    with pytest.raises(ValidationError, match='opening_suite_path'):
        ChessExperimentConfiguration.model_validate(candidate)


def test_network_rejects_unknown_parameters() -> None:
    candidate = yaml.safe_load(DEFAULT_EXPERIMENT_PATH.read_text(encoding='utf-8'))
    candidate['chess']['network']['experimental_width'] = 256

    with pytest.raises(ValidationError, match='experimental_width'):
        ChessExperimentConfiguration.model_validate(candidate)


def test_direct_inference_cannot_be_combined_with_a_cache() -> None:
    candidate = yaml.safe_load(DEFAULT_EXPERIMENT_PATH.read_text(encoding='utf-8'))
    candidate['run']['topology']['self_play']['inference']['direct'] = {
        'inference_workers': 1,
        'inference_batch_size': 64,
        'outstanding_batches_per_worker': 2,
    }

    with pytest.raises(ValidationError, match='mutually exclusive'):
        ChessExperimentConfiguration.model_validate(candidate)
