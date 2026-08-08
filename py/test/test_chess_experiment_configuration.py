from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.experiment.base_configuration import BaseExperimentConfiguration
from src.experiment.configuration import (
    load_experiment_configuration,
    load_chess_experiment_configuration,
    validate_experiment_queue,
    write_resolved_chess_experiment,
)
from src.games.chess.configuration import ChessExperimentConfiguration
from src.games.go.configuration import GoExperimentConfiguration
from test_helpers.chess_configuration import CHESS_EXPERIMENT, CHESS_TRAINING
from src.util.frozen_model import JsonValue


CHESS_EXPERIMENT_TEMPLATE_PATH = Path('configs/chess-experiment-template.yaml')


def test_every_checked_in_experiment_uses_the_current_contract_and_dependency_lock() -> None:
    paths = tuple(sorted(Path('configs').glob('*-experiment-template.yaml')))
    configurations = validate_experiment_queue(paths)
    dependency_lock_sha256 = hashlib.sha256(Path('requirements-training.lock').read_bytes()).hexdigest()

    assert tuple(configuration.game for configuration in configurations) == ('chess', 'go', 'go')
    assert all(
        configuration.run.environment.dependency_lock_sha256 == dependency_lock_sha256
        for configuration in configurations
    )


def test_game_experiments_extend_the_shared_run_and_training_configuration() -> None:
    assert set(BaseExperimentConfiguration.model_fields) == {'run', 'training', 'evaluation'}
    assert issubclass(ChessExperimentConfiguration, BaseExperimentConfiguration)
    assert issubclass(GoExperimentConfiguration, BaseExperimentConfiguration)


@pytest.mark.parametrize(
    ('configuration_type', 'path'),
    (
        (ChessExperimentConfiguration, CHESS_EXPERIMENT_TEMPLATE_PATH),
        (GoExperimentConfiguration, Path('configs/go-7x7-experiment-template.yaml')),
    ),
)
def test_base_configuration_validates_shared_training_arguments(
    configuration_type: type[BaseExperimentConfiguration],
    path: Path,
) -> None:
    candidate = yaml.safe_load(path.read_text(encoding='utf-8'))
    candidate['training']['trainer']['global_batch_size'] += 1

    with pytest.raises(ValidationError, match='Global batch size must equal'):
        configuration_type.model_validate(candidate)


def test_chess_experiment_template_loads_canonical_runtime_configuration() -> None:
    configuration = load_chess_experiment_configuration(CHESS_EXPERIMENT_TEMPLATE_PATH)

    assert isinstance(configuration, ChessExperimentConfiguration)
    assert configuration.game == 'chess'
    assert configuration.training.network.hidden_size == 112
    assert configuration.chess.self_play.search.exploration_constant.value_at(0) == pytest.approx(1.5)
    assert configuration.training.topology.trainer.ddp_device_ids == (0,)
    assert configuration.chess.self_play.search.parallel_searches == 2
    assert configuration.chess.self_play.inference.inference_workers == 2
    assert configuration.chess.self_play.inference.inference_batch_size == 64
    assert configuration.chess.self_play.inference.outstanding_batches_per_worker == 2
    assert configuration.evaluation.cadence_seconds == 1200
    assert configuration.evaluation.dataset.path.endswith('chess-evaluation-v1.bin')
    assert configuration.evaluation.openings.opening_count == 50
    assert configuration.evaluation.engine.kind == 'stockfish'
    assert tuple(definition.kind for definition in configuration.evaluation.definitions) == (
        'fixed_dataset',
        'random',
        'previous_checkpoint',
        'stockfish',
    )


def test_experiment_owns_training_configuration_without_an_adapter() -> None:
    assert CHESS_TRAINING is CHESS_EXPERIMENT.training


def test_experiment_queue_validation_loads_multiple_experiments() -> None:
    configurations = validate_experiment_queue((CHESS_EXPERIMENT_TEMPLATE_PATH, CHESS_EXPERIMENT_TEMPLATE_PATH))

    assert len(configurations) == 2
    assert configurations[0] == configurations[1]


@pytest.mark.parametrize(
    ('path', 'board_size', 'action_count'),
    (
        (Path('configs/go-7x7-experiment-template.yaml'), 7, 50),
        (Path('configs/go-9x9-experiment-template.yaml'), 9, 82),
    ),
)
def test_go_experiments_resolve_deterministically(path: Path, board_size: int, action_count: int) -> None:
    configuration = load_experiment_configuration(path)

    assert isinstance(configuration, GoExperimentConfiguration)
    assert configuration.go.representation.board_size == board_size
    assert configuration.go.representation.action_count == action_count
    assert load_experiment_configuration(path) == configuration


def test_queue_validation_supports_both_games() -> None:
    configurations = validate_experiment_queue(
        (
            CHESS_EXPERIMENT_TEMPLATE_PATH,
            Path('configs/go-7x7-experiment-template.yaml'),
            Path('configs/go-9x9-experiment-template.yaml'),
        )
    )

    assert tuple(configuration.game for configuration in configurations) == ('chess', 'go', 'go')


@pytest.mark.parametrize(
    ('field_path', 'value', 'message'),
    (
        (('go', 'representation', 'board_size'), 8, 'Input should be 7 or 9'),
        (('go', 'rules', 'maximum_moves'), 10, 'twice the board point count'),
        (
            ('go', 'objective', 'root_value_blend'),
            {'kind': 'constant', 'value': 1.5},
            'must remain in',
        ),
        (('go', 'self_play', 'maximum_game_plies'), 200, 'Extra inputs are not permitted'),
    ),
)
def test_invalid_go_combinations_fail_precisely(field_path: tuple[str, ...], value: JsonValue, message: str) -> None:
    candidate = yaml.safe_load(Path('configs/go-7x7-experiment-template.yaml').read_text(encoding='utf-8'))
    owner = candidate
    for segment in field_path[:-1]:
        owner = owner[segment]
    owner[field_path[-1]] = value

    with pytest.raises(ValidationError, match=message):
        GoExperimentConfiguration.model_validate(candidate)


def test_resolved_experiment_round_trips_as_canonical_json(tmp_path: Path) -> None:
    configuration = load_chess_experiment_configuration(CHESS_EXPERIMENT_TEMPLATE_PATH)
    resolved_path = tmp_path / 'resolved-chess-experiment.json'

    write_resolved_chess_experiment(resolved_path, configuration)

    assert load_chess_experiment_configuration(resolved_path) == configuration


def test_legacy_run_topology_is_rejected() -> None:
    candidate = yaml.safe_load(CHESS_EXPERIMENT_TEMPLATE_PATH.read_text(encoding='utf-8'))
    candidate['run']['topology'] = {'trainer_device_type': 'cpu'}

    with pytest.raises(ValidationError, match='topology'):
        ChessExperimentConfiguration.model_validate(candidate)


def test_chess_evaluation_fields_are_rejected_from_shared_training() -> None:
    candidate = yaml.safe_load(CHESS_EXPERIMENT_TEMPLATE_PATH.read_text(encoding='utf-8'))
    candidate['training']['evaluation'] = {'opening_suite_path': 'openings.tsv'}

    with pytest.raises(ValidationError, match='evaluation'):
        ChessExperimentConfiguration.model_validate(candidate)


def test_game_specific_external_engines_are_validated() -> None:
    chess_candidate = yaml.safe_load(CHESS_EXPERIMENT_TEMPLATE_PATH.read_text(encoding='utf-8'))
    chess_candidate['evaluation']['engine'] = {
        'kind': 'katago',
        'executable_path': 'katago',
        'model_path': 'model.bin.gz',
        'analysis_configuration_path': 'analysis.cfg',
        'label_max_visits': 64,
        'match_max_visits': 32,
    }
    chess_candidate['evaluation']['definitions'] = [
        definition for definition in chess_candidate['evaluation']['definitions'] if definition['kind'] != 'stockfish'
    ]

    with pytest.raises(ValidationError, match='KataGo|Stockfish'):
        ChessExperimentConfiguration.model_validate(chess_candidate)


def test_network_rejects_unknown_parameters() -> None:
    candidate = yaml.safe_load(CHESS_EXPERIMENT_TEMPLATE_PATH.read_text(encoding='utf-8'))
    candidate['training']['network']['experimental_width'] = 256

    with pytest.raises(ValidationError, match='experimental_width'):
        ChessExperimentConfiguration.model_validate(candidate)


def test_training_configuration_is_frozen() -> None:
    configuration = load_chess_experiment_configuration(CHESS_EXPERIMENT_TEMPLATE_PATH)

    with pytest.raises(ValidationError, match='frozen'):
        configuration.training.save_path = 'different-path'


def test_validated_copy_reruns_field_validation() -> None:
    configuration = load_chess_experiment_configuration(CHESS_EXPERIMENT_TEMPLATE_PATH)

    with pytest.raises(ValidationError, match='greater than 0'):
        configuration.training.trainer.validated_copy(update={'global_batch_size': 0})


@pytest.mark.parametrize(
    ('owner_path', 'field_name', 'value'),
    (
        (('training', 'trainer'), 'num_workers', 2),
        (('training', 'topology'), 'max_concurrent_evaluations', 1),
        (('evaluation',), 'evaluate_initial_checkpoint', True),
        (('chess', 'self_play'), 'use_inference_cache', True),
        (('chess', 'self_play'), 'inference_cache_capacity', 250_000),
        (('chess', 'self_play', 'inference'), 'mode', 'cached'),
        (('chess', 'self_play', 'inference'), 'capacity', 250_000),
    ),
)
def test_removed_configuration_fields_are_rejected(
    owner_path: tuple[str, ...],
    field_name: str,
    value: JsonValue,
) -> None:
    candidate = yaml.safe_load(CHESS_EXPERIMENT_TEMPLATE_PATH.read_text(encoding='utf-8'))
    owner = candidate
    for segment in owner_path:
        owner = owner[segment]
    owner[field_name] = value

    with pytest.raises(ValidationError, match=field_name):
        ChessExperimentConfiguration.model_validate(candidate)
