from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError
from src.experiment.base_configuration import BaseExperimentConfiguration
from src.experiment.configuration import (
    experiment_configuration_sha256,
    load_chess_experiment_configuration,
    load_experiment_configuration,
    validate_experiment_queue,
    write_resolved_chess_experiment,
)
from src.games.chess.configuration import ChessExperimentConfiguration
from src.games.go.configuration import GoExperimentConfiguration
from src.self_play.configuration import EnabledForcedPlayoutConfiguration, SdpaBackend
from src.self_play.parameters import (
    FirstPlayUrgencyParameters,
    ParentValueFirstPlayUrgencyParameters,
    ReducedParentValueFirstPlayUrgencyParameters,
    ZeroFirstPlayUrgencyParameters,
)
from src.training.configuration import TrainingCompilation, TrainingPrecision
from src.util.frozen_model import JsonValue
from test_helpers.chess_configuration import CHESS_EXPERIMENT, CHESS_TRAINING

CHESS_EXPERIMENT_TEMPLATE_PATH = Path('test/configs/chess-experiment.yaml')
OPTIMAL_CHESS_EXPERIMENT_PATH = Path('configs/production/vast-chess-8gpu-optimal.yaml')


def test_test_experiment_fixtures_use_the_current_contract_and_dependency_lock() -> None:
    paths = tuple(sorted(Path('test/configs').glob('*-experiment.yaml')))
    configurations = validate_experiment_queue(paths)
    dependency_lock_sha256 = hashlib.sha256(Path('requirements-training.lock').read_bytes()).hexdigest()

    assert configurations
    assert all(
        configuration.run.environment.dependency_lock_sha256 == dependency_lock_sha256
        for configuration in configurations
    )


def test_every_screening_configuration_parses() -> None:
    paths = tuple(sorted(Path('configs/screening').rglob('*.yaml')))

    assert all(load_experiment_configuration(path) for path in paths)


def test_optimal_chess_experiment_uses_progressive_replay_and_parallel_search() -> None:
    configuration = load_experiment_configuration(OPTIMAL_CHESS_EXPERIMENT_PATH)

    assert isinstance(configuration, ChessExperimentConfiguration)
    replay = configuration.training.lifecycle.replay
    assert replay.maximum_capacity == 2_500_000
    assert tuple(replay.capacity_at(generation) for generation in (0, 25, 50, 100, 250, 400)) == (
        300_000,
        600_000,
        1_000_000,
        1_500_000,
        2_000_000,
        2_500_000,
    )
    assert configuration.training.lifecycle.credit.replay_ratio == 10
    assert configuration.chess.self_play.search.parallel_searches == 4
    assert configuration.chess.self_play.inference.sdpa_backend is SdpaBackend.MEMORY_EFFICIENT
    assert configuration.training.trainer.precision is TrainingPrecision.BFLOAT16
    assert configuration.training.trainer.compilation is TrainingCompilation.DEFAULT
    progressive_model_sizing = configuration.training.progressive_model_sizing
    assert progressive_model_sizing is not None
    assert tuple(model.training_start_days for model in progressive_model_sizing.models) == (
        0.0,
        0.75,
        2.0,
    )
    expected_model_ids = ('chess-attention-500k', 'chess-attention-2m', 'chess-attention-4m5')
    assert tuple(model.model_id for model in progressive_model_sizing.models) == expected_model_ids
    assert tuple(
        (model.network.num_layers, model.network.embedding_size) for model in progressive_model_sizing.models
    ) == ((6, 96), (10, 160), (15, 192))
    assert configuration.training.network == progressive_model_sizing.models[0].network


@pytest.mark.parametrize('search_correction_count', (0, 2))
def test_adaptive_learned_gate_requires_exactly_one_search_correction_target(
    search_correction_count: int,
) -> None:
    candidate = yaml.safe_load(OPTIMAL_CHESS_EXPERIMENT_PATH.read_text(encoding='utf-8'))
    targets = candidate['chess']['objective']['auxiliary_targets']
    without_search_correction = tuple(target for target in targets if target['kind'] != 'search_correction')
    search_correction = next(target for target in targets if target['kind'] == 'search_correction')
    candidate['chess']['objective']['auxiliary_targets'] = [
        *without_search_correction,
        *[search_correction] * search_correction_count,
    ]

    with pytest.raises(ValidationError, match='requires exactly one search-correction'):
        ChessExperimentConfiguration.model_validate(candidate)


@pytest.mark.parametrize('coefficient', (0.0, -1.0, float('inf'), float('nan')))
def test_enabled_forced_playout_coefficient_must_be_positive_and_finite(coefficient: float) -> None:
    with pytest.raises(ValidationError):
        EnabledForcedPlayoutConfiguration.model_validate({'kind': 'enabled', 'coefficient': coefficient})


def test_game_experiments_extend_the_shared_run_and_training_configuration() -> None:
    assert set(BaseExperimentConfiguration.model_fields) == {'run', 'training', 'evaluation'}
    assert issubclass(ChessExperimentConfiguration, BaseExperimentConfiguration)
    assert issubclass(GoExperimentConfiguration, BaseExperimentConfiguration)


@pytest.mark.parametrize(
    ('configuration_type', 'path'),
    (
        (ChessExperimentConfiguration, CHESS_EXPERIMENT_TEMPLATE_PATH),
        (GoExperimentConfiguration, Path('test/configs/go-7x7-experiment.yaml')),
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


def test_experiment_owns_training_configuration_without_an_adapter() -> None:
    assert CHESS_TRAINING is CHESS_EXPERIMENT.training


def test_experiment_queue_validation_loads_multiple_experiments() -> None:
    configurations = validate_experiment_queue((CHESS_EXPERIMENT_TEMPLATE_PATH, CHESS_EXPERIMENT_TEMPLATE_PATH))

    assert len(configurations) == 2
    assert configurations[0] == configurations[1]


@pytest.mark.parametrize(
    ('path', 'board_size', 'action_count'),
    (
        (Path('test/configs/go-7x7-experiment.yaml'), 7, 50),
        (Path('test/configs/go-9x9-experiment.yaml'), 9, 82),
    ),
)
def test_go_experiments_resolve_deterministically(path: Path, board_size: int, action_count: int) -> None:
    configuration = load_experiment_configuration(path)

    assert isinstance(configuration, GoExperimentConfiguration)
    assert configuration.go.representation.board_size == board_size
    assert configuration.go.representation.action_count == action_count
    assert load_experiment_configuration(path) == configuration


@pytest.mark.parametrize(
    ('authored', 'expected'),
    (
        ({'kind': 'zero'}, ZeroFirstPlayUrgencyParameters()),
        ({'kind': 'parent_value'}, ParentValueFirstPlayUrgencyParameters()),
        (
            {'kind': 'reduced_parent_value', 'reduction': 0.2},
            ReducedParentValueFirstPlayUrgencyParameters(reduction=0.2),
        ),
    ),
)
def test_first_play_urgency_modes_resolve_explicitly(
    authored: JsonValue,
    expected: FirstPlayUrgencyParameters,
) -> None:
    candidate = yaml.safe_load(Path('test/configs/go-9x9-experiment.yaml').read_text(encoding='utf-8'))
    candidate['go']['self_play']['search']['first_play_urgency'] = authored
    configuration = GoExperimentConfiguration.model_validate(candidate)

    resolved = configuration.go.self_play.resolve(0, configuration.go.rules.maximum_moves, 1.0)

    assert resolved.first_play_urgency == expected


def test_queue_validation_supports_both_games() -> None:
    configurations = validate_experiment_queue(
        (
            CHESS_EXPERIMENT_TEMPLATE_PATH,
            Path('test/configs/go-7x7-experiment.yaml'),
            Path('test/configs/go-9x9-experiment.yaml'),
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
            1.5,
            'must remain in',
        ),
        (
            ('go', 'objective', 'value_discount_per_ply'),
            0.0,
            'must remain in',
        ),
        (
            ('go', 'self_play', 'search', 'first_play_urgency'),
            {'kind': 'reduced_parent_value', 'reduction': -0.1},
            'Reduced-parent FPU reduction must remain finite and positive',
        ),
        (('go', 'self_play', 'maximum_game_plies'), 200, 'Extra inputs are not permitted'),
    ),
)
def test_invalid_go_combinations_fail_precisely(field_path: tuple[str, ...], value: JsonValue, message: str) -> None:
    candidate = yaml.safe_load(Path('test/configs/go-7x7-experiment.yaml').read_text(encoding='utf-8'))
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


def test_experiment_configuration_extends_and_deeply_overrides_a_base(tmp_path: Path) -> None:
    base_path = tmp_path / 'base.yaml'
    base_path.write_text(Path('test/configs/go-7x7-experiment.yaml').read_text(encoding='utf-8'), encoding='utf-8')
    override_path = tmp_path / 'override.yaml'
    override_path.write_text(
        '\n'.join(
            (
                'extends: base.yaml',
                'run:',
                '  run_name: inherited-run',
                'training:',
                '  save_path: inherited-output',
                '  trainer:',
                '    learning_rate: {kind: linear, start_generation: 0, end_generation: 10, start_value: 0.004, end_value: 0.001, rounding: none}',
                '  topology:',
                '    self_play:',
                '      node_ids_to_pause_during_training: [0]',
            )
        )
        + '\n',
        encoding='utf-8',
    )

    configuration = load_experiment_configuration(override_path)

    assert isinstance(configuration, GoExperimentConfiguration)
    assert configuration.run.run_name == 'inherited-run'
    assert configuration.training.save_path == 'inherited-output'
    assert configuration.training.trainer.learning_rate.value_at(0) == pytest.approx(0.004)
    assert configuration.training.topology.self_play.node_ids_to_pause_during_training == (0,)
    assert configuration.go.representation.board_size == 7


def test_experiment_configuration_hash_includes_resolved_base_content(tmp_path: Path) -> None:
    base_path = tmp_path / 'base.yaml'
    source = Path('test/configs/go-7x7-experiment.yaml').read_text(encoding='utf-8')
    base_path.write_text(source, encoding='utf-8')
    override_path = tmp_path / 'override.yaml'
    override_path.write_text('extends: base.yaml\n', encoding='utf-8')
    original = experiment_configuration_sha256(load_experiment_configuration(override_path))
    base_path.write_text(source.replace('learning_rate: 0.002', 'learning_rate: 0.004', 1), encoding='utf-8')

    changed = experiment_configuration_sha256(load_experiment_configuration(override_path))

    assert changed != original


def test_experiment_configuration_rejects_inheritance_cycles(tmp_path: Path) -> None:
    first = tmp_path / 'first.yaml'
    second = tmp_path / 'second.yaml'
    first.write_text('extends: second.yaml\n', encoding='utf-8')
    second.write_text('extends: first.yaml\n', encoding='utf-8')

    with pytest.raises(ValueError, match='inheritance contains a cycle'):
        load_experiment_configuration(first)


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


def test_chess_maximum_ply_syzygy_paths_are_optional_and_typed() -> None:
    candidate = yaml.safe_load(CHESS_EXPERIMENT_TEMPLATE_PATH.read_text(encoding='utf-8'))
    candidate['chess']['self_play']['maximum_ply_syzygy_paths'] = ['/tablebases/3-5', '/tablebases/6-7']

    configuration = ChessExperimentConfiguration.model_validate(candidate)

    assert configuration.chess.self_play.maximum_ply_syzygy_paths == (
        '/tablebases/3-5',
        '/tablebases/6-7',
    )


def test_chess_maximum_ply_syzygy_paths_reject_empty_entries() -> None:
    candidate = yaml.safe_load(CHESS_EXPERIMENT_TEMPLATE_PATH.read_text(encoding='utf-8'))
    candidate['chess']['self_play']['maximum_ply_syzygy_paths'] = ['']

    with pytest.raises(ValidationError, match='nonempty directory path'):
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
