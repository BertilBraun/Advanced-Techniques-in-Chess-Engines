from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.experiment.base_configuration import BaseExperimentConfiguration
from src.experiment.configuration import (
    experiment_configuration_sha256,
    experiment_configuration_source_paths,
    load_experiment_configuration,
    load_chess_experiment_configuration,
    validate_experiment_queue,
    write_resolved_chess_experiment,
)
from src.games.chess.configuration import ChessExperimentConfiguration
from src.games.chess.training import ChessImplementation
from src.games.go.configuration import GoExperimentConfiguration
from src.self_play.configuration import EnabledForcedPlayoutConfiguration
from src.self_play.parameters import (
    FirstPlayUrgencyParameters,
    ParentValueFirstPlayUrgencyParameters,
    ReducedParentValueFirstPlayUrgencyParameters,
    ZeroFirstPlayUrgencyParameters,
)
from src.training.network import (
    GlobalPoolingResidualContext,
    ResidualContextPlacement,
    SqueezeExcitationResidualContext,
)
from test_helpers.chess_configuration import CHESS_EXPERIMENT, CHESS_TRAINING
from src.util.frozen_model import JsonValue


CHESS_EXPERIMENT_TEMPLATE_PATH = Path('configs/chess-experiment-template.yaml')
CHESS_R3_EXPERIMENT_PATH = Path('configs/production/vast-chess-8gpu-1d-r3.yaml')
CHESS_R4_EXPERIMENT_PATH = Path('configs/production/vast-chess-8gpu-1d-r4.yaml')
CHESS_OPTIMAL_EXPERIMENT_PATH = Path('configs/production/vast-chess-8gpu-optimal.yaml')


def test_every_checked_in_experiment_uses_the_current_contract_and_dependency_lock() -> None:
    paths = tuple(sorted(Path('configs').glob('*-experiment-template.yaml')))
    configurations = validate_experiment_queue(paths)
    dependency_lock_sha256 = hashlib.sha256(Path('requirements-training.lock').read_bytes()).hexdigest()

    assert tuple(configuration.game for configuration in configurations) == ('chess', 'go', 'go')
    assert all(
        configuration.run.environment.dependency_lock_sha256 == dependency_lock_sha256
        for configuration in configurations
    )


def test_every_screening_configuration_parses() -> None:
    paths = tuple(sorted(Path('configs/screening').rglob('*.yaml')))

    assert all(load_experiment_configuration(path) for path in paths)


def test_global_pooling_screen_replaces_baseline_squeeze_excitation() -> None:
    baseline = load_experiment_configuration(Path('configs/screening/go-9x9-model-size/00-medium-8x96.yaml'))
    global_pooling = load_experiment_configuration(
        Path('configs/screening/go-9x9-model-size/03-global-pooling-8x96.yaml')
    )

    assert baseline.training.network.residual_context == SqueezeExcitationResidualContext(
        placement=ResidualContextPlacement.EVERY_SECOND_BLOCK
    )
    assert global_pooling.training.network.residual_context == GlobalPoolingResidualContext(
        placement=ResidualContextPlacement.EVERY_SECOND_BLOCK
    )
    assert (
        global_pooling.training.network.model_copy(
            update={'residual_context': baseline.training.network.residual_context}
        )
        == baseline.training.network
    )


def test_go9_strong_baseline_resolves_the_authored_search_progression() -> None:
    configuration = load_experiment_configuration(Path('configs/screening/go-9x9-strong/00-baseline.yaml'))

    assert configuration.evaluation.cadence_seconds == 1200
    assert configuration.evaluation.job_timeout_seconds == 1800
    match_definitions = tuple(
        definition for definition in configuration.evaluation.definitions if definition.kind != 'fixed_dataset'
    )
    assert all(definition.search.searches_per_move == 64 for definition in match_definitions)
    assert tuple(definition.opening_pair_count for definition in match_definitions) == (100, 100, 100, 100, 50)
    katago_definition = match_definitions[-1]
    assert katago_definition.kind == 'katago'
    assert katago_definition.definition_id == 'katago-64'
    assert katago_definition.maximum_visits == 64
    assert isinstance(configuration.training.network.residual_context, GlobalPoolingResidualContext)
    assert configuration.training.trainer.learning_rate.value_at(0) == pytest.approx(0.01)
    assert configuration.training.trainer.learning_rate.value_at(150) == pytest.approx(0.001)
    initial = configuration.go.self_play.resolve(0, configuration.go.rules.maximum_moves, 1.0)
    mixed = configuration.go.self_play.resolve(25, configuration.go.rules.maximum_moves, 1.0)
    mature = configuration.go.self_play.resolve(50, configuration.go.rules.maximum_moves, 1.0)
    assert (initial.full_searches, initial.fast_searches, initial.full_search_probability) == (64, 16, 1.0)
    assert (mixed.full_searches, mixed.fast_searches, mixed.full_search_probability) == (160, 40, 0.25)
    assert (mature.full_searches, mature.fast_searches, mature.full_search_probability) == (256, 64, 0.25)
    assert isinstance(initial.first_play_urgency, ReducedParentValueFirstPlayUrgencyParameters)
    assert initial.first_play_urgency.reduction == pytest.approx(0.2)
    assert initial.forced_playout_coefficient == pytest.approx(2.0)
    assert initial.retained_root_visit_fraction == pytest.approx(0.6)
    assert configuration.go.objective.root_value_blend.value_at(29) == pytest.approx(0.0)
    assert configuration.go.objective.root_value_blend.value_at(75) == pytest.approx(0.15)
    assert tuple(target.kind for target in configuration.go.objective.auxiliary_targets) == (
        'next_policy',
        'remaining_game_length',
    )


def test_forced_playout_screen_resolves_canonical_coefficient() -> None:
    configuration = load_experiment_configuration(Path('configs/screening/go-7x7-overnight/15-forced-playouts.yaml'))

    assert configuration.game == 'go'
    assert configuration.go.self_play.resolve(
        0, configuration.go.rules.maximum_moves, 1.0
    ).forced_playout_coefficient == pytest.approx(2.0)


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
    assert configuration.training.trainer.global_batch_size == 2048
    assert configuration.training.trainer.local_batch_size == 2048
    assert configuration.training.trainer.learning_rate.value_at(199) == pytest.approx(0.005)
    assert configuration.training.trainer.learning_rate.value_at(200) == pytest.approx(0.0035)
    assert configuration.training.trainer.learning_rate.value_at(500) == pytest.approx(0.002)
    assert configuration.training.trainer.duplicate_multiplicity_weight_cap is None
    assert configuration.training.topology.self_play.device_ids == (0, 0, 0, 0)
    assert configuration.training.lifecycle.credit.retained_checkpoint_interval_generations == 3
    assert configuration.training.limits.maximum_wall_time_seconds == 2 * 24 * 60 * 60
    assert configuration.run.environment.minimum_open_file_soft_limit == 65_536
    assert configuration.chess.self_play.search.full_searches.value_at(0) == 300
    assert configuration.chess.self_play.search.full_searches.value_at(20) == 600
    assert configuration.chess.self_play.search.fast_searches.value_at(0) == 75
    assert configuration.chess.self_play.search.fast_searches.value_at(20) == 150
    assert configuration.chess.self_play.search.parallel_searches == 1
    assert configuration.chess.self_play.search.forced_playouts.kind == 'disabled'
    assert configuration.chess.self_play.search.first_play_urgency.kind == 'zero'
    assert configuration.chess.self_play.inference.inference_workers == 2
    assert configuration.chess.self_play.inference.inference_batch_size == 64
    assert configuration.chess.self_play.inference.outstanding_batches_per_worker == 2
    assert configuration.evaluation.cadence_seconds == 1200
    assert configuration.evaluation.dataset.path.endswith('chess-evaluation-v1.bin')
    assert configuration.evaluation.openings.opening_count == 50
    assert configuration.evaluation.engine.kind == 'stockfish'
    assert configuration.evaluation.engine.hash_mib == 1024
    assert configuration.chess.self_play.maximum_game_plies is not None
    assert configuration.chess.self_play.maximum_game_plies.value_at(0) == 200
    assert configuration.chess.self_play.maximum_game_plies.value_at(25) == 400
    assert configuration.chess.self_play.maximum_game_plies.value_at(50) == 600
    assert configuration.chess.self_play.force_fast_search_after_ply is not None
    assert configuration.chess.self_play.force_fast_search_after_ply.value_at(0) == 250
    start_position = configuration.chess.self_play.start_position
    assert start_position.kind == 'restart_state'
    assert start_position.maximum_absolute_root_value == pytest.approx(0.8)
    assert configuration.chess.objective.root_value_blend.value_at(9) == pytest.approx(0.0)
    assert configuration.chess.objective.root_value_blend.value_at(20) == pytest.approx(0.075)
    assert configuration.chess.objective.root_value_blend.value_at(30) == pytest.approx(0.15)
    assert tuple(definition.kind for definition in configuration.evaluation.definitions) == (
        'fixed_dataset',
        'random',
        'policy_random',
        *('previous_checkpoint',) * 3,
        *('stockfish',) * 4,
    )


def test_experiment_owns_training_configuration_without_an_adapter() -> None:
    assert CHESS_TRAINING is CHESS_EXPERIMENT.training


def test_eight_gpu_chess_production_configuration_resolves_authored_curriculum() -> None:
    configuration = load_chess_experiment_configuration(Path('configs/production/vast-chess-8gpu-1d.yaml'))
    trainer = configuration.training.trainer
    topology = configuration.training.topology
    credit = configuration.training.lifecycle.credit
    self_play = configuration.chess.self_play

    assert topology.trainer.ddp_device_ids == (0, 1, 2, 3)
    assert trainer.global_batch_size == 2048
    assert trainer.local_batch_size == 512
    assert len(topology.self_play.device_ids) == 24
    assert topology.self_play.node_ids_to_pause_during_training == (1, 2, 4, 5, 7, 8, 10, 11)
    assert credit.replay_ratio == 8
    assert credit.optimizer_steps_per_quantum == 400
    assert self_play.search.forced_playouts.kind == 'enabled'
    assert self_play.search.forced_playouts.coefficient == pytest.approx(1.5)
    assert self_play.greedy_after_ply.value_at(0) == 200
    assert tuple(
        self_play.search.full_searches.value_at(generation) for generation in (0, 5, 10, 20, 35, 135, 235)
    ) == (
        200,
        300,
        400,
        500,
        600,
        700,
        800,
    )
    assert tuple(self_play.search.fast_searches.value_at(generation) for generation in (0, 5, 10, 20, 35, 235)) == (
        50,
        75,
        100,
        125,
        150,
        150,
    )
    assert configuration.evaluation.openings.source.kind == 'chess_book'
    assert sum(definition.kind == 'stockfish_fixed_nodes' for definition in configuration.evaluation.definitions) == 1


def test_eight_gpu_chess_r3_configuration_resolves_game_length_curriculum() -> None:
    configuration = load_chess_experiment_configuration(CHESS_R3_EXPERIMENT_PATH)
    topology = configuration.training.topology
    lifecycle = configuration.training.lifecycle
    self_play = configuration.chess.self_play

    assert topology.self_play.device_ids == tuple(device_id for device_id in range(8) for _ in range(2))
    assert topology.self_play.parallel_games_per_process == 512
    assert topology.self_play.node_ids_to_pause_during_training == tuple(
        worker_id for worker_id in range(16) if worker_id % 2 != 0
    )
    assert lifecycle.credit.optimizer_steps_per_quantum == 500
    assert lifecycle.credit.maximum_optimizer_steps is None
    assert lifecycle.credit.unique_samples_per_quantum(configuration.training.trainer.global_batch_size) == 128_000
    assert lifecycle.credit.retained_checkpoint_interval_generations == 6
    assert configuration.training.limits.maximum_cost is None
    assert configuration.training.limits.maximum_wall_time_seconds is None
    assert lifecycle.inference_retention.recent_checkpoint_count == 22
    assert lifecycle.inference_retention.milestone_interval == 20
    assert lifecycle.replay.capacity.value_at(0) == 300_000
    assert tuple(
        self_play.search.full_searches.value_at(generation) for generation in (0, 10, 30, 50, 90, 360, 630)
    ) == (200, 300, 400, 500, 600, 700, 800)
    assert tuple(self_play.search.fast_searches.value_at(generation) for generation in (0, 10, 30, 50, 90)) == (
        50,
        75,
        100,
        125,
        150,
    )
    assert self_play.start_position.maximum_age_generations == 40
    assert self_play.full_search_probability.value_at(70) == pytest.approx(0.25)
    assert tuple(self_play.greedy_after_ply.value_at(generation) for generation in (0, 109, 110, 199, 200, 500)) == (
        60,
        60,
        80,
        80,
        100,
        100,
    )
    learning_rate = configuration.training.trainer.learning_rate
    assert tuple(learning_rate.value_at(generation) for generation in (0, 99, 100, 299, 300, 500)) == pytest.approx(
        (0.005, 0.005, 0.0035, 0.0035, 0.002, 0.002)
    )
    replay = configuration.training.lifecycle.replay
    assert tuple(replay.capacity_at(generation) for generation in (0, 50, 100, 500)) == (
        300_000,
        1_150_000,
        2_000_000,
        2_000_000,
    )
    assert replay.maximum_capacity == 2_000_000
    resignation = self_play.resignation
    assert resignation.kind == 'calibrated'
    assert resignation.first_production_generation == 70
    assert resignation.false_nonloss_rate_ceiling == pytest.approx(0.03)
    assert resignation.continuation_game_probability == pytest.approx(0.10)
    assert resignation.triggered_game_window == 2000
    assert resignation.candidate_thresholds == pytest.approx(tuple(-0.99 + index * 0.01 for index in range(30)))
    assert resignation.minimum_evidence_trigger_count == 100
    assert resignation.confidence_level == pytest.approx(0.95)
    assert resignation.maximum_relaxation_per_generation == pytest.approx(0.01)
    assert self_play.maximum_game_plies is not None
    assert tuple(
        self_play.maximum_game_plies.value_at(generation)
        for generation in (0, 49, 50, 80, 110, 140, 160, 200, 240, 500)
    ) == (150, 150, 160, 180, 200, 250, 300, 350, 400, 400)
    assert self_play.force_fast_search_after_ply is not None
    assert self_play.force_fast_search_after_ply.value_at(0) == 200
    assert configuration.chess.objective.root_value_blend.value_at(50) == pytest.approx(0.0)
    assert configuration.chess.objective.root_value_blend.value_at(110) == pytest.approx(0.15)


def test_eight_gpu_chess_r3_configuration_is_fully_self_declared() -> None:
    configuration = load_chess_experiment_configuration(CHESS_R3_EXPERIMENT_PATH)

    assert experiment_configuration_source_paths(CHESS_R3_EXPERIMENT_PATH) == (CHESS_R3_EXPERIMENT_PATH.resolve(),)
    assert configuration.model_dump(exclude_unset=True) == configuration.model_dump()


def test_eight_gpu_chess_r4_configuration_resolves_checkpoint_continuation() -> None:
    configuration = load_chess_experiment_configuration(CHESS_R4_EXPERIMENT_PATH)

    assert experiment_configuration_source_paths(CHESS_R4_EXPERIMENT_PATH) == (
        CHESS_R3_EXPERIMENT_PATH.resolve(),
        CHESS_R4_EXPERIMENT_PATH.resolve(),
    )
    assert configuration.run.run_name == 'vast-chess-8gpu-1d-r4'
    assert configuration.run.stage.value == 'continuation'
    assert configuration.run.hardware.minimum_disk_gib == pytest.approx(10)
    assert configuration.run.resume.mode == 'checkpoint'
    assert configuration.run.resume.generation == 150
    assert configuration.training.save_path.endswith('vast-chess-8gpu-1d-r4')
    assert configuration.training.trainer.learning_rate.value_at(150) == pytest.approx(0.004)
    assert configuration.training.trainer.learning_rate.value_at(349) == pytest.approx(0.004)
    assert configuration.training.trainer.learning_rate.value_at(350) == pytest.approx(0.003)
    assert configuration.training.lifecycle.replay.capacity_at(150) == 1_500_000
    assert configuration.training.lifecycle.replay.maximum_capacity == 2_500_000
    assert configuration.training.lifecycle.credit.replay_ratio == 10
    assert configuration.training.limits.minimum_free_disk_gib == pytest.approx(10.0)
    self_play = configuration.chess.self_play
    assert self_play.resignation.false_nonloss_rate_ceiling == pytest.approx(0.025)
    assert self_play.start_position.minimum_remaining_plies == 25
    assert tuple(self_play.search.full_searches.value_at(generation) for generation in (150, 180, 250)) == (
        600,
        700,
        800,
    )
    assert self_play.search.fast_searches.value_at(150) == 150
    assert self_play.full_search_probability.value_at(150) == pytest.approx(0.25)
    assert self_play.greedy_after_ply.value_at(150) == 80
    assert self_play.maximum_game_plies is not None
    assert self_play.maximum_game_plies.value_at(150) == 200
    assert self_play.force_fast_search_after_ply is not None
    assert self_play.force_fast_search_after_ply.value_at(150) == 160
    assert configuration.chess.objective.root_value_blend.value_at(150) == pytest.approx(0.10)
    assert configuration.chess.objective.value_discount_per_ply.value_at(299) == pytest.approx(0.9985)
    assert configuration.chess.objective.value_discount_per_ply.value_at(300) == pytest.approx(0.9960)
    assert ChessImplementation(configuration).self_play_parameters_at(150).value_discount_per_ply == pytest.approx(
        0.9985
    )
    assert ChessImplementation(configuration).self_play_parameters_at(300).value_discount_per_ply == pytest.approx(
        0.9960
    )
    remaining_length = configuration.chess.objective.auxiliary_targets[1]
    assert remaining_length.kind == 'remaining_game_length'
    assert remaining_length.normalization_scale == pytest.approx(400.0)


def test_eight_gpu_chess_optimal_configuration_resolves_distilled_curriculum() -> None:
    configuration = load_chess_experiment_configuration(CHESS_OPTIMAL_EXPERIMENT_PATH)
    training = configuration.training
    self_play = configuration.chess.self_play
    objective = configuration.chess.objective

    assert experiment_configuration_source_paths(CHESS_OPTIMAL_EXPERIMENT_PATH) == (
        CHESS_OPTIMAL_EXPERIMENT_PATH.resolve(),
    )
    assert configuration.run.stage.value == 'clean_retrain'
    assert configuration.run.resume.mode == 'random_initialization'
    assert training.limits.minimum_free_disk_gib == pytest.approx(10.0)
    assert training.lifecycle.replay.maximum_capacity == 1_500_000
    assert tuple(training.lifecycle.replay.capacity_at(generation) for generation in (0, 50, 100, 500)) == (
        300_000,
        900_000,
        1_500_000,
        1_500_000,
    )
    assert training.lifecycle.credit.replay_ratio == 8
    assert tuple(
        training.trainer.learning_rate.value_at(generation) for generation in (0, 100, 350, 550)
    ) == pytest.approx((0.005, 0.004, 0.003, 0.002))
    assert tuple(self_play.search.full_searches.value_at(generation) for generation in (0, 30, 90, 250, 550)) == (
        200,
        400,
        600,
        800,
        1000,
    )
    assert tuple(self_play.search.fast_searches.value_at(generation) for generation in (0, 30, 90)) == (
        50,
        100,
        150,
    )
    assert self_play.maximum_game_plies is not None
    assert tuple(self_play.maximum_game_plies.value_at(generation) for generation in (0, 50, 100, 500)) == (
        150,
        180,
        200,
        200,
    )
    assert self_play.force_fast_search_after_ply is not None
    assert self_play.force_fast_search_after_ply.value_at(100) == 160
    assert self_play.resignation.false_nonloss_rate_ceiling == pytest.approx(0.025)
    assert objective.root_value_blend.value_at(110) == pytest.approx(0.10)
    assert objective.value_discount_per_ply.value_at(299) == pytest.approx(0.9985)
    assert objective.value_discount_per_ply.value_at(300) == pytest.approx(0.996)


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


@pytest.mark.parametrize(
    ('authored', 'expected'),
    (
        ({'kind': 'zero'}, ZeroFirstPlayUrgencyParameters()),
        ({'kind': 'parent_value'}, ParentValueFirstPlayUrgencyParameters()),
        (
            {'kind': 'reduced_parent_value', 'reduction': {'kind': 'constant', 'value': 0.2}},
            ReducedParentValueFirstPlayUrgencyParameters(reduction=0.2),
        ),
    ),
)
def test_first_play_urgency_modes_resolve_explicitly(
    authored: JsonValue,
    expected: FirstPlayUrgencyParameters,
) -> None:
    candidate = yaml.safe_load(Path('configs/go-9x9-experiment-template.yaml').read_text(encoding='utf-8'))
    candidate['go']['self_play']['search']['first_play_urgency'] = authored
    configuration = GoExperimentConfiguration.model_validate(candidate)

    resolved = configuration.go.self_play.resolve(0, configuration.go.rules.maximum_moves, 1.0)

    assert resolved.first_play_urgency == expected


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
        (
            ('go', 'objective', 'value_discount_per_ply'),
            {'kind': 'constant', 'value': 0.0},
            'must remain in',
        ),
        (
            ('go', 'self_play', 'search', 'first_play_urgency'),
            {'kind': 'reduced_parent_value', 'reduction': {'kind': 'constant', 'value': -0.1}},
            'Reduced-parent FPU reduction must remain finite and positive',
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


def test_experiment_configuration_extends_and_deeply_overrides_a_base(tmp_path: Path) -> None:
    base_path = tmp_path / 'base.yaml'
    base_path.write_text(Path('configs/go-7x7-experiment-template.yaml').read_text(encoding='utf-8'), encoding='utf-8')
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
    source = Path('configs/go-7x7-experiment-template.yaml').read_text(encoding='utf-8')
    base_path.write_text(source, encoding='utf-8')
    override_path = tmp_path / 'override.yaml'
    override_path.write_text('extends: base.yaml\n', encoding='utf-8')
    original = experiment_configuration_sha256(load_experiment_configuration(override_path))
    base_path.write_text(source.replace('value: 0.002', 'value: 0.004', 1), encoding='utf-8')

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
