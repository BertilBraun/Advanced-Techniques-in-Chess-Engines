from __future__ import annotations

import pickle
from pathlib import Path
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.experiment.cost_accounting import CostCurrency
from src.experiment.run_configuration import (
    BudgetConfiguration,
    LearningRateStage,
    PiecewiseLearningRate,
    ApprovalRecord,
    ResolvedHardware,
    RunConfiguration,
    RunManifest,
    apply_run_configuration,
    configuration_sha256,
    load_run_configuration,
    validate_run_configuration,
    validate_approval,
    write_run_manifest,
)
from src.train.TrainingArgs import (
    ArtifactRetention,
    ClusterParams,
    EvaluationParams,
    MCTSParams,
    NetworkParams,
    OptimizerType,
    RuntimeLimits,
    SelfPlayParams,
    TrainingArgs,
    TrainingParams,
)


CONFIGURATION_PATH = Path('configs/chess-continuation-4x4070-pilot.json')
SCALING_CONFIGURATION_PATH = Path('configs/chess-continuation-4x4070-scaling-pilot.json')
SHUTDOWN_CONFIGURATION_PATH = Path('configs/chess-continuation-4x4070-shutdown-smoke.json')
TUNING_CONFIGURATION_PATHS = tuple(Path(f'configs/chess-clean-tuning-{variant}.json') for variant in ('a', 'b', 'c'))
MAIN_CONFIGURATION_PATH = Path('configs/chess-clean-4x4070-main.json')
V4_CONFIGURATION_PATH = Path('configs/chess-clean-4x4070-v4.json')
V5_CONFIGURATION_PATH = Path('configs/chess-clean-4x4070-v5.json')


def sampling_window(_: int) -> int:
    return 1


def learning_rate(_: int, __: OptimizerType) -> float:
    return 0.01


def learning_rate_scheduler(_: float, learning_rate_value: float) -> float:
    return learning_rate_value


def training_args() -> TrainingArgs:
    return TrainingArgs(
        save_path='unused',
        num_iterations=1,
        num_games_per_iteration=1,
        network=NetworkParams(num_layers=1, hidden_size=8),
        self_play=SelfPlayParams(
            mcts=MCTSParams(
                num_searches_per_turn=8,
                fast_searches_proportion_of_full_searches=0.5,
                playout_cap_randomization=0.25,
                num_parallel_searches=1,
                dirichlet_epsilon=0.25,
                dirichlet_alpha=0.3,
                c_param=1.5,
                num_threads=1,
                percentage_of_node_visits_to_keep=0.5,
            ),
            num_parallel_games=1,
            inference_cache_capacity=1,
            use_inference_cache=True,
            num_moves_after_which_to_play_greedy=10,
            portion_of_samples_to_keep=0.5,
        ),
        training=TrainingParams(
            num_epochs=1,
            global_batch_size=8,
            local_batch_size=8,
            optimizer='adamw',
            sampling_window=sampling_window,
            learning_rate=learning_rate,
            learning_rate_scheduler=learning_rate_scheduler,
            num_workers=0,
        ),
        cluster=ClusterParams(
            trainer_device_type='cpu',
            trainer_process_group_backend='gloo',
            trainer_rank_zero_device_id=0,
            trainer_ddp_device_ids=(0,),
            evaluation_device_cycle=(0,),
            self_play_device_ids=(0,),
            self_play_tensorboard_processes=1,
            trainer_cpu_threads=1,
            trainer_interop_threads=1,
            self_play_node_ids_to_pause_during_training=(),
            max_concurrent_evaluations=1,
        ),
        run_limits=RuntimeLimits(
            cost_currency=CostCurrency.EUR,
            hourly_price=1.0,
            maximum_cost=1.0,
            maximum_wall_time_seconds=3600,
            maximum_open_file_count=1024,
            maximum_host_ram_percent=90,
            minimum_free_disk_gib=1,
        ),
        artifact_retention=ArtifactRetention(
            checkpoint_count=5,
            replay_window_iterations=30,
            recent_inference_checkpoint_count=11,
            milestone_inference_interval=10,
        ),
        random_seed=0,
        self_play_search_warmup_iterations=1,
        self_play_value_warmup_iterations=1,
        evaluation=EvaluationParams(
            num_searches_per_turn=8,
            num_games=2,
            every_n_iterations=1,
            evaluate_initial_checkpoint=False,
            max_concurrent_tasks=1,
            inference_cache_capacity=1,
            use_inference_cache=True,
            dataset_path=None,
            reference_model_path=None,
            opening_suite_path=None,
            raw_results_path=None,
            maximum_game_plies=200,
            bootstrap_seed=0,
            bootstrap_samples=100,
            mcts_threads=1,
            previous_model_offsets=(),
            historical_model_iterations=(),
            historical_model_rotation_period=1,
            stockfish_skill_levels=(),
            stockfish_binary_path=None,
            stockfish_nodes_per_move=1_000,
            stockfish_threads=1,
            stockfish_hash_mib=128,
            evaluate_random=False,
        ),
    )


def resolved_pilot_hardware() -> ResolvedHardware:
    return ResolvedHardware(
        visible_gpu_names=('NVIDIA GeForce RTX 4070 SUPER',) * 4,
        visible_gpu_count=4,
        logical_cpu_count=64,
        total_ram_gib=125.7,
        free_disk_gib=99.9,
    )


@pytest.mark.parametrize(
    'configuration_path',
    (
        CONFIGURATION_PATH,
        SCALING_CONFIGURATION_PATH,
        SHUTDOWN_CONFIGURATION_PATH,
        *TUNING_CONFIGURATION_PATHS,
        MAIN_CONFIGURATION_PATH,
        V4_CONFIGURATION_PATH,
        V5_CONFIGURATION_PATH,
    ),
)
def test_pilot_configuration_is_valid_for_quoted_hardware(configuration_path: Path) -> None:
    configuration = load_run_configuration(configuration_path)

    validate_run_configuration(configuration, resolved_pilot_hardware())


def test_configuration_rejects_cpu_oversubscription() -> None:
    configuration = load_run_configuration(CONFIGURATION_PATH)
    data = configuration.model_dump()
    data['topology']['trainer_cpu_threads'] = 32
    oversubscribed = RunConfiguration.model_validate(data)

    with pytest.raises(ValueError, match='logical CPU slots'):
        validate_run_configuration(oversubscribed, resolved_pilot_hardware())


@pytest.mark.parametrize(
    ('device_ids', 'rank_zero_device_id', 'message'),
    (
        ((), 0, 'At least one trainer device'),
        ((3, 3), 3, 'must be unique'),
        ((2, 3), 3, 'rank-zero trainer device'),
        ((-1,), 0, 'cannot be negative'),
    ),
)
def test_configuration_rejects_invalid_ddp_device_lists(
    device_ids: tuple[int, ...],
    rank_zero_device_id: int,
    message: str,
) -> None:
    configuration = load_run_configuration(CONFIGURATION_PATH)
    data = configuration.model_dump()
    data['topology']['trainer_ddp_device_ids'] = device_ids
    data['topology']['trainer_rank_zero_device_id'] = rank_zero_device_id

    with pytest.raises(ValidationError, match=message):
        RunConfiguration.model_validate(data)


def test_configuration_rejects_out_of_range_ddp_device() -> None:
    configuration = load_run_configuration(CONFIGURATION_PATH)
    data = configuration.model_dump()
    data['topology']['trainer_rank_zero_device_id'] = 4
    data['topology']['trainer_ddp_device_ids'] = (4,)
    out_of_range = RunConfiguration.model_validate(data)

    with pytest.raises(ValueError, match='outside the configured GPU range'):
        validate_run_configuration(out_of_range, resolved_pilot_hardware())


def test_configuration_accepts_explicit_cpu_gloo_trainer() -> None:
    configuration = load_run_configuration(CONFIGURATION_PATH)
    data = configuration.model_dump()
    data['topology']['trainer_device_type'] = 'cpu'
    data['topology']['trainer_process_group_backend'] = 'gloo'
    data['topology']['trainer_rank_zero_device_id'] = 0
    data['topology']['trainer_ddp_device_ids'] = (0,)

    cpu_configuration = RunConfiguration.model_validate(data)

    validate_run_configuration(cpu_configuration, resolved_pilot_hardware())


def test_configuration_rejects_nccl_for_cpu_trainer() -> None:
    configuration = load_run_configuration(CONFIGURATION_PATH)
    data = configuration.model_dump()
    data['topology']['trainer_device_type'] = 'cpu'
    data['topology']['trainer_rank_zero_device_id'] = 0
    data['topology']['trainer_ddp_device_ids'] = (0,)

    with pytest.raises(ValidationError, match='NCCL can only'):
        RunConfiguration.model_validate(data)


def test_configuration_rejects_global_local_batch_mismatch() -> None:
    configuration = load_run_configuration(V4_CONFIGURATION_PATH)
    data = configuration.model_dump()
    data['workload']['training_local_batch_size'] = 1024

    with pytest.raises(ValidationError, match='Global training batch size'):
        RunConfiguration.model_validate(data)


def test_configuration_rejects_legacy_data_parallel_fields() -> None:
    configuration = load_run_configuration(CONFIGURATION_PATH)
    data = configuration.model_dump()
    data['topology']['trainer_device_id'] = data['topology'].pop('trainer_rank_zero_device_id')
    data['topology']['trainer_data_parallel_device_ids'] = data['topology'].pop('trainer_ddp_device_ids')

    with pytest.raises(ValidationError, match='trainer_device_id|trainer_data_parallel_device_ids'):
        RunConfiguration.model_validate(data)


@pytest.mark.parametrize(
    'training_process_counts',
    (
        (6, 6, 6),
        (6, 6, 6, 3, 0),
        (7, 6, 6, 2),
    ),
)
def test_configuration_rejects_invalid_training_self_play_counts(
    training_process_counts: tuple[int, ...],
) -> None:
    configuration = load_run_configuration(CONFIGURATION_PATH)
    data = configuration.model_dump()
    data['topology']['self_play_processes_per_device_during_training'] = training_process_counts

    with pytest.raises(ValidationError, match='Training self-play process counts'):
        RunConfiguration.model_validate(data)


def test_budget_rejects_cost_above_ceiling() -> None:
    with pytest.raises(ValidationError, match='Projected cost'):
        BudgetConfiguration(
            currency=CostCurrency.EUR,
            hourly_price=1.5,
            maximum_cost=1.0,
            maximum_wall_time_minutes=120,
        )


def test_budget_allows_elapsed_time_only_accounting() -> None:
    budget = BudgetConfiguration(
        currency=CostCurrency.USD,
        hourly_price=0.40,
        maximum_cost=None,
        maximum_wall_time_minutes=12 * 60,
    )

    assert budget.maximum_cost is None


def test_clean_retrain_uses_random_initialization_without_a_model_path() -> None:
    configuration = load_run_configuration(CONFIGURATION_PATH)
    data = configuration.model_dump()
    data['resume'] = {
        'mode': 'random_initialization',
        'optimizer': 'adamw',
    }

    clean_retrain = RunConfiguration.model_validate(data)

    assert clean_retrain.resume.mode.value == 'random_initialization'


def test_piecewise_learning_rate_uses_latest_started_stage() -> None:
    schedule = PiecewiseLearningRate(
        (
            LearningRateStage(start_iteration=0, learning_rate=0.005),
            LearningRateStage(start_iteration=60, learning_rate=0.002),
        )
    )

    assert schedule(59, 'adamw') == pytest.approx(0.005)
    assert schedule(60, 'adamw') == pytest.approx(0.002)
    assert pickle.dumps(schedule)


def test_run_configuration_applies_explicit_topology_and_workload() -> None:
    configuration = load_run_configuration(CONFIGURATION_PATH)
    arguments = training_args()

    apply_run_configuration(arguments, configuration)

    assert arguments.cluster.trainer_device_type == 'cuda'
    assert arguments.cluster.trainer_process_group_backend == 'nccl'
    assert arguments.cluster.trainer_rank_zero_device_id == 3
    assert arguments.cluster.trainer_ddp_device_ids == (3,)
    assert arguments.cluster.evaluation_device_cycle == (3,)
    assert arguments.cluster.self_play_device_ids == (0,) * 6 + (1,) * 6 + (2,) * 6 + (3,) * 2
    assert arguments.cluster.self_play_tensorboard_processes == 1
    assert arguments.cluster.trainer_cpu_threads == 8
    assert arguments.cluster.self_play_node_ids_to_pause_during_training == tuple(range(20))
    assert arguments.self_play.mcts.num_threads == 2
    assert arguments.self_play.num_parallel_games == 64
    assert arguments.self_play.inference_cache_capacity == 250_000
    assert arguments.self_play.use_inference_cache
    assert arguments.self_play.mcts.num_searches_per_turn == 600
    assert arguments.self_play.mcts.fast_searches_proportion_of_full_searches == pytest.approx(0.25)
    assert arguments.training.num_workers == 0
    assert arguments.training.global_batch_size == 1024
    assert arguments.training.local_batch_size == 1024
    assert arguments.training.learning_rate(0, 'adamw') == pytest.approx(0.0002)
    assert pickle.dumps(arguments.training.learning_rate)
    assert arguments.num_iterations == 2
    assert arguments.num_games_per_iteration == 200
    assert arguments.self_play.num_games_after_which_to_write == 2
    assert arguments.self_play_search_warmup_iterations == 1
    assert arguments.self_play_value_warmup_iterations == 1
    assert arguments.random_seed == 20260717
    assert arguments.evaluation is not None
    assert arguments.evaluation.num_games == 16
    assert arguments.evaluation.max_concurrent_tasks == 1
    assert arguments.evaluation.inference_cache_capacity == 50_000
    assert arguments.evaluation.use_inference_cache
    assert arguments.evaluation.evaluate_initial_checkpoint
    assert arguments.evaluation.stockfish_binary_path == '/workspace/chess/stockfish-18'
    assert arguments.evaluation.stockfish_nodes_per_move == 1_000
    assert arguments.evaluation.stockfish_threads == 1
    assert arguments.evaluation.stockfish_hash_mib == 128


def test_run_configuration_selects_non_cached_inference_client() -> None:
    configuration = load_run_configuration(CONFIGURATION_PATH)
    topology = configuration.topology.model_copy(
        update={
            'use_inference_cache': False,
            'inference_cache_capacity_per_process': 0,
        }
    )
    arguments = training_args()

    apply_run_configuration(arguments, configuration.model_copy(update={'topology': topology}))

    assert not arguments.self_play.use_inference_cache
    assert arguments.self_play.inference_cache_capacity == 0


@pytest.mark.parametrize(
    ('use_inference_cache', 'capacity'),
    (
        (True, 0),
        (False, 1),
    ),
)
def test_run_configuration_rejects_inconsistent_inference_cache_settings(
    use_inference_cache: bool,
    capacity: int,
) -> None:
    configuration = load_run_configuration(CONFIGURATION_PATH)
    data = configuration.model_dump()
    data['topology']['use_inference_cache'] = use_inference_cache
    data['topology']['inference_cache_capacity_per_process'] = capacity

    with pytest.raises(ValidationError, match='Inference cache capacity'):
        RunConfiguration.model_validate(data)


@pytest.mark.parametrize(
    ('use_inference_cache', 'capacity'),
    (
        (True, 0),
        (False, 1),
    ),
)
def test_run_configuration_rejects_inconsistent_evaluation_cache_settings(
    use_inference_cache: bool,
    capacity: int,
) -> None:
    configuration = load_run_configuration(CONFIGURATION_PATH)
    data = configuration.model_dump()
    data['topology']['use_evaluation_inference_cache'] = use_inference_cache
    data['topology']['evaluation_inference_cache_capacity_per_process'] = capacity

    with pytest.raises(ValidationError, match='Evaluation inference cache capacity'):
        RunConfiguration.model_validate(data)


def test_rule_complete_main_applies_training_and_monitoring_schedule() -> None:
    configuration = load_run_configuration(MAIN_CONFIGURATION_PATH)
    arguments = training_args()

    apply_run_configuration(arguments, configuration)

    assert configuration.stage.value == 'clean_retrain'
    assert configuration.run_name == 'complete-training-run-v3'
    assert configuration.tensorboard_run_directory == 'complete-training-run-v3'
    assert configuration.resume.mode.value == 'random_initialization'
    assert configuration.hardware.minimum_disk_gib == 2
    assert configuration.safety.minimum_free_disk_gib == 2
    assert arguments.num_iterations == 500
    assert arguments.num_games_per_iteration == 3500
    assert arguments.self_play.num_games_after_which_to_write == 100
    assert arguments.training.learning_rate(0, 'adamw') == pytest.approx(0.005)
    assert arguments.training.learning_rate(50, 'adamw') == pytest.approx(0.0035)
    assert arguments.training.learning_rate(100, 'adamw') == pytest.approx(0.002)
    assert arguments.self_play_search_warmup_iterations == 15
    assert arguments.self_play_value_warmup_iterations == 30
    assert arguments.self_play_endgame_shortcut_fade_iterations == 50
    assert arguments.self_play.maximum_game_plies == 200
    assert arguments.self_play.maximum_game_plies_until_iteration == 50
    assert arguments.self_play.inference_cache_capacity == 100_000
    assert arguments.self_play.use_inference_cache
    assert arguments.self_play.mcts.num_threads == 6
    assert arguments.self_play.num_parallel_games == 96
    assert arguments.cluster.self_play_tensorboard_processes == 1
    assert arguments.cluster.trainer_ddp_device_ids == (3,)
    assert arguments.cluster.evaluation_device_cycle == (0, 1, 2, 3)
    assert arguments.cluster.self_play_device_ids == (0,) * 6 + (1,) * 6 + (2,) * 6 + (3,) * 6
    assert arguments.cluster.self_play_node_ids_to_pause_during_training == (21, 22, 23)
    assert configuration.topology.maximum_cpu_oversubscription_ratio == pytest.approx(2.5)
    assert arguments.evaluation is not None
    assert arguments.evaluation.num_games == 100
    assert arguments.evaluation.every_n_iterations == 2
    assert arguments.evaluation.max_concurrent_tasks == 16
    assert arguments.evaluation.previous_model_offsets == (5, 10)
    assert arguments.evaluation.historical_model_rotation_period == 5
    assert arguments.evaluation.stockfish_skill_levels == (0, 1, 2, 3)
    assert arguments.evaluation.evaluate_random


def test_v4_configuration_matches_approved_launch_parameters() -> None:
    configuration = load_run_configuration(V4_CONFIGURATION_PATH)
    arguments = training_args()

    validate_run_configuration(configuration, resolved_pilot_hardware())
    apply_run_configuration(arguments, configuration)

    assert configuration.run_name == 'complete-training-run-v4'
    assert configuration.output_path == 'py/training_data/complete-training-run-v4'
    assert configuration.resume.mode.value == 'random_initialization'
    assert arguments.self_play.mcts.num_searches_per_turn == 600
    assert arguments.self_play.mcts.fast_searches_proportion_of_full_searches == pytest.approx(1 / 6)
    assert not arguments.self_play.use_inference_cache
    assert arguments.self_play.inference_cache_capacity == 0
    assert arguments.self_play.maximum_game_plies == 200
    assert arguments.self_play.maximum_game_plies_until_iteration == 50
    assert arguments.self_play_endgame_shortcut_fade_iterations == 50
    assert arguments.self_play.mcts.num_threads == 3
    assert arguments.self_play.num_parallel_games == 96
    assert arguments.cluster.self_play_device_ids == (0,) * 10 + (1,) * 10 + (2,) * 10 + (3,) * 10
    assert arguments.cluster.trainer_rank_zero_device_id == 3
    assert arguments.cluster.trainer_ddp_device_ids == (3, 2, 1, 0)
    assert arguments.training.global_batch_size == 2048
    assert arguments.training.local_batch_size == 512
    assert arguments.cluster.self_play_node_ids_to_pause_during_training == (
        5,
        6,
        7,
        8,
        9,
        15,
        16,
        17,
        18,
        19,
        25,
        26,
        27,
        28,
        29,
        35,
        36,
        37,
        38,
        39,
    )
    assert arguments.evaluation is not None
    assert not arguments.evaluation.use_inference_cache
    assert arguments.evaluation.inference_cache_capacity == 0
    assert arguments.evaluation.maximum_game_plies is None
    assert arguments.evaluation.max_concurrent_tasks == 16
    assert arguments.evaluation.stockfish_hash_mib == 1024


def test_v5_configuration_is_a_fresh_identity_with_v4_parameters() -> None:
    v4_configuration = load_run_configuration(V4_CONFIGURATION_PATH)
    v5_configuration = load_run_configuration(V5_CONFIGURATION_PATH)
    expected = v4_configuration.model_dump()
    expected['run_name'] = 'complete-training-run-v5'
    expected['tensorboard_run_directory'] = 'complete-training-run-v5'
    expected['output_path'] = 'py/training_data/complete-training-run-v5'
    expected['evaluation_protocol']['reference_model_path'] = None
    expected['topology']['trainer_ddp_device_ids'] = (3, 2)
    expected['topology']['self_play_processes_per_device_during_training'] = (10, 10, 5, 5)
    expected['topology']['max_concurrent_evaluation_tasks'] = 8
    expected['workload']['training_global_batch_size'] = 1024
    expected['workload']['training_sampling_window'] = 15
    expected['workload']['self_play_fast_searches_per_turn'] = 150
    expected['workload']['self_play_maximum_game_plies_until_iteration'] = 80
    expected['workload']['self_play_final_maximum_game_plies'] = 300
    expected['retention']['replay_window_iterations'] = 15
    expected['evaluation_protocol']['historical_model_iterations'] = (0,) + expected['evaluation_protocol'][
        'historical_model_iterations'
    ]
    expected['evaluation_protocol']['historical_model_rotation_period'] = 2

    assert v5_configuration.model_dump() == expected


def test_v5_configuration_uses_a_fixed_15_iteration_replay_window() -> None:
    configuration = load_run_configuration(V5_CONFIGURATION_PATH)
    arguments = training_args()

    validate_run_configuration(configuration, resolved_pilot_hardware())
    apply_run_configuration(arguments, configuration)

    assert arguments.training.sampling_window(0) == 15
    assert arguments.training.sampling_window(500) == 15
    assert pickle.loads(pickle.dumps(arguments.training.sampling_window))(144) == 15
    assert arguments.artifact_retention.replay_window_iterations == 15
    assert arguments.self_play.maximum_game_plies == 200
    assert arguments.self_play.maximum_game_plies_until_iteration == 80
    assert arguments.self_play.final_maximum_game_plies == 300


@pytest.mark.parametrize('run_directory', ('../escape', 'nested/run', 'run name'))
def test_run_configuration_rejects_unsafe_tensorboard_run_directory(run_directory: str) -> None:
    configuration = load_run_configuration(CONFIGURATION_PATH)
    data = configuration.model_dump()
    data['tensorboard_run_directory'] = run_directory

    with pytest.raises(ValidationError, match='tensorboard_run_directory'):
        RunConfiguration.model_validate(data)


def test_training_rejects_legacy_configuration_without_retention() -> None:
    configuration = load_run_configuration(CONFIGURATION_PATH)
    legacy_configuration = configuration.model_copy(update={'retention': None})

    with pytest.raises(ValueError, match='retention must be configured'):
        apply_run_configuration(training_args(), legacy_configuration)


def test_approval_must_match_exact_configuration_and_source() -> None:
    configuration = load_run_configuration(CONFIGURATION_PATH)
    source_revision = '1' * 40
    approval = ApprovalRecord(
        approved_by='Bertil',
        approved_at_utc=datetime(2026, 7, 17, tzinfo=timezone.utc),
        run_name=configuration.run_name,
        source_revision=source_revision,
        configuration_sha256=configuration_sha256(configuration),
        provider_name=configuration.hardware.provider_name,
        offer_id=configuration.hardware.offer_id,
        cost_currency=configuration.budget.currency,
        hourly_price=configuration.budget.hourly_price,
        maximum_cost=configuration.budget.maximum_cost,
        maximum_wall_time_minutes=configuration.budget.maximum_wall_time_minutes,
    )

    validate_approval(configuration, approval, source_revision)

    mismatched_approval = approval.model_copy(update={'maximum_cost': 1.0})
    with pytest.raises(ValueError, match='cost ceiling'):
        validate_approval(configuration, mismatched_approval, source_revision)

    changed_data = configuration.model_dump()
    changed_data['workload']['training_global_batch_size'] = 512
    changed_data['workload']['training_local_batch_size'] = 512
    changed_configuration = RunConfiguration.model_validate(changed_data)
    with pytest.raises(ValueError, match='configuration hash'):
        validate_approval(changed_configuration, approval, source_revision)


def test_existing_manifest_preserves_initial_dynamic_hardware_measurements(tmp_path: Path) -> None:
    configuration = load_run_configuration(CONFIGURATION_PATH)
    source_revision = '1' * 40
    approval = ApprovalRecord(
        approved_by='Bertil',
        approved_at_utc=datetime(2026, 7, 17, tzinfo=timezone.utc),
        run_name=configuration.run_name,
        source_revision=source_revision,
        configuration_sha256=configuration_sha256(configuration),
        provider_name=configuration.hardware.provider_name,
        offer_id=configuration.hardware.offer_id,
        cost_currency=configuration.budget.currency,
        hourly_price=configuration.budget.hourly_price,
        maximum_cost=configuration.budget.maximum_cost,
        maximum_wall_time_minutes=configuration.budget.maximum_wall_time_minutes,
    )
    initial_hardware = resolved_pilot_hardware()
    initial_manifest = RunManifest(
        configuration=configuration,
        approval=approval,
        resolved_hardware=initial_hardware,
        source_revision=source_revision,
        source_worktree_clean=True,
        initial_model_sha256='2' * 64,
        evaluation_dataset_sha256=None,
        stockfish_binary_sha256='3' * 64,
        open_file_soft_limit=4096,
        torch_version='2.7.1+cu128',
        cuda_version='12.8',
    )
    manifest_path = tmp_path / 'run_manifest.json'
    write_run_manifest(manifest_path, initial_manifest)
    current_manifest = initial_manifest.model_copy(
        update={'resolved_hardware': initial_hardware.model_copy(update={'free_disk_gib': 91.2})}
    )

    preserved_manifest = write_run_manifest(manifest_path, current_manifest)

    assert preserved_manifest == initial_manifest


def test_changed_manifest_archives_previous_phase(tmp_path: Path) -> None:
    configuration = load_run_configuration(CONFIGURATION_PATH)
    initial_manifest = RunManifest(
        configuration=configuration,
        approval=ApprovalRecord(
            approved_by='Bertil',
            approved_at_utc=datetime(2026, 7, 17, tzinfo=timezone.utc),
            run_name=configuration.run_name,
            source_revision='1' * 40,
            configuration_sha256=configuration_sha256(configuration),
            provider_name=configuration.hardware.provider_name,
            offer_id=configuration.hardware.offer_id,
            cost_currency=configuration.budget.currency,
            hourly_price=configuration.budget.hourly_price,
            maximum_cost=configuration.budget.maximum_cost,
            maximum_wall_time_minutes=configuration.budget.maximum_wall_time_minutes,
        ),
        resolved_hardware=resolved_pilot_hardware(),
        source_revision='1' * 40,
        source_worktree_clean=True,
        initial_model_sha256='2' * 64,
        evaluation_dataset_sha256=None,
        stockfish_binary_sha256='3' * 64,
        open_file_soft_limit=4096,
        torch_version='2.7.1+cu128',
        cuda_version='12.8',
    )
    manifest_path = tmp_path / 'run_manifest.json'
    write_run_manifest(manifest_path, initial_manifest)
    current_manifest = initial_manifest.model_copy(
        update={
            'source_revision': '4' * 40,
            'approval': initial_manifest.approval.model_copy(update={'source_revision': '4' * 40}),
        }
    )

    written_manifest = write_run_manifest(manifest_path, current_manifest)

    history_paths = tuple((tmp_path / 'run_manifests').glob('run_manifest-*.json'))
    assert written_manifest == current_manifest
    assert RunManifest.model_validate_json(manifest_path.read_text(encoding='utf-8')) == current_manifest
    assert len(history_paths) == 1
    assert RunManifest.model_validate_json(history_paths[0].read_text(encoding='utf-8')) == initial_manifest


def test_changed_manifest_rejects_different_run_identity(tmp_path: Path) -> None:
    configuration = load_run_configuration(CONFIGURATION_PATH)
    initial_manifest = RunManifest(
        configuration=configuration,
        approval=ApprovalRecord(
            approved_by='Bertil',
            approved_at_utc=datetime(2026, 7, 17, tzinfo=timezone.utc),
            run_name=configuration.run_name,
            source_revision='1' * 40,
            configuration_sha256=configuration_sha256(configuration),
            provider_name=configuration.hardware.provider_name,
            offer_id=configuration.hardware.offer_id,
            cost_currency=configuration.budget.currency,
            hourly_price=configuration.budget.hourly_price,
            maximum_cost=configuration.budget.maximum_cost,
            maximum_wall_time_minutes=configuration.budget.maximum_wall_time_minutes,
        ),
        resolved_hardware=resolved_pilot_hardware(),
        source_revision='1' * 40,
        source_worktree_clean=True,
        initial_model_sha256='2' * 64,
        evaluation_dataset_sha256=None,
        stockfish_binary_sha256='3' * 64,
        open_file_soft_limit=4096,
        torch_version='2.7.1+cu128',
        cuda_version='12.8',
    )
    manifest_path = tmp_path / 'run_manifest.json'
    write_run_manifest(manifest_path, initial_manifest)
    other_configuration = configuration.model_copy(update={'run_name': 'other-run'})
    other_manifest = initial_manifest.model_copy(update={'configuration': other_configuration})

    with pytest.raises(ValueError, match='different run'):
        write_run_manifest(manifest_path, other_manifest)
