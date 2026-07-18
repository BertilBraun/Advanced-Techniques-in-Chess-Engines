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
        network=NetworkParams(num_layers=1, hidden_size=8, se_positions=()),
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
            num_moves_after_which_to_play_greedy=10,
            portion_of_samples_to_keep=0.5,
        ),
        training=TrainingParams(
            num_epochs=1,
            batch_size=8,
            optimizer='adamw',
            sampling_window=sampling_window,
            learning_rate=learning_rate,
            learning_rate_scheduler=learning_rate_scheduler,
            num_workers=0,
        ),
        cluster=ClusterParams(
            trainer_device_id=0,
            evaluation_device_id=0,
            self_play_device_ids=(0,),
            trainer_cpu_threads=1,
            trainer_interop_threads=1,
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

    assert arguments.cluster.trainer_device_id == 3
    assert arguments.cluster.evaluation_device_id == 3
    assert arguments.cluster.self_play_device_ids == (0,) * 6 + (1,) * 6 + (2,) * 6 + (3,) * 2
    assert arguments.cluster.trainer_cpu_threads == 8
    assert arguments.self_play.mcts.num_threads == 2
    assert arguments.self_play.num_parallel_games == 64
    assert arguments.self_play.inference_cache_capacity == 250_000
    assert arguments.training.num_workers == 0
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
    assert arguments.evaluation.evaluate_initial_checkpoint
    assert arguments.evaluation.stockfish_binary_path == '/workspace/chess/stockfish-18'
    assert arguments.evaluation.stockfish_nodes_per_move == 1_000
    assert arguments.evaluation.stockfish_threads == 1
    assert arguments.evaluation.stockfish_hash_mib == 128


def test_clean_main_reproduces_historical_training_and_monitoring_schedule() -> None:
    configuration = load_run_configuration(MAIN_CONFIGURATION_PATH)
    arguments = training_args()

    apply_run_configuration(arguments, configuration)

    assert configuration.stage.value == 'clean_retrain'
    assert configuration.resume.mode.value == 'random_initialization'
    assert arguments.num_iterations == 200
    assert arguments.num_games_per_iteration == 5000
    assert arguments.self_play.num_games_after_which_to_write == 10
    assert arguments.training.learning_rate(0, 'adamw') == pytest.approx(0.005)
    assert arguments.training.learning_rate(60, 'adamw') == pytest.approx(0.002)
    assert arguments.self_play_search_warmup_iterations == 15
    assert arguments.self_play_value_warmup_iterations == 30
    assert arguments.self_play.inference_cache_capacity == 1_500_000
    assert arguments.evaluation is not None
    assert arguments.evaluation.num_games == 100
    assert arguments.evaluation.every_n_iterations == 1
    assert arguments.evaluation.previous_model_offsets == (5, 10)
    assert arguments.evaluation.stockfish_skill_levels == (0, 1, 2, 3)
    assert arguments.evaluation.evaluate_random


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
