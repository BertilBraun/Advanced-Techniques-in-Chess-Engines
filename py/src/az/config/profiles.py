from __future__ import annotations

from src.az.config.experiment import HardwareConfiguration, ManifestPolicy
from src.az.config.model import (
    FixedModelSchedule,
    ModelConfiguration,
)
from src.az.config.runtime import (
    DeviceAssignment,
    RetentionConfiguration,
    TelemetryConfiguration,
    TelemetryMetric,
    TopologyConfiguration,
)
from src.az.config.search import (
    ConstantTemperature,
    DirichletRootExploration,
    DisabledRootExploration,
    DisabledTreeReuse,
    FpuConfiguration,
    FullBudgetStopping,
    PuctSearchConfiguration,
    PlyTemperatureSchedule,
    RootExplorationConfiguration,
    SearchAlgorithmConfiguration,
    SearchConfiguration,
    SearchInferenceConfiguration,
    TemperatureConfiguration,
    TemperatureStage,
    TreeReuseConfiguration,
    VisitedChildMeanFpu,
    FixedSearchBudget,
)
from src.az.config.training import (
    AdamWOptimizerConfiguration,
    ConstantLearningRate,
    InitialStateOnly,
    ReplayCreditConfiguration,
    SelfPlayConfiguration,
    TrainingConfiguration,
)
from src.az.games.go.configuration import (
    DisabledResignation,
    GoGameConfiguration,
    GoObjectiveConfiguration,
    RandomGoOpponent,
    ResidualGoModelConfiguration,
)
from src.az.config.base import DeterminismMode


def default_manifest_policy() -> ManifestPolicy:
    return ManifestPolicy(
        require_clean_source=True,
        record_dependency_versions=True,
        determinism_mode=DeterminismMode.SEEDED_CONCURRENT,
    )


def planned_hardware_profile() -> HardwareConfiguration:
    return HardwareConfiguration(
        profile_name='planned-two-gpu-experiment-node',
        provider='planned-gpu-rental',
        offer_id='planned-2x-rtx4090-profile',
        expected_gpu_model='NVIDIA GeForce RTX 4090',
        expected_gpu_count=2,
        minimum_logical_cpu_count=16,
        minimum_ram_gib=64,
        minimum_free_disk_gib=100,
        hourly_cost=None,
        currency='EUR',
    )


def default_topology() -> TopologyConfiguration:
    return TopologyConfiguration(
        trainer=DeviceAssignment(device_ids=(0, 1)),
        self_play=DeviceAssignment(device_ids=(0, 1)),
        evaluation=DeviceAssignment(device_ids=(0,)),
        self_play_workers_per_device=2,
        native_threads_per_worker=4,
        inference_workers_per_device=1,
        inference_batch_size=128,
        maximum_pending_inference_batches=2,
        data_loader_workers_per_rank=4,
        evaluation_concurrency=1,
    )


def default_game() -> GoGameConfiguration:
    return GoGameConfiguration(
        kind='go',
        board_size=7,
        komi_half_points=15,
        scoring_rule='area',
        ko_rule='positional_superko',
        suicide_rule='illegal',
        pass_exempt_from_superko=True,
        score_comparison='doubled_integer_points',
        safety_ply_cap=512,
        history_length=8,
        history_planes_per_position=2,
        include_color_plane=True,
        pass_action='last',
        normal_termination='two_consecutive_passes',
        symmetry_group='dihedral_8',
        capped_game_value_target_weight=0,
        resignation=DisabledResignation(kind='disabled'),
    )


def default_model_architecture() -> ResidualGoModelConfiguration:
    return ResidualGoModelConfiguration(
        family='residual_go',
        channels=128,
        residual_blocks=10,
        policy_channels=2,
        value_hidden_size=256,
        normalization='batch',
        activation='relu',
    )


def default_model() -> ModelConfiguration:
    return ModelConfiguration(
        schedule=FixedModelSchedule(
            kind='fixed',
            architecture=default_model_architecture(),
        )
    )


def default_search_algorithm() -> SearchAlgorithmConfiguration:
    return PuctSearchConfiguration(kind='puct', exploration_constant=1.5)


def default_fpu() -> FpuConfiguration:
    return VisitedChildMeanFpu(kind='visited_child_mean', no_visited_child_value=0)


def default_root_exploration() -> RootExplorationConfiguration:
    return DirichletRootExploration(kind='dirichlet', alpha=0.3, exploration_fraction=0.25)


def default_temperature() -> TemperatureConfiguration:
    return PlyTemperatureSchedule(
        kind='by_ply',
        stages=(TemperatureStage(maximum_ply_exclusive=20, temperature=1),),
        final_temperature=0,
    )


def default_tree_reuse() -> TreeReuseConfiguration:
    return DisabledTreeReuse(kind='disabled')


def default_search_inference() -> SearchInferenceConfiguration:
    return SearchInferenceConfiguration(
        maximum_batch_size=128,
        maximum_wait_microseconds=1_000,
        cache_capacity=100_000,
    )


def default_self_play() -> SelfPlayConfiguration:
    return SelfPlayConfiguration(
        start_states=InitialStateOnly(kind='initial_state_only'),
        concurrent_games_per_worker=64,
        games_per_shard=32,
        value_target_weight=1,
        capped_game_policy_targets_remain_eligible=True,
        policy_target_source='search_budget',
    )


def default_replay_credits() -> ReplayCreditConfiguration:
    return ReplayCreditConfiguration(
        target_reuse=4,
        optimizer_steps_per_quantum=50,
        minimum_positions_before_training=50_000,
    )


def default_training() -> TrainingConfiguration:
    return TrainingConfiguration(
        global_batch_size=1_024,
        local_batch_size=512,
        maximum_optimizer_steps=500_000,
        optimizer=AdamWOptimizerConfiguration(
            kind='adamw',
            learning_rate=0.002,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
            weight_decay=1e-4,
        ),
        learning_rate_schedule=ConstantLearningRate(kind='constant', multiplier=1),
        precision='bfloat16',
        objective=GoObjectiveConfiguration(
            kind='go_policy_value',
            policy_loss_weight=1,
            value_loss_weight=1,
            l2_regularization_weight=0,
        ),
        checkpoint_every_optimizer_steps=1_000,
        gradient_clip_norm=1,
    )


def default_evaluation_search() -> SearchConfiguration:
    return SearchConfiguration(
        algorithm=default_search_algorithm(),
        budget=FixedSearchBudget(kind='fixed', simulations=128),
        stopping=FullBudgetStopping(kind='full_budget'),
        fpu=default_fpu(),
        root_exploration=DisabledRootExploration(kind='disabled'),
        temperature=ConstantTemperature(kind='constant', temperature=0),
        tree_reuse=DisabledTreeReuse(kind='disabled'),
        inference=default_search_inference(),
        backup_discount=1,
    )


def default_evaluation_opponent() -> RandomGoOpponent:
    return RandomGoOpponent(kind='random')


def default_telemetry() -> TelemetryConfiguration:
    return TelemetryConfiguration(
        write_every_seconds=10,
        resource_sample_every_seconds=5,
        required_metrics=tuple(TelemetryMetric),
        search_trace_sample_probability=0,
    )


def default_retention() -> RetentionConfiguration:
    return RetentionConfiguration(
        recent_checkpoint_count=5,
        milestone_every_optimizer_steps=10_000,
        retain_replay_shards=True,
        retain_search_traces=True,
        retain_raw_evaluation_games=True,
    )
