from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from src.experiment.run_configuration import (
    BudgetConfiguration,
    EnvironmentConfiguration,
    EvaluationScheduleConfiguration,
    HardwareConfiguration,
    PiecewiseModelVersionLearningRate,
    ResumeConfiguration,
    RetentionConfiguration,
    SafetyConfiguration,
    TrainingStage,
    WorkloadConfiguration,
)
from src.games.chess.ChessGame import BINARY_CHANNELS, SCALAR_CHANNELS
from src.train.TrainingArgs import (
    ArtifactRetention,
    ClusterParams,
    DirectSelfPlayParams,
    EvaluationParams,
    MCTSParams,
    NetworkParams,
    RuntimeLimits,
    SelfPlayParams,
    TrainingArgs,
    TrainingParams,
)
from src.util.atomic_file import write_text_atomically
from src.util.frozen_model import FrozenModel


class TrainerTopologyConfiguration(FrozenModel):
    device_type: Literal['cuda', 'cpu']
    process_group_backend: Literal['nccl', 'gloo']
    rank_zero_device_id: int = Field(ge=0)
    ddp_device_ids: tuple[int, ...]
    cpu_threads: int = Field(gt=0)
    interop_threads: int = Field(gt=0)
    dataloader_workers: int = Field(ge=0)

    @model_validator(mode='after')
    def validate_devices(self) -> TrainerTopologyConfiguration:
        if not self.ddp_device_ids:
            raise ValueError('At least one trainer device must be configured.')
        if any(device_id < 0 for device_id in self.ddp_device_ids):
            raise ValueError('DDP trainer device IDs cannot be negative.')
        if self.ddp_device_ids[0] != self.rank_zero_device_id:
            raise ValueError('The rank-zero trainer device must be first in the DDP device list.')
        if len(set(self.ddp_device_ids)) != len(self.ddp_device_ids):
            raise ValueError('DDP trainer devices must be unique.')
        if self.process_group_backend == 'nccl' and self.device_type != 'cuda':
            raise ValueError('NCCL can only be selected for CUDA trainer devices.')
        if self.device_type == 'cpu' and (self.process_group_backend != 'gloo' or self.ddp_device_ids != (0,)):
            raise ValueError('CPU training requires Gloo on the single logical device ID 0.')
        return self


class InferenceTopologyConfiguration(FrozenModel):
    cache_capacity_per_process: int = Field(ge=0)
    direct: DirectSelfPlayParams | None = None

    @model_validator(mode='after')
    def validate_inference_mode(self) -> InferenceTopologyConfiguration:
        if self.direct is not None and self.cache_capacity_per_process > 0:
            raise ValueError('Direct inference and inference caching are mutually exclusive.')
        return self


class SearchTopologyConfiguration(FrozenModel):
    threads_per_process: int = Field(gt=0)
    parallel_searches: int = Field(gt=0)


class SelfPlayTopologyConfiguration(FrozenModel):
    processes_per_device: tuple[int, ...]
    processes_per_device_during_training: tuple[int, ...]
    tensorboard_processes: int = Field(ge=1)
    parallel_games_per_process: int = Field(gt=0)
    search: SearchTopologyConfiguration
    inference: InferenceTopologyConfiguration

    @model_validator(mode='after')
    def validate_process_counts(self) -> SelfPlayTopologyConfiguration:
        if not self.processes_per_device or any(count < 0 for count in self.processes_per_device):
            raise ValueError('Self-play process counts must contain non-negative entries.')
        if sum(self.processes_per_device) == 0:
            raise ValueError('At least one self-play process must be configured.')
        if len(self.processes_per_device_during_training) != len(self.processes_per_device):
            raise ValueError('Training self-play process counts must contain one entry per device.')
        if any(
            not 0 <= training_count <= configured_count
            for training_count, configured_count in zip(
                self.processes_per_device_during_training,
                self.processes_per_device,
            )
        ):
            raise ValueError('Training self-play process counts cannot exceed configured process counts.')
        if self.tensorboard_processes > sum(self.processes_per_device):
            raise ValueError('TensorBoard self-play processes cannot exceed all self-play processes.')
        return self


class EvaluationTopologyConfiguration(FrozenModel):
    device_cycle: tuple[int, ...]
    maximum_concurrent_evaluations: int = Field(ge=1)
    maximum_concurrent_tasks: int = Field(ge=1)
    search: SearchTopologyConfiguration
    inference: InferenceTopologyConfiguration

    @model_validator(mode='after')
    def validate_devices(self) -> EvaluationTopologyConfiguration:
        if not self.device_cycle or any(device_id < 0 for device_id in self.device_cycle):
            raise ValueError('Evaluation device cycle must contain non-negative device IDs.')
        return self


class HostTopologyConfiguration(FrozenModel):
    reserved_logical_cpus: int = Field(ge=1)
    maximum_cpu_oversubscription_ratio: float = Field(ge=1.0, le=5.0)


class ExperimentTopologyConfiguration(FrozenModel):
    training: TrainerTopologyConfiguration
    self_play: SelfPlayTopologyConfiguration
    evaluation: EvaluationTopologyConfiguration
    host: HostTopologyConfiguration


class SharedEvaluationConfiguration(FrozenModel):
    games: int = Field(gt=0)
    searches_per_turn: int = Field(gt=0)
    teacher_searches_per_turn: int = Field(gt=0)
    teacher_games: int = Field(gt=0)
    schedule: EvaluationScheduleConfiguration

    @model_validator(mode='after')
    def validate_teacher_games(self) -> SharedEvaluationConfiguration:
        if self.teacher_games % 2 or self.teacher_games > self.games:
            raise ValueError('Teacher evaluation games must be an even subset of paired evaluation games.')
        return self


class ChessRunConfiguration(FrozenModel):
    run_name: str = Field(min_length=1)
    tensorboard_run_directory: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9_-]*$')
    stage: TrainingStage
    requires_explicit_approval: bool
    output_path: str
    resume: ResumeConfiguration
    budget: BudgetConfiguration
    hardware: HardwareConfiguration
    topology: ExperimentTopologyConfiguration
    workload: WorkloadConfiguration
    safety: SafetyConfiguration
    retention: RetentionConfiguration
    evaluation: SharedEvaluationConfiguration
    environment: EnvironmentConfiguration

    @model_validator(mode='after')
    def validate_shared_configuration(self) -> ChessRunConfiguration:
        world_size = len(self.topology.training.ddp_device_ids)
        expected_global_batch_size = self.workload.training_local_batch_size * world_size
        if self.workload.training_global_batch_size != expected_global_batch_size:
            raise ValueError(
                f'Global training batch size {self.workload.training_global_batch_size} must equal '
                f'local batch size {self.workload.training_local_batch_size} times world size {world_size}.'
            )
        self.evaluation.schedule.to_parameters(self.workload.credit_training.optimizer_steps_per_quantum)
        if len(self.topology.self_play.processes_per_device) != self.hardware.gpu_count:
            raise ValueError('Self-play process counts must contain one entry per configured GPU.')
        return self


class ChessRulesConfiguration(FrozenModel):
    variant: Literal['standard'] = 'standard'
    chess960: bool = False
    automatic_fifty_move_draw: bool = True
    automatic_threefold_repetition_draw: bool = True


class ChessRepresentationConfiguration(FrozenModel):
    board_length: Literal[8] = 8
    binary_channels: tuple[int, ...] = BINARY_CHANNELS
    scalar_channels: tuple[int, ...] = SCALAR_CHANNELS
    action_encoding: Literal['chess-move2index-v1'] = 'chess-move2index-v1'
    canonical_player_perspective: bool = True

    @model_validator(mode='after')
    def validate_channels(self) -> ChessRepresentationConfiguration:
        if set(self.binary_channels) & set(self.scalar_channels):
            raise ValueError('Chess binary and scalar channels must be disjoint.')
        expected_channels = tuple(range(len(self.binary_channels) + len(self.scalar_channels)))
        actual_channels = tuple(sorted(self.binary_channels + self.scalar_channels))
        if actual_channels != expected_channels:
            raise ValueError('Chess representation channels must form one dense range starting at zero.')
        return self


class ChessSelfPlayConfiguration(FrozenModel):
    num_moves_after_which_to_play_greedy: int = Field(default=50, ge=1)
    starting_temperature: float = Field(default=1.3, gt=0.0)
    final_temperature: float = Field(default=0.1, gt=0.0)
    exploration_constant: float = Field(default=1.5, gt=0.0)
    dirichlet_alpha: float = Field(default=0.3, gt=0.0)
    dirichlet_epsilon: float = Field(default=0.25, ge=0.0, le=1.0)
    percentage_of_node_visits_to_keep: float = Field(default=0.6, ge=0.0, le=1.0)
    playout_cap_randomization: float = Field(default=0.25, ge=0.0, le=1.0)

    @model_validator(mode='after')
    def validate_temperatures(self) -> ChessSelfPlayConfiguration:
        if self.final_temperature > self.starting_temperature:
            raise ValueError('Chess final temperature cannot exceed the starting temperature.')
        return self


class ChessDatasetEvaluationConfiguration(FrozenModel):
    path: str


class ChessModelLadderConfiguration(FrozenModel):
    reference_model_path: str | None
    previous_model_offsets: tuple[int, ...]
    historical_model_versions: tuple[int, ...]
    historical_model_rotation_period: int = Field(gt=0)

    @model_validator(mode='after')
    def validate_model_versions(self) -> ChessModelLadderConfiguration:
        if any(offset <= 0 for offset in self.previous_model_offsets):
            raise ValueError('Previous-model offsets must be positive.')
        if any(model_version < 0 for model_version in self.historical_model_versions) or (
            tuple(sorted(set(self.historical_model_versions))) != self.historical_model_versions
        ):
            raise ValueError('Historical model versions must be unique, non-negative, and increasing.')
        return self


class StockfishEvaluationConfiguration(FrozenModel):
    skill_levels: tuple[int, ...]
    binary_path: str | None
    nodes_per_move: int = Field(gt=0)
    threads: int = Field(gt=0)
    hash_mib: int = Field(gt=0)

    @model_validator(mode='after')
    def validate_skill_levels(self) -> StockfishEvaluationConfiguration:
        if any(not 0 <= level <= 20 for level in self.skill_levels):
            raise ValueError('Stockfish skill levels must be between 0 and 20.')
        return self


class ChessEvaluationConfiguration(FrozenModel):
    opening_suite_path: str
    raw_results_subdirectory: str
    maximum_game_plies: int | None
    bootstrap_seed: int = Field(ge=0)
    bootstrap_samples: int = Field(gt=0)
    evaluate_initial_checkpoint: bool
    evaluate_random: bool
    exploration_constant: float = Field(default=1.0, gt=0.0)
    dataset: ChessDatasetEvaluationConfiguration | None
    model_ladder: ChessModelLadderConfiguration
    stockfish: StockfishEvaluationConfiguration | None


class ChessConfiguration(FrozenModel):
    game: Literal['chess'] = 'chess'
    rules: ChessRulesConfiguration = ChessRulesConfiguration()
    representation: ChessRepresentationConfiguration = ChessRepresentationConfiguration()
    network: NetworkParams = NetworkParams(num_layers=12, hidden_size=112)
    self_play: ChessSelfPlayConfiguration = ChessSelfPlayConfiguration()
    evaluation: ChessEvaluationConfiguration


class ChessExperimentConfiguration(FrozenModel):
    run: ChessRunConfiguration
    chess: ChessConfiguration

    @model_validator(mode='after')
    def validate_chess_evaluation(self) -> ChessExperimentConfiguration:
        retention = self.run.retention
        ladder = self.chess.evaluation.model_ladder
        if ladder.previous_model_offsets and (
            max(ladder.previous_model_offsets) >= retention.recent_inference_checkpoint_count
        ):
            raise ValueError('Recent inference-checkpoint retention must exceed every previous-model offset.')
        if any(
            model_version % retention.milestone_inference_interval != 0
            for model_version in ladder.historical_model_versions
        ):
            raise ValueError('Historical model versions must align with retained milestone checkpoints.')
        return self


def load_chess_experiment_configuration(path: Path) -> ChessExperimentConfiguration:
    payload = path.read_text(encoding='utf-8')
    parsed = yaml.safe_load(payload) if path.suffix.casefold() in {'.yaml', '.yml'} else json.loads(payload)
    if not isinstance(parsed, dict):
        raise ValueError(f'Chess experiment file must contain a mapping: {path}')
    return ChessExperimentConfiguration.model_validate(parsed)


def validate_experiment_queue(paths: tuple[Path, ...]) -> tuple[ChessExperimentConfiguration, ...]:
    if not paths:
        raise ValueError('Experiment queue validation requires at least one configuration path.')
    return tuple(load_chess_experiment_configuration(path) for path in paths)


def write_resolved_chess_experiment(path: Path, configuration: ChessExperimentConfiguration) -> None:
    write_text_atomically(path, configuration.model_dump_json(indent=2) + '\n')


def build_chess_training_args(configuration: ChessExperimentConfiguration) -> TrainingArgs:
    run = configuration.run
    workload = run.workload
    return TrainingArgs(
        save_path=run.output_path,
        network=configuration.chess.network,
        self_play=_build_self_play_parameters(configuration),
        training=_build_training_parameters(configuration),
        cluster=_build_cluster_parameters(run.topology),
        run_limits=RuntimeLimits(
            cost_currency=run.budget.currency,
            hourly_price=run.budget.hourly_price,
            maximum_cost=run.budget.maximum_cost,
            maximum_wall_time_seconds=run.budget.maximum_wall_time_minutes * 60,
            maximum_open_file_count=run.safety.maximum_open_file_count,
            maximum_host_ram_percent=run.safety.maximum_host_ram_percent,
            minimum_free_disk_gib=run.safety.minimum_free_disk_gib,
        ),
        artifact_retention=ArtifactRetention(**run.retention.model_dump()),
        evaluation_schedule=run.evaluation.schedule.to_parameters(workload.credit_training.optimizer_steps_per_quantum),
        random_seed=workload.random_seed,
        self_play_search_warmup_model_versions=workload.self_play_search_warmup_model_versions,
        self_play_endgame_shortcut_fade_model_versions=workload.self_play_endgame_shortcut_fade_model_versions,
        evaluation=_build_evaluation_parameters(configuration),
    )


def _build_self_play_parameters(configuration: ChessExperimentConfiguration) -> SelfPlayParams:
    workload = configuration.run.workload
    chess = configuration.chess.self_play
    topology = configuration.run.topology.self_play
    return SelfPlayParams(
        mcts=MCTSParams(
            num_searches_per_turn=workload.self_play_searches_per_turn,
            fast_searches_proportion_of_full_searches=(
                workload.self_play_fast_searches_per_turn / workload.self_play_searches_per_turn
            ),
            playout_cap_randomization=chess.playout_cap_randomization,
            num_parallel_searches=topology.search.parallel_searches,
            dirichlet_epsilon=chess.dirichlet_epsilon,
            dirichlet_alpha=chess.dirichlet_alpha,
            c_param=chess.exploration_constant,
            num_threads=topology.search.threads_per_process,
            percentage_of_node_visits_to_keep=chess.percentage_of_node_visits_to_keep,
        ),
        num_parallel_games=topology.parallel_games_per_process,
        inference_cache_capacity=topology.inference.cache_capacity_per_process,
        use_inference_cache=topology.inference.cache_capacity_per_process > 0,
        num_moves_after_which_to_play_greedy=chess.num_moves_after_which_to_play_greedy,
        maximum_game_plies=workload.self_play_maximum_game_plies,
        maximum_game_plies_until_model_version=workload.self_play_maximum_game_plies_until_model_version,
        maximum_game_plies_hold_until_model_version=workload.self_play_maximum_game_plies_hold_until_model_version,
        final_maximum_game_plies=workload.self_play_final_maximum_game_plies,
        endgame_continuation_start_plies=workload.self_play_endgame_continuation_start_plies,
        low_material_termination_minimum_plies=workload.self_play_low_material_termination_minimum_plies,
        low_material_termination_piece_threshold_per_player=(
            workload.self_play_low_material_termination_piece_threshold_per_player
        ),
        low_material_termination_probability=workload.self_play_low_material_termination_probability,
        starting_temperature=chess.starting_temperature,
        final_temperature=chess.final_temperature,
        resignation=workload.resignation.to_parameters(),
        direct_inference=topology.inference.direct,
        disagreement_prefix_start_probability=workload.disagreement_prefix_start_probability,
        disagreement_prefix_maximum_ply=workload.disagreement_prefix_maximum_ply,
        disagreement_prefix_archive_capacity=workload.disagreement_prefix_archive_capacity,
        disagreement_prefix_weight_smoothing=workload.disagreement_prefix_weight_smoothing,
        disagreement_prefix_weight_cap=workload.disagreement_prefix_weight_cap,
        initial_num_searches_per_turn=workload.self_play_initial_searches_per_turn,
    )


def _build_training_parameters(configuration: ChessExperimentConfiguration) -> TrainingParams:
    run = configuration.run
    workload = run.workload
    return TrainingParams(
        num_epochs=1,
        global_batch_size=workload.training_global_batch_size,
        local_batch_size=workload.training_local_batch_size,
        optimizer=run.resume.optimizer,
        learning_rate=PiecewiseModelVersionLearningRate(
            workload.credit_training.learning_rate_schedule,
            workload.credit_training.optimizer_steps_per_quantum,
        ),
        learning_rate_scheduler=lambda batch_percentage, base_learning_rate: base_learning_rate,
        credit_training=workload.credit_training.to_parameters(),
        max_buffer_samples=workload.credit_training.maximum_replay_capacity_unique_positions,
        outcome_value_loss_weight=workload.outcome_value_loss_weight,
        mcts_value_loss_weight=workload.mcts_value_loss_weight,
        mcts_value_loss_scale=workload.mcts_value_loss_scale,
        mcts_value_target_warmup_optimizer_steps=workload.mcts_value_target_warmup_optimizer_steps,
        duplicate_multiplicity_weight_cap=workload.duplicate_multiplicity_weight_cap,
        num_workers=run.topology.training.dataloader_workers,
    )


def _build_evaluation_parameters(configuration: ChessExperimentConfiguration) -> EvaluationParams:
    run = configuration.run
    shared = run.evaluation
    chess = configuration.chess.evaluation
    topology = run.topology.evaluation
    ladder = chess.model_ladder
    stockfish = chess.stockfish
    return EvaluationParams(
        num_searches_per_turn=shared.searches_per_turn,
        num_games=shared.games,
        every_n_model_versions=1,
        evaluate_initial_checkpoint=chess.evaluate_initial_checkpoint,
        max_concurrent_tasks=topology.maximum_concurrent_tasks,
        inference_cache_capacity=topology.inference.cache_capacity_per_process,
        use_inference_cache=topology.inference.cache_capacity_per_process > 0,
        dataset_path=chess.dataset.path if chess.dataset is not None else None,
        reference_model_path=ladder.reference_model_path,
        opening_suite_path=chess.opening_suite_path,
        raw_results_path=str(Path(run.output_path) / chess.raw_results_subdirectory),
        maximum_game_plies=chess.maximum_game_plies,
        bootstrap_seed=chess.bootstrap_seed,
        bootstrap_samples=chess.bootstrap_samples,
        mcts_threads=topology.search.threads_per_process,
        previous_model_offsets=ladder.previous_model_offsets,
        historical_model_versions=ladder.historical_model_versions,
        historical_model_rotation_period=ladder.historical_model_rotation_period,
        stockfish_skill_levels=stockfish.skill_levels if stockfish is not None else (),
        stockfish_binary_path=stockfish.binary_path if stockfish is not None else None,
        stockfish_nodes_per_move=stockfish.nodes_per_move if stockfish is not None else 1,
        stockfish_threads=stockfish.threads if stockfish is not None else 1,
        stockfish_hash_mib=stockfish.hash_mib if stockfish is not None else 1,
        evaluate_random=chess.evaluate_random,
        search_exploration_constant=chess.exploration_constant,
        parallel_searches=topology.search.parallel_searches,
        direct_inference=topology.inference.direct,
        teacher_searches_per_turn=shared.teacher_searches_per_turn,
        teacher_evaluation_games=shared.teacher_games,
    )


def _build_cluster_parameters(topology: ExperimentTopologyConfiguration) -> ClusterParams:
    training = topology.training
    self_play = topology.self_play
    evaluation = topology.evaluation
    return ClusterParams(
        trainer_device_type=training.device_type,
        trainer_process_group_backend=training.process_group_backend,
        trainer_rank_zero_device_id=training.rank_zero_device_id,
        trainer_ddp_device_ids=training.ddp_device_ids,
        evaluation_device_cycle=evaluation.device_cycle,
        self_play_device_ids=_self_play_device_ids(self_play.processes_per_device),
        self_play_tensorboard_processes=self_play.tensorboard_processes,
        trainer_cpu_threads=training.cpu_threads,
        trainer_interop_threads=training.interop_threads,
        self_play_node_ids_to_pause_during_training=_self_play_node_ids_to_pause(
            self_play.processes_per_device,
            self_play.processes_per_device_during_training,
        ),
        max_concurrent_evaluations=evaluation.maximum_concurrent_evaluations,
    )


def _self_play_device_ids(processes_per_device: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(
        device_id for device_id, process_count in enumerate(processes_per_device) for _ in range(process_count)
    )


def _self_play_node_ids_to_pause(
    processes_per_device: tuple[int, ...],
    processes_per_device_during_training: tuple[int, ...],
) -> tuple[int, ...]:
    node_ids_to_pause: list[int] = []
    first_node_id = 0
    for configured_count, training_count in zip(
        processes_per_device,
        processes_per_device_during_training,
    ):
        node_ids_to_pause.extend(range(first_node_id + training_count, first_node_id + configured_count))
        first_node_id += configured_count
    return tuple(node_ids_to_pause)
