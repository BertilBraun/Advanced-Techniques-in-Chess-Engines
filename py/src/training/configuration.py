from __future__ import annotations

from decimal import Decimal
from enum import Enum
from typing import Literal

from pydantic import Field, model_validator

from src.experiment.cost_accounting import CostCurrency
from src.games.chess.resignation import ResignationParams
from src.util.frozen_model import FrozenModel


class SelfPlaySearchParams(FrozenModel):
    num_searches_per_turn: int = Field(gt=0)
    fast_searches_proportion_of_full_searches: float = Field(gt=0.0, le=1.0)
    playout_cap_randomization: float = Field(ge=0.0, le=1.0)
    num_parallel_searches: int = Field(gt=0)
    dirichlet_epsilon: float = Field(ge=0.0, le=1.0)
    dirichlet_alpha: float = Field(gt=0.0)
    c_param: float = Field(gt=0.0)
    percentage_of_node_visits_to_keep: float = Field(ge=0.0, le=1.0)
    min_visit_count: int = Field(default=0, ge=0)


class BatchedInferenceParams(FrozenModel):
    inference_workers: int = Field(gt=0)
    inference_batch_size: int = Field(gt=0)
    outstanding_batches_per_worker: int = Field(ge=1, le=2)


class SEPlacement(str, Enum):
    DISABLED = 'disabled'
    EVERY_BLOCK = 'every_block'
    EVERY_SECOND_BLOCK = 'every_second_block'

    def applies_to(self, block_index: int) -> bool:
        if self is SEPlacement.DISABLED:
            return False
        if self is SEPlacement.EVERY_BLOCK:
            return True
        return block_index % 2 == 1


class NetworkParams(FrozenModel):
    num_layers: int = Field(gt=0)
    hidden_size: int = Field(gt=0)
    se_placement: SEPlacement = SEPlacement.DISABLED
    num_policy_channels: int = Field(default=4, gt=0)
    num_value_channels: int = Field(default=2, gt=0)
    value_fc_size: int = Field(default=48, gt=0)


class SelfPlayParams(FrozenModel):
    search: SelfPlaySearchParams
    inference: BatchedInferenceParams
    num_moves_after_which_to_play_greedy: int = Field(gt=0)
    maximum_game_plies: int | None = Field(default=None, gt=0)
    maximum_game_plies_until_model_version: int = Field(default=0, ge=0)
    maximum_game_plies_hold_until_model_version: int = Field(default=0, ge=0)
    final_maximum_game_plies: int | None = Field(default=None, gt=0)
    endgame_continuation_start_plies: int | None = Field(default=None, ge=0)
    low_material_termination_minimum_plies: int = Field(default=0, ge=0)
    low_material_termination_piece_threshold_per_player: int = Field(default=0, ge=0)
    low_material_termination_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    starting_temperature: float = Field(default=1.25, gt=0.0)
    final_temperature: float = Field(default=0.1, gt=0.0)
    resignation: ResignationParams = ResignationParams()
    disagreement_prefix_start_probability: float = Field(default=0.15, ge=0.0, le=1.0)
    disagreement_prefix_maximum_ply: int = Field(default=10, ge=0)
    disagreement_prefix_archive_capacity: int = Field(default=2_000, gt=0)
    disagreement_prefix_weight_smoothing: float = Field(default=0.05, gt=0.0)
    disagreement_prefix_weight_cap: float = Field(default=4.0, ge=1.0)
    initial_num_searches_per_turn: int | None = Field(default=None, gt=0)
    search_warmup_model_versions: int = Field(default=0, ge=0)
    endgame_shortcut_fade_model_versions: int = Field(default=0, ge=0)

    @model_validator(mode='after')
    def validate_temperatures(self) -> SelfPlayParams:
        if self.final_temperature > self.starting_temperature:
            raise ValueError('Final self-play temperature cannot exceed the starting temperature.')
        return self


class TrainerTopologyParams(FrozenModel):
    device_type: Literal['cuda', 'cpu']
    process_group_backend: Literal['nccl', 'gloo']
    rank_zero_device_id: int = Field(ge=0)
    ddp_device_ids: tuple[int, ...] = Field(min_length=1)
    cpu_threads: int = Field(gt=0)
    interop_threads: int = Field(gt=0)

    @model_validator(mode='after')
    def validate_devices(self) -> TrainerTopologyParams:
        if self.ddp_device_ids[0] != self.rank_zero_device_id:
            raise ValueError('Rank-zero trainer device must be first in the DDP device list.')
        if len(set(self.ddp_device_ids)) != len(self.ddp_device_ids):
            raise ValueError('Trainer DDP device IDs must be unique.')
        if self.process_group_backend == 'nccl' and self.device_type != 'cuda':
            raise ValueError('NCCL requires CUDA training.')
        if self.device_type == 'cpu' and (self.process_group_backend != 'gloo' or self.ddp_device_ids != (0,)):
            raise ValueError('CPU training requires Gloo on logical device zero.')
        return self


class SelfPlayTopologyParams(FrozenModel):
    device_ids: tuple[int, ...] = Field(min_length=1)
    parallel_games_per_process: int = Field(gt=0)
    tensorboard_processes: int = Field(gt=0)
    node_ids_to_pause_during_training: tuple[int, ...]


class EvaluationTopologyParams(FrozenModel):
    device_cycle: tuple[int, ...] = Field(min_length=1)


class TopologyParams(FrozenModel):
    trainer: TrainerTopologyParams
    self_play: SelfPlayTopologyParams
    evaluation: EvaluationTopologyParams


class RuntimeLimits(FrozenModel):
    cost_currency: CostCurrency
    hourly_price: float = Field(ge=0.0)
    maximum_cost: float | None = Field(default=None, gt=0.0)
    maximum_wall_time_seconds: float = Field(gt=0.0)
    maximum_open_file_count: int = Field(gt=0)
    maximum_host_ram_percent: float = Field(gt=0.0, le=100.0)
    minimum_free_disk_gib: float = Field(ge=0.0)
    resource_telemetry_interval_seconds: float = Field(gt=0.0)


class InferenceRetentionParams(FrozenModel):
    recent_checkpoint_count: int = Field(gt=0)
    milestone_interval: int = Field(gt=0)


OptimizerType = Literal['adamw', 'sgd']


class EvaluationScheduleParams(FrozenModel):
    interval_optimizer_steps: int = Field(gt=0)
    full_interval_optimizer_steps: int = Field(gt=0)
    timeout_seconds: float = Field(gt=0.0)
    maximum_attempts: int = Field(gt=0)
    retry_backoff_seconds: float = Field(ge=0.0)

    def validate_for_optimizer_quantum(self, optimizer_steps_per_quantum: int) -> None:
        if self.interval_optimizer_steps % optimizer_steps_per_quantum:
            raise ValueError('Evaluation interval must align with training quanta.')
        if (
            self.full_interval_optimizer_steps < self.interval_optimizer_steps
            or self.full_interval_optimizer_steps % self.interval_optimizer_steps
        ):
            raise ValueError('Full evaluation interval must be a multiple of the inspection interval.')


class CreditTrainingParams(FrozenModel):
    replay_ratio: Decimal = Field(gt=0)
    optimizer_steps_per_quantum: int = Field(gt=0)
    maximum_optimizer_steps: int = Field(gt=0)
    initial_replay_capacity_unique_positions: int = Field(gt=0)
    maximum_replay_capacity_unique_positions: int = Field(gt=0)
    replay_capacity_ramp_model_versions: int = Field(gt=0)
    retained_checkpoint_interval_steps: int = Field(gt=0)

    @model_validator(mode='after')
    def validate_schedule(self) -> CreditTrainingParams:
        if self.maximum_optimizer_steps % self.optimizer_steps_per_quantum:
            raise ValueError('Maximum optimizer steps must contain complete training quanta.')
        if self.maximum_replay_capacity_unique_positions < self.initial_replay_capacity_unique_positions:
            raise ValueError('Maximum replay capacity must not be smaller than its initial capacity.')
        if self.retained_checkpoint_interval_steps % self.optimizer_steps_per_quantum:
            raise ValueError('Retained checkpoint interval must align with training quanta.')
        return self

    def presentation_credits_per_quantum(self, global_batch_size: int) -> int:
        if global_batch_size <= 0:
            raise ValueError('Global batch size must be positive.')
        return global_batch_size * self.optimizer_steps_per_quantum

    def replay_capacity_for_model_version(self, model_version: int) -> int:
        if model_version < 0:
            raise ValueError('Model version must be nonnegative.')
        completed_ramp_versions = min(model_version, self.replay_capacity_ramp_model_versions)
        capacity_range = self.maximum_replay_capacity_unique_positions - self.initial_replay_capacity_unique_positions
        return self.initial_replay_capacity_unique_positions + (
            capacity_range * completed_ramp_versions // self.replay_capacity_ramp_model_versions
        )

    def unique_samples_per_quantum(self, global_batch_size: int) -> int:
        required_samples = Decimal(self.presentation_credits_per_quantum(global_batch_size)) / self.replay_ratio
        if required_samples != required_samples.to_integral_value():
            raise ValueError('Replay ratio must produce an integral unique-sample quantum.')
        return int(required_samples)


class ModelVersionLearningRateStage(FrozenModel):
    start_model_version: int = Field(ge=0)
    learning_rate: float = Field(gt=0.0)


class ModelVersionLearningRate(FrozenModel):
    stages: tuple[ModelVersionLearningRateStage, ...] = Field(min_length=1)
    optimizer_steps_per_model_version: int = Field(gt=0)

    @model_validator(mode='after')
    def validate_stages(self) -> ModelVersionLearningRate:
        model_versions = tuple(stage.start_model_version for stage in self.stages)
        if model_versions[0] != 0 or tuple(sorted(set(model_versions))) != model_versions:
            raise ValueError('Learning-rate stages must start at model version zero and increase uniquely.')
        return self

    def __call__(self, optimizer_step: int, _: OptimizerType) -> float:
        model_version = optimizer_step // self.optimizer_steps_per_model_version
        selected_stage = self.stages[0]
        for stage in self.stages[1:]:
            if stage.start_model_version > model_version:
                break
            selected_stage = stage
        return selected_stage.learning_rate


class TrainingParams(FrozenModel):
    global_batch_size: int = Field(gt=0)
    local_batch_size: int = Field(gt=0)
    optimizer: OptimizerType
    learning_rate: ModelVersionLearningRate
    max_grad_norm: float = Field(default=0.5, gt=0.0)
    value_loss_weight: float = Field(default=0.5, ge=0.0)
    outcome_value_loss_weight: float = Field(default=0.85, ge=0.0)
    mcts_value_loss_weight: float = Field(default=0.15, ge=0.0)
    mcts_value_target_warmup_optimizer_steps: int = Field(default=0, ge=0)
    duplicate_multiplicity_weight_cap: float | None = Field(default=4.0, ge=1.0)
    policy_loss_weight: float = Field(default=1.0, ge=0.0)

    @model_validator(mode='after')
    def validate_value_weights(self) -> TrainingParams:
        if abs(self.outcome_value_loss_weight + self.mcts_value_loss_weight - 1.0) > 1e-9:
            raise ValueError('Value-objective component weights must sum to 1.')
        return self


class TrainingLifecycleParams(FrozenModel):
    credit: CreditTrainingParams
    evaluation: EvaluationScheduleParams
    inference_retention: InferenceRetentionParams


class TrainingArgs(FrozenModel):
    save_path: str
    network: NetworkParams
    self_play: SelfPlayParams
    trainer: TrainingParams
    topology: TopologyParams
    lifecycle: TrainingLifecycleParams
    limits: RuntimeLimits
    random_seed: int

    @model_validator(mode='after')
    def validate_training(self) -> TrainingArgs:
        world_size = len(self.topology.trainer.ddp_device_ids)
        if self.trainer.global_batch_size != self.trainer.local_batch_size * world_size:
            raise ValueError('Global batch size must equal local batch size times trainer world size.')
        credit = self.lifecycle.credit
        credit.presentation_credits_per_quantum(self.trainer.global_batch_size)
        credit.unique_samples_per_quantum(self.trainer.global_batch_size)
        self.lifecycle.evaluation.validate_for_optimizer_quantum(credit.optimizer_steps_per_quantum)
        return self
