from __future__ import annotations

from decimal import Decimal
from enum import Enum
from typing import Literal

from pydantic import Field, model_validator

from src.experiment.generation_schedule import (
    FloatGenerationSchedule,
    IntegerGenerationSchedule,
    defined_schedule_values,
)
from src.self_play.parameters import ResolvedSelfPlayParameters
from src.training.targets import AuxiliaryTargetConfiguration
from src.util.frozen_model import FrozenModel


class SelfPlaySearchParams(FrozenModel):
    full_searches: IntegerGenerationSchedule
    fast_searches: IntegerGenerationSchedule
    parallel_searches: int = Field(gt=0)
    dirichlet_epsilon: FloatGenerationSchedule
    dirichlet_alpha: FloatGenerationSchedule
    exploration_constant: FloatGenerationSchedule
    minimum_root_visits: IntegerGenerationSchedule

    @model_validator(mode='after')
    def validate_scheduled_values(self) -> SelfPlaySearchParams:
        if any(value <= self.parallel_searches for value in defined_schedule_values(self.full_searches)):
            raise ValueError('Every full-search budget must exceed the parallel-search count.')
        if any(value <= 0 for value in defined_schedule_values(self.fast_searches)):
            raise ValueError('Every fast-search budget must be positive.')
        if any(not 0.0 <= value <= 1.0 for value in defined_schedule_values(self.dirichlet_epsilon)):
            raise ValueError('Dirichlet epsilon must remain in [0, 1].')
        if any(value <= 0.0 for value in defined_schedule_values(self.dirichlet_alpha)):
            raise ValueError('Dirichlet alpha must remain positive.')
        if any(value <= 0.0 for value in defined_schedule_values(self.exploration_constant)):
            raise ValueError('Exploration constant must remain positive.')
        if any(value < 0 for value in defined_schedule_values(self.minimum_root_visits)):
            raise ValueError('Minimum root visits must remain nonnegative.')
        return self


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


class SelfPlayConfiguration(FrozenModel):
    search: SelfPlaySearchParams
    inference: BatchedInferenceParams
    random_opening_plies: IntegerGenerationSchedule
    full_search_probability: FloatGenerationSchedule
    retained_root_visit_fraction: FloatGenerationSchedule
    greedy_after_ply: IntegerGenerationSchedule
    starting_temperature: FloatGenerationSchedule
    final_temperature: FloatGenerationSchedule
    primary_sample_weight: FloatGenerationSchedule
    detailed_statistics_workers: int = Field(default=1, ge=0)

    @model_validator(mode='after')
    def validate_temperatures(self) -> SelfPlayConfiguration:
        for schedule, name in (
            (self.starting_temperature, 'Starting temperature'),
            (self.final_temperature, 'Final temperature'),
        ):
            if any(value <= 0.0 for value in defined_schedule_values(schedule)):
                raise ValueError(f'{name} must remain positive.')
        if any(value < 0 for value in defined_schedule_values(self.random_opening_plies)):
            raise ValueError('Random opening plies must remain nonnegative.')
        if any(value <= 0 for value in defined_schedule_values(self.greedy_after_ply)):
            raise ValueError('Greedy ply must remain positive.')
        if any(not 0.0 < value <= 1.0 for value in defined_schedule_values(self.full_search_probability)):
            raise ValueError('Full-search probability must remain in (0, 1].')
        if any(not 0.0 <= value <= 1.0 for value in defined_schedule_values(self.retained_root_visit_fraction)):
            raise ValueError('Retained-root fraction must remain in [0, 1].')
        if any(value <= 0.0 for value in defined_schedule_values(self.primary_sample_weight)):
            raise ValueError('Primary sample weight must remain positive.')
        return self

    def resolve(
        self,
        model_generation: int,
        maximum_game_plies: int | None,
    ) -> ResolvedSelfPlayParameters:
        search = self.search
        return ResolvedSelfPlayParameters(
            random_opening_plies=self.random_opening_plies.value_at(model_generation),
            full_search_probability=self.full_search_probability.value_at(model_generation),
            parallel_searches=search.parallel_searches,
            full_searches=search.full_searches.value_at(model_generation),
            fast_searches=search.fast_searches.value_at(model_generation),
            minimum_root_visits=search.minimum_root_visits.value_at(model_generation),
            exploration_constant=search.exploration_constant.value_at(model_generation),
            dirichlet_alpha=search.dirichlet_alpha.value_at(model_generation),
            dirichlet_epsilon=search.dirichlet_epsilon.value_at(model_generation),
            retained_root_visit_fraction=self.retained_root_visit_fraction.value_at(model_generation),
            starting_temperature=self.starting_temperature.value_at(model_generation),
            final_temperature=self.final_temperature.value_at(model_generation),
            greedy_after_ply=self.greedy_after_ply.value_at(model_generation),
            maximum_game_plies=maximum_game_plies,
            primary_sample_weight=self.primary_sample_weight.value_at(model_generation),
        )


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


class ReplayConfiguration(FrozenModel):
    capacity: IntegerGenerationSchedule
    maximum_capacity: int = Field(gt=0)
    maximum_policy_entries: int = Field(ge=1, le=255)

    @model_validator(mode='after')
    def validate_capacity(self) -> ReplayConfiguration:
        capacities = defined_schedule_values(self.capacity)
        if any(capacity <= 0 for capacity in capacities):
            raise ValueError('Replay capacity must remain positive.')
        if any(capacity > self.maximum_capacity for capacity in capacities):
            raise ValueError('Scheduled replay capacity cannot exceed its static maximum capacity.')
        return self

    def capacity_at(self, model_generation: int) -> int:
        return self.capacity.value_at(model_generation)


class CreditTrainingParams(FrozenModel):
    replay_ratio: Decimal = Field(gt=0)
    optimizer_steps_per_quantum: int = Field(gt=0)
    maximum_optimizer_steps: int = Field(gt=0)
    retained_checkpoint_interval_generations: int = Field(gt=0)

    @model_validator(mode='after')
    def validate_schedule(self) -> CreditTrainingParams:
        if self.maximum_optimizer_steps % self.optimizer_steps_per_quantum:
            raise ValueError('Maximum optimizer steps must contain complete training quanta.')
        return self

    def presentation_credits_per_quantum(self, global_batch_size: int) -> int:
        if global_batch_size <= 0:
            raise ValueError('Global batch size must be positive.')
        return global_batch_size * self.optimizer_steps_per_quantum

    def unique_samples_per_quantum(self, global_batch_size: int) -> int:
        required_samples = Decimal(self.presentation_credits_per_quantum(global_batch_size)) / self.replay_ratio
        if required_samples != required_samples.to_integral_value():
            raise ValueError('Replay ratio must produce an integral unique-sample quantum.')
        return int(required_samples)


OptimizerType = Literal['adamw', 'sgd']


class TrainingParams(FrozenModel):
    global_batch_size: int = Field(gt=0)
    local_batch_size: int = Field(gt=0)
    optimizer: OptimizerType
    learning_rate: FloatGenerationSchedule
    max_grad_norm: float = Field(default=0.5, gt=0.0)
    duplicate_multiplicity_weight_cap: float | None = Field(default=None, ge=1.0)


class TrainingObjectiveConfiguration(FrozenModel):
    policy_loss_weight: FloatGenerationSchedule
    value_loss_weight: FloatGenerationSchedule
    root_value_blend: FloatGenerationSchedule
    auxiliary_targets: tuple[AuxiliaryTargetConfiguration, ...] = ()

    @model_validator(mode='after')
    def validate_scheduled_weights(self) -> TrainingObjectiveConfiguration:
        for schedule, name in (
            (self.policy_loss_weight, 'Policy loss weight'),
            (self.value_loss_weight, 'Value loss weight'),
        ):
            if any(value < 0.0 for value in defined_schedule_values(schedule)):
                raise ValueError(f'{name} must remain nonnegative.')
        if any(not 0.0 <= value <= 1.0 for value in defined_schedule_values(self.root_value_blend)):
            raise ValueError('Root-value blend must remain in [0, 1].')
        return self


class TrainingLifecycleParams(FrozenModel):
    replay: ReplayConfiguration
    credit: CreditTrainingParams
    inference_retention: InferenceRetentionParams


class TrainingArgs(FrozenModel):
    save_path: str
    network: NetworkParams
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
        if any(
            capacity < self.trainer.global_batch_size
            for capacity in defined_schedule_values(self.lifecycle.replay.capacity)
        ):
            raise ValueError('Every scheduled replay capacity must contain at least one global batch.')
        if any(value <= 0.0 for value in defined_schedule_values(self.trainer.learning_rate)):
            raise ValueError('Learning rate must remain positive.')
        maximum_generation = credit.maximum_optimizer_steps // credit.optimizer_steps_per_quantum
        if maximum_generation > 4_294_967_295:
            raise ValueError('Maximum model generation must fit uint32 replay metadata.')
        return self

    def validate_game(self, action_size: int, self_play: SelfPlayConfiguration) -> None:
        if action_size > 65_536:
            raise ValueError('Game action IDs must fit uint16 replay storage.')
        if self.lifecycle.replay.maximum_policy_entries > action_size:
            raise ValueError('Maximum retained policy entries cannot exceed the game action count.')
        search = self_play.search
        search_budgets = (
            *defined_schedule_values(search.full_searches),
            *defined_schedule_values(search.fast_searches),
        )
        if any(search_budget > 65_535 for search_budget in search_budgets):
            raise ValueError('Configured search visits must fit uint16 replay storage.')
