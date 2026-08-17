from __future__ import annotations

from decimal import Decimal
from enum import Enum
from typing import Literal

from pydantic import Field, model_validator

from src.experiment.generation_schedule import (
    FloatGenerationSchedule,
    defined_schedule_values,
)
from src.replay.configuration import ReplayConfiguration
from src.self_play.configuration import SelfPlayConfiguration
from src.training.run_limits import RuntimeLimits
from src.training.network import NetworkConfiguration
from src.training.targets import AuxiliaryTargetConfiguration
from src.util.frozen_model import FrozenModel


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

    @model_validator(mode='after')
    def validate_worker_assignments(self) -> SelfPlayTopologyParams:
        if self.tensorboard_processes > len(self.device_ids):
            raise ValueError('TensorBoard process count cannot exceed the self-play worker count.')
        if len(set(self.node_ids_to_pause_during_training)) != len(self.node_ids_to_pause_during_training):
            raise ValueError('Self-play worker IDs paused during training must be unique.')
        if any(
            worker_id < 0 or worker_id >= len(self.device_ids) for worker_id in self.node_ids_to_pause_during_training
        ):
            raise ValueError('Self-play worker ID paused during training is outside the configured worker range.')
        return self


class EvaluationTopologyParams(FrozenModel):
    device_cycle: tuple[int, ...] = Field(min_length=1)


class TopologyParams(FrozenModel):
    trainer: TrainerTopologyParams
    self_play: SelfPlayTopologyParams
    evaluation: EvaluationTopologyParams


class InferenceRetentionParams(FrozenModel):
    recent_checkpoint_count: int = Field(gt=0)
    milestone_interval: int = Field(gt=0)


class CreditTrainingParams(FrozenModel):
    replay_ratio: Decimal = Field(gt=0)
    optimizer_steps_per_quantum: int = Field(gt=0)
    maximum_optimizer_steps: int | None = Field(default=None, gt=0)
    retained_checkpoint_interval_generations: int = Field(gt=0)
    self_play_backpressure_quanta: int = Field(gt=0)

    @model_validator(mode='after')
    def validate_schedule(self) -> CreditTrainingParams:
        if self.maximum_optimizer_steps is not None and self.maximum_optimizer_steps % self.optimizer_steps_per_quantum:
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

    def requires_self_play_backpressure(self, available_credits: Decimal, global_batch_size: int) -> bool:
        threshold = Decimal(
            self.presentation_credits_per_quantum(global_batch_size) * self.self_play_backpressure_quanta
        )
        return available_credits > threshold


OptimizerType = Literal['adamw', 'sgd']


class TrainingPrecision(str, Enum):
    FLOAT32 = 'float32'
    BFLOAT16 = 'bfloat16'


class TrainingCompilation(str, Enum):
    DISABLED = 'disabled'
    DEFAULT = 'default'


class TrainingParams(FrozenModel):
    global_batch_size: int = Field(gt=0)
    local_batch_size: int = Field(gt=0)
    optimizer: OptimizerType
    precision: TrainingPrecision
    compilation: TrainingCompilation
    learning_rate: FloatGenerationSchedule
    max_grad_norm: float = Field(default=0.5, gt=0.0)
    duplicate_multiplicity_weight_cap: float | None = Field(default=None, ge=1.0)


class TrainingObjectiveConfiguration(FrozenModel):
    policy_loss_weight: FloatGenerationSchedule
    value_loss_weight: FloatGenerationSchedule
    root_value_blend: FloatGenerationSchedule
    value_discount_per_ply: FloatGenerationSchedule
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
        if any(not 0.0 < value <= 1.0 for value in defined_schedule_values(self.value_discount_per_ply)):
            raise ValueError('Value discount per ply must remain in (0, 1].')
        return self


class TrainingLifecycleParams(FrozenModel):
    replay: ReplayConfiguration
    credit: CreditTrainingParams
    inference_retention: InferenceRetentionParams


class TrainingArgs(FrozenModel):
    save_path: str
    network: NetworkConfiguration
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
        if self.trainer.precision is TrainingPrecision.BFLOAT16 and self.topology.trainer.device_type != 'cuda':
            raise ValueError('Bfloat16 mixed-precision training requires CUDA.')
        if credit.maximum_optimizer_steps is not None:
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
