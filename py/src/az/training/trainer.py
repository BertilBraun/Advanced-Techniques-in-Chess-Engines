from __future__ import annotations

import math
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from decimal import Decimal
from typing import Generic, TypeVar
from uuid import UUID

import torch

from src.az.config.training import ReplayCreditConfiguration, TrainingConfiguration
from src.az.config.seeds import SeedPurpose, TrainingRandomSeedCoordinates, derive_seed
from src.az.games.training import GameTrainingModule
from src.az.replay.credits import ReplayCreditSnapshot, ReplayCreditState
from src.az.replay.envelope import ReplayRecord
from src.az.replay.sampling import DeterministicReplaySampler, ReplaySamplerState
from src.az.replay.storage import ReplayShardStorage
from src.az.training.checkpoints import (
    CheckpointPurpose,
    CheckpointRepository,
    LoadedCheckpoint,
    TrainerCheckpointState,
)
from src.az.training.distributed import TrainingRank
from src.az.training.optimizer import (
    LearningRateController,
    create_optimizer,
    optimizer_base_learning_rate,
    restore_optimizer,
    restore_torch_random_state,
    serialize_optimizer,
    serialize_torch_random_state,
)


BatchType = TypeVar('BatchType')


@dataclass(frozen=True)
class TrainingQuantumResult:
    checkpoint: LoadedCheckpoint
    loss_values: tuple[float, ...]
    sampled_sample_ids: tuple[tuple[UUID, ...], ...]
    augmentation_seeds: tuple[tuple[int, ...], ...]


class CreditTrainer(Generic[BatchType]):
    def __init__(
        self,
        game_module: GameTrainingModule[BatchType, ReplayRecord],
        replay_storage: ReplayShardStorage,
        checkpoint_repository: CheckpointRepository,
        training_configuration: TrainingConfiguration,
        credit_configuration: ReplayCreditConfiguration,
        root_seed: int,
        rank: TrainingRank,
    ) -> None:
        if rank.world_size != 1:
            raise ValueError('Stage 6 trainer execution supports one rank; multi-process lifecycle is Stage 7.')
        if rank.device.type != 'cpu':
            raise ValueError('Stage 6 trainer execution is CPU-only; GPU process lifecycle is Stage 7.')
        if training_configuration.global_batch_size != training_configuration.local_batch_size:
            raise ValueError('Single-rank training requires equal global and local batch sizes.')
        self._game_module = game_module
        self._replay_storage = replay_storage
        self._credit_journal = replay_storage.credit_journal
        self._checkpoint_repository = checkpoint_repository
        self._training = training_configuration
        self._credits = credit_configuration
        self._device_type = rank.device.type
        self._target_reuse = Decimal(str(credit_configuration.target_reuse))
        self._requires_restart = False
        self._optimizer = create_optimizer(game_module.model, training_configuration.optimizer)
        if checkpoint_repository.has_current():
            loaded = checkpoint_repository.load_current()
            game_module.restore_model(loaded.model_artifact)
            restore_optimizer(self._optimizer, loaded.optimizer_artifact)
            restore_torch_random_state(loaded.torch_random_state_artifact)
            self._ledger = loaded.manifest.state.replay_credits
            sampler_state = loaded.manifest.state.replay_sampler
            learning_rate_state = loaded.manifest.state.learning_rate
        else:
            self._ledger = ReplayCreditState.initial()
            sampler_state = ReplaySamplerState(next_optimizer_step=0)
            learning_rate_state = None
            torch.manual_seed(
                derive_seed(
                    root_seed,
                    TrainingRandomSeedCoordinates(
                        purpose=SeedPurpose.TRAINING_RANDOM,
                        trainer_rank=rank.rank,
                    ),
                )
            )
        self._sampler = DeterministicReplaySampler(root_seed, rank.rank, sampler_state)
        self._learning_rate = LearningRateController(
            self._optimizer,
            optimizer_base_learning_rate(training_configuration.optimizer),
            training_configuration.learning_rate_schedule,
            learning_rate_state,
        )
        self._validate_restored_state()

    @property
    def credit_state(self) -> ReplayCreditState:
        return self._ledger

    @property
    def sampler_state(self) -> ReplaySamplerState:
        return self._sampler.state

    def train_quantum(self) -> TrainingQuantumResult:
        if self._requires_restart:
            raise RuntimeError(
                'Trainer state may have advanced without publication; restart from the current checkpoint.'
            )
        population = tuple(self._replay_storage.records())
        if len(population) < self._credits.minimum_positions_before_training:
            raise ValueError('Replay population has not reached the minimum size for training.')
        reconciled = self._ledger.reconcile(self._credit_journal.snapshot, self._target_reuse)
        remaining_steps = self._training.maximum_optimizer_steps - reconciled.completed_optimizer_steps
        if remaining_steps <= 0:
            raise ValueError('Training has reached the configured optimizer-step limit.')
        quantum_steps = min(self._credits.optimizer_steps_per_quantum, remaining_steps)
        prepared = reconciled.prepare_training_quantum(
            optimizer_steps=quantum_steps,
            global_batch_size=self._training.global_batch_size,
            maximum_optimizer_steps=self._training.maximum_optimizer_steps,
        )
        loss_values: list[float] = []
        sampled_sample_ids: list[tuple[UUID, ...]] = []
        augmentation_seeds: list[tuple[int, ...]] = []
        self._game_module.model.train()
        self._requires_restart = True
        for _ in range(quantum_steps):
            sampled = self._sampler.sample(population, self._training.local_batch_size)
            batch = self._game_module.move_batch(
                self._game_module.create_training_batch(sampled.records, sampled.augmentation_seeds)
            )
            self._optimizer.zero_grad(set_to_none=True)
            precision_context = self._precision_context()
            with precision_context:
                loss = self._game_module.calculate_loss(batch)
            if not torch.isfinite(loss.total):
                raise ValueError('Training loss must be finite.')
            loss.total.backward()
            torch.nn.utils.clip_grad_norm_(
                self._game_module.model.parameters(),
                self._training.gradient_clip_norm,
                error_if_nonfinite=True,
            )
            self._optimizer.step()
            self._learning_rate.advance()
            loss_values.append(float(loss.total.detach().cpu().item()))
            sampled_sample_ids.append(tuple(record.envelope.sample_id for record in sampled.records))
            augmentation_seeds.append(sampled.augmentation_seeds)
        checkpoint_state = TrainerCheckpointState(
            replay_credits=prepared,
            replay_sampler=self._sampler.state,
            learning_rate=self._learning_rate.state,
        )
        checkpoint = self._checkpoint_repository.publish(
            state=checkpoint_state,
            purpose=self._checkpoint_purpose(prepared),
            model_artifact=self._game_module.serialize_model(),
            optimizer_artifact=serialize_optimizer(self._optimizer),
            torch_random_state_artifact=serialize_torch_random_state(),
        )
        self._ledger = prepared
        self._requires_restart = False
        return TrainingQuantumResult(
            checkpoint=checkpoint,
            loss_values=tuple(loss_values),
            sampled_sample_ids=tuple(sampled_sample_ids),
            augmentation_seeds=tuple(augmentation_seeds),
        )

    def _precision_context(self) -> AbstractContextManager[None]:
        match self._training.precision:
            case 'float32':
                return nullcontext()
            case 'bfloat16':
                return torch.autocast(device_type=self._device_type, dtype=torch.bfloat16)
            case 'float16':
                raise ValueError('CPU training does not support configured float16 precision.')

    def _validate_restored_state(self) -> None:
        steps = self._ledger.completed_optimizer_steps
        if self._sampler.state.next_optimizer_step != steps:
            raise ValueError('Restored replay sampler does not match the credit ledger.')
        if self._learning_rate.state.completed_optimizer_steps != steps:
            raise ValueError('Restored learning-rate schedule does not match the credit ledger.')
        if self._ledger.earned_position_credits != (
            Decimal(self._ledger.credited_unique_positions) * self._target_reuse
        ):
            raise ValueError('Restored replay credit earnings do not match configured target reuse.')
        if not math.isfinite(self._learning_rate.state.current_learning_rate):
            raise ValueError('Restored learning rate must be finite.')
        self._credit_journal.verify_snapshot(
            ReplayCreditSnapshot(
                credited_unique_positions=self._ledger.credited_unique_positions,
                prefix_sha256=self._ledger.credit_journal_prefix_sha256,
            )
        )

    def _checkpoint_purpose(self, prepared: ReplayCreditState) -> CheckpointPurpose:
        if prepared.completed_optimizer_steps == self._training.maximum_optimizer_steps:
            return CheckpointPurpose.FINAL
        if prepared.completed_optimizer_steps % self._training.checkpoint_every_optimizer_steps == 0:
            return CheckpointPurpose.SCHEDULED
        return CheckpointPurpose.CREDIT_COMMIT
