from __future__ import annotations

import math
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from decimal import Decimal
from typing import Generic, TypeVar
from uuid import UUID

import torch
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel

from src.az.config.training import ReplayCreditConfiguration, TrainingConfiguration
from src.az.config.base import DeterminismMode as RunDeterminismMode
from src.az.config.seeds import SeedPurpose, TrainingRandomSeedCoordinates, derive_seed
from src.az.games.training import GameTrainingModule
from src.az.replay.credits import ReplayCreditSnapshot, ReplayCreditState
from src.az.replay.envelope import ReplayRecord
from src.az.replay.sampling import DeterministicReplaySampler, ReplaySamplerState
from src.az.replay.storage import IncrementalReplayCatalog, ReplayShardStorage
from src.az.training.checkpoints import (
    CheckpointPurpose,
    CheckpointRepository,
    LoadedCheckpoint,
    LoadedDistributedCheckpoint,
    TrainerCheckpointState,
)
from src.az.training.distributed import TrainingRank
from src.az.training.optimizer import (
    LearningRateController,
    create_optimizer,
    optimizer_base_learning_rate,
    restore_optimizer,
    restore_assigned_cuda_random_state,
    restore_gradient_scaler,
    restore_torch_random_state,
    serialize_assigned_cuda_random_state,
    serialize_gradient_scaler,
    serialize_optimizer,
    serialize_torch_random_state,
)


BatchType = TypeVar('BatchType')


@dataclass(frozen=True)
class TrainingQuantumResult:
    checkpoint: LoadedCheckpoint | LoadedDistributedCheckpoint
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
        run_determinism_mode: RunDeterminismMode,
    ) -> None:
        if training_configuration.global_batch_size != training_configuration.local_batch_size * rank.world_size:
            raise ValueError('Global batch size must equal local batch size times trainer world size.')
        self._game_module = game_module
        self._replay_storage = replay_storage
        self._credit_journal = replay_storage.credit_journal
        self._checkpoint_repository = checkpoint_repository
        self._training = training_configuration
        self._credits = credit_configuration
        self._device_type = rank.device.type
        self._rank = rank
        self._training_determinism = rank.training_determinism(run_determinism_mode)
        self._process_group = rank.lifecycle()
        self._target_reuse = Decimal(str(credit_configuration.target_reuse))
        self._requires_restart = False
        self._optimizer = create_optimizer(game_module.model, training_configuration.optimizer)
        self._scaler = torch.amp.GradScaler(
            device=rank.device.type,
            enabled=training_configuration.precision == 'float16',
        )
        if rank.world_size > 1:
            device_ids = [rank.device.index] if rank.device.type == 'cuda' else None
            self._training_model: torch.nn.Module = DistributedDataParallel(
                game_module.model,
                device_ids=device_ids,
            )
        else:
            self._training_model = game_module.model
        self._catalog = IncrementalReplayCatalog(replay_storage)
        if checkpoint_repository.has_current():
            loaded = (
                checkpoint_repository.load_distributed(rank.rank)
                if rank.world_size > 1
                else checkpoint_repository.load_current()
            )
            game_module.restore_model(loaded.model_artifact)
            restore_optimizer(self._optimizer, loaded.optimizer_artifact)
            restore_torch_random_state(loaded.torch_random_state_artifact)
            restore_assigned_cuda_random_state(
                loaded.cuda_random_stream_artifact,
                rank.device,
            )
            restore_gradient_scaler(self._scaler, loaded.gradient_scaler_artifact)
            match loaded:
                case LoadedDistributedCheckpoint(rank=rank_checkpoint):
                    restored_state = rank_checkpoint.state
                case LoadedCheckpoint(manifest=manifest):
                    restored_state = manifest.state
            if restored_state.process_group != self._process_group:
                raise ValueError('Checkpoint process-group lifecycle does not match this trainer rank.')
            if restored_state.training_determinism != self._training_determinism:
                raise ValueError('Checkpoint determinism mode does not match this trainer rank.')
            self._ledger = restored_state.replay_credits
            sampler_state = restored_state.replay_sampler
            learning_rate_state = restored_state.learning_rate
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
        population = self._catalog.refresh()
        if population.position_count < self._credits.minimum_positions_before_training:
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
            sampled = self._sampler.sample_catalog(self._catalog, self._training.local_batch_size)
            batch = self._game_module.move_batch(
                self._game_module.create_training_batch(sampled.records, sampled.augmentation_seeds)
            )
            self._optimizer.zero_grad(set_to_none=True)
            precision_context = self._precision_context()
            with precision_context:
                loss = self._game_module.calculate_loss_with_model(batch, self._training_model)
            if not torch.isfinite(loss.total):
                raise ValueError('Training loss must be finite.')
            self._scaler.scale(loss.total).backward()
            self._scaler.unscale_(self._optimizer)
            torch.nn.utils.clip_grad_norm_(
                self._game_module.model.parameters(),
                self._training.gradient_clip_norm,
                error_if_nonfinite=True,
            )
            self._scaler.step(self._optimizer)
            self._scaler.update()
            self._learning_rate.advance()
            loss_values.append(float(loss.total.detach().cpu().item()))
            sampled_sample_ids.append(tuple(record.envelope.sample_id for record in sampled.records))
            augmentation_seeds.append(sampled.augmentation_seeds)
        checkpoint_state = TrainerCheckpointState(
            replay_credits=prepared,
            replay_sampler=self._sampler.state,
            learning_rate=self._learning_rate.state,
            process_group=self._process_group,
            training_determinism=self._training_determinism,
        )
        model_artifact = self._game_module.serialize_model()
        optimizer_artifact = serialize_optimizer(self._optimizer)
        torch_random_state_artifact = serialize_torch_random_state()
        cuda_random_stream_artifact = serialize_assigned_cuda_random_state(self._rank.device)
        gradient_scaler_artifact = serialize_gradient_scaler(self._scaler)
        purpose = self._checkpoint_purpose(prepared)
        if self._rank.world_size == 1:
            checkpoint: LoadedCheckpoint | LoadedDistributedCheckpoint = self._checkpoint_repository.publish(
                state=checkpoint_state,
                purpose=purpose,
                model_artifact=model_artifact,
                optimizer_artifact=optimizer_artifact,
                torch_random_state_artifact=torch_random_state_artifact,
                cuda_random_stream_artifact=cuda_random_stream_artifact,
                gradient_scaler_artifact=gradient_scaler_artifact,
            )
        else:
            staging_error: Exception | None = None
            try:
                self._checkpoint_repository.stage_distributed_rank(
                    state=checkpoint_state,
                    model_artifact=model_artifact,
                    optimizer_artifact=optimizer_artifact,
                    torch_random_state_artifact=torch_random_state_artifact,
                    cuda_random_stream_artifact=cuda_random_stream_artifact,
                    gradient_scaler_artifact=gradient_scaler_artifact,
                )
            except Exception as error:
                staging_error = error
            self._require_successful_distributed_checkpoint_phase('staging', staging_error)
            commit_error: Exception | None = None
            if self._rank.rank == 0:
                try:
                    self._checkpoint_repository.commit_distributed_generation(
                        state=checkpoint_state,
                        purpose=purpose,
                        model_artifact=model_artifact,
                        optimizer_artifact=optimizer_artifact,
                        gradient_scaler_artifact=gradient_scaler_artifact,
                    )
                except Exception as error:
                    commit_error = error
            self._require_successful_distributed_checkpoint_phase('commit', commit_error)
            checkpoint = self._checkpoint_repository.load_distributed(self._rank.rank)
        self._ledger = prepared
        self._requires_restart = False
        return TrainingQuantumResult(
            checkpoint=checkpoint,
            loss_values=tuple(loss_values),
            sampled_sample_ids=tuple(sampled_sample_ids),
            augmentation_seeds=tuple(augmentation_seeds),
        )

    def _require_successful_distributed_checkpoint_phase(
        self,
        phase: str,
        local_error: Exception | None,
    ) -> None:
        phase_status = torch.tensor(
            1 if local_error is None else 0,
            dtype=torch.int32,
            device=self._rank.device,
        )
        distributed.all_reduce(phase_status, op=distributed.ReduceOp.MIN)
        if int(phase_status.item()) == 1:
            return
        if local_error is not None:
            raise RuntimeError(f'Distributed checkpoint {phase} failed on this rank.') from local_error
        raise RuntimeError(f'Distributed checkpoint {phase} failed on another rank.')

    def _precision_context(self) -> AbstractContextManager[None]:
        match self._training.precision:
            case 'float32':
                return nullcontext()
            case 'bfloat16':
                return torch.autocast(device_type=self._device_type, dtype=torch.bfloat16)
            case 'float16':
                if self._device_type != 'cuda':
                    raise ValueError('Float16 training requires CUDA.')
                return torch.autocast(device_type='cuda', dtype=torch.float16)

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
