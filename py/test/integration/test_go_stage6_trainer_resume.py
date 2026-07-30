from __future__ import annotations

from pathlib import Path
from uuid import UUID

import torch
from src.az.config.base import DeterminismMode

from src.az.config.seeds import ModelInitializationSeedCoordinates, SeedPurpose, derive_seed
from src.az.config.training import (
    AdamWOptimizerConfiguration,
    ConstantLearningRate,
    ReplayCreditConfiguration,
    TrainingConfiguration,
)
from src.az.games.api import GameIdentifier
from src.az.games.go.module import create_go_training_module
from src.az.games.go.replay_codec import GoReplayCodec
from src.az.replay.credits import ReplayCreditJournal
from src.az.replay.envelope import ReplayRecord
from src.az.replay.storage import ReplayShardStorage
from src.az.training.checkpoints import CheckpointRepository
from src.az.training.distributed import TrainingRank
from src.az.training.optimizer import restore_torch_random_state
from src.az.training.trainer import CreditTrainer
from test.unit.go_stage5_helpers import (
    envelope,
    game_configuration,
    model_configuration,
    objective_configuration,
    sample,
)


ROOT_SEED = 9182
RUN_ID = UUID(int=700)
CONFIGURATION_SHA256 = 'b' * 64


def _training_configuration() -> TrainingConfiguration:
    return TrainingConfiguration(
        global_batch_size=2,
        local_batch_size=2,
        maximum_optimizer_steps=2,
        optimizer=AdamWOptimizerConfiguration(
            kind='adamw',
            learning_rate=0.002,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
            weight_decay=0,
        ),
        learning_rate_schedule=ConstantLearningRate(kind='constant', multiplier=1),
        precision='float32',
        objective=objective_configuration(),
        checkpoint_every_optimizer_steps=1,
        gradient_clip_norm=1,
    )


def _credit_configuration() -> ReplayCreditConfiguration:
    return ReplayCreditConfiguration(
        target_reuse=2,
        optimizer_steps_per_quantum=1,
        minimum_positions_before_training=2,
    )


def _storage(path: Path) -> ReplayShardStorage:
    codec = GoReplayCodec(game_configuration(), 1)
    records = tuple(
        ReplayRecord(
            envelope(index).model_copy(
                update={
                    'sample_id': UUID(int=100 + index),
                    'replay_credit_id': UUID(int=200 + index),
                }
            ),
            codec.encode(sample(value_target=1.0 if index == 1 else -1.0)),
        )
        for index in (1, 2)
    )
    storage = ReplayShardStorage(
        path,
        2,
        4,
        GameIdentifier.GO,
        1,
        'none',
        ReplayCreditJournal(path.parent / 'credit-identities.bin'),
    )
    storage.publish(0, records)
    return storage


def _trainer(root: Path, storage: ReplayShardStorage) -> CreditTrainer:
    model_seed = derive_seed(
        ROOT_SEED,
        ModelInitializationSeedCoordinates(
            purpose=SeedPurpose.MODEL_INITIALIZATION,
            model_stage=0,
        ),
    )
    game_module = create_go_training_module(
        game_configuration=game_configuration(),
        model_configuration=model_configuration(),
        objective_configuration=objective_configuration(),
        payload_schema_version=1,
        device=torch.device('cpu'),
        model_initialization_seed=model_seed,
    )
    return CreditTrainer(
        game_module=game_module,
        replay_storage=storage,
        checkpoint_repository=CheckpointRepository(
            (root / 'checkpoints').resolve(),
            RUN_ID,
            CONFIGURATION_SHA256,
        ),
        training_configuration=_training_configuration(),
        credit_configuration=_credit_configuration(),
        root_seed=ROOT_SEED,
        rank=TrainingRank(rank=0, world_size=1, device=torch.device('cpu')),
        run_determinism_mode=DeterminismMode.SEEDED_CONCURRENT,
    )


def test_cpu_go_training_checkpoint_resume_reproduces_next_update(tmp_path: Path) -> None:
    uninterrupted_root = tmp_path / 'uninterrupted'
    resumed_root = tmp_path / 'resumed'
    uninterrupted = _trainer(uninterrupted_root, _storage(uninterrupted_root / 'replay'))
    before_restart = _trainer(resumed_root, _storage(resumed_root / 'replay'))

    uninterrupted_first = uninterrupted.train_quantum()
    resumed_first = before_restart.train_quantum()
    assert uninterrupted_first.sampled_sample_ids == resumed_first.sampled_sample_ids
    assert uninterrupted_first.augmentation_seeds == resumed_first.augmentation_seeds
    assert uninterrupted_first.loss_values == resumed_first.loss_values

    restore_torch_random_state(uninterrupted_first.checkpoint.torch_random_state_artifact)
    expected_next_random_values = torch.rand(4)
    uninterrupted_next = uninterrupted.train_quantum()
    restarted = _trainer(resumed_root, _storage_from_existing(resumed_root / 'replay'))
    actual_next_random_values = torch.rand(4)
    resumed_next = restarted.train_quantum()

    assert torch.equal(actual_next_random_values, expected_next_random_values)
    assert resumed_next.sampled_sample_ids == uninterrupted_next.sampled_sample_ids
    assert resumed_next.augmentation_seeds == uninterrupted_next.augmentation_seeds
    assert resumed_next.loss_values == uninterrupted_next.loss_values
    assert resumed_next.checkpoint.manifest.state == uninterrupted_next.checkpoint.manifest.state
    assert resumed_next.checkpoint.model_artifact == uninterrupted_next.checkpoint.model_artifact
    assert resumed_next.checkpoint.optimizer_artifact == uninterrupted_next.checkpoint.optimizer_artifact
    assert restarted.credit_state.completed_optimizer_steps == 2
    assert restarted.credit_state.credited_unique_positions == 2
    assert restarted.credit_state.earned_position_credits == 4
    assert restarted.credit_state.consumed_position_credits == 4
    assert restarted.credit_state.available_position_credits == 0
    assert (
        CheckpointRepository(
            (resumed_root / 'checkpoints').resolve(),
            RUN_ID,
            CONFIGURATION_SHA256,
        )
        .load_current()
        .manifest.state.replay_credits
        == restarted.credit_state
    )


def _storage_from_existing(path: Path) -> ReplayShardStorage:
    return ReplayShardStorage(
        path,
        2,
        4,
        GameIdentifier.GO,
        1,
        'none',
        ReplayCreditJournal(path.parent / 'credit-identities.bin'),
    )
