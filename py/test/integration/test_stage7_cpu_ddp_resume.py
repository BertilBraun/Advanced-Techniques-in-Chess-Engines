from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import UUID

import pytest
import torch
import torch.distributed as distributed
import torch.multiprocessing as torch_multiprocessing
from pydantic import Field

from src.az.config.base import FrozenModel
from src.az.config.base import DeterminismMode
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
from src.az.training.distributed import DistributedBackend, TrainingRank
from src.az.training.trainer import CreditTrainer
from test.unit.go_stage5_helpers import (
    envelope,
    game_configuration,
    model_configuration,
    objective_configuration,
    sample,
)


RUN_ID = UUID(int=702)
CONFIGURATION_SHA256 = 'd' * 64


class _RankResult(FrozenModel):
    rank: int = Field(ge=0)
    model_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    completed_steps: int = Field(ge=0)
    resumed_sample_ids: tuple[UUID, ...]


def _rank_main(rank: int, world_size: int, root: str, rendezvous: str) -> None:
    distributed.init_process_group(
        backend='gloo',
        init_method=f'file://{rendezvous}',
        rank=rank,
        world_size=world_size,
    )
    try:
        root_path = Path(root)
        storage = ReplayShardStorage(
            directory=root_path / 'replay',
            maximum_positions_per_shard=2,
            capacity_positions=4,
            game_identifier=GameIdentifier.GO,
            payload_schema_version=1,
            compression='none',
            credit_journal=ReplayCreditJournal(root_path / 'credits.azc'),
        )
        repository = CheckpointRepository(
            (root_path / 'checkpoints').resolve(),
            RUN_ID,
            CONFIGURATION_SHA256,
        )

        def trainer() -> CreditTrainer:
            module = create_go_training_module(
                game_configuration=game_configuration(),
                model_configuration=model_configuration(),
                objective_configuration=objective_configuration(),
                payload_schema_version=1,
                device=torch.device('cpu'),
                model_initialization_seed=23,
            )
            return CreditTrainer(
                game_module=module,
                replay_storage=storage,
                checkpoint_repository=repository,
                training_configuration=TrainingConfiguration(
                    global_batch_size=2,
                    local_batch_size=1,
                    maximum_optimizer_steps=2,
                    optimizer=AdamWOptimizerConfiguration(
                        kind='adamw',
                        learning_rate=0.001,
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
                ),
                credit_configuration=ReplayCreditConfiguration(
                    target_reuse=2,
                    optimizer_steps_per_quantum=1,
                    minimum_positions_before_training=2,
                ),
                root_seed=29,
                rank=TrainingRank(
                    rank=rank,
                    world_size=world_size,
                    device=torch.device('cpu'),
                    backend=DistributedBackend.GLOO,
                ),
                run_determinism_mode=DeterminismMode.SEEDED_CONCURRENT,
            )

        first = trainer()
        first.train_quantum()
        resumed = trainer()
        second = resumed.train_quantum()
        result = _RankResult(
            rank=rank,
            model_sha256=hashlib.sha256(second.checkpoint.model_artifact).hexdigest(),
            completed_steps=resumed.credit_state.completed_optimizer_steps,
            resumed_sample_ids=second.sampled_sample_ids[0],
        )
        (root_path / f'rank-{rank}.json').write_text(result.model_dump_json(), encoding='utf-8')
    finally:
        distributed.destroy_process_group()


@pytest.mark.integration
def test_cpu_gloo_ddp_checkpoint_restores_rank_lifecycle_before_next_quantum(
    tmp_path: Path,
) -> None:
    codec = GoReplayCodec(game_configuration(), 1)
    records = tuple(
        ReplayRecord(
            envelope=envelope(index).model_copy(
                update={
                    'sample_id': UUID(int=300 + index),
                    'replay_credit_id': UUID(int=400 + index),
                }
            ),
            payload=codec.encode(sample(value_target=1.0 if index == 1 else -1.0)),
        )
        for index in (1, 2)
    )
    ReplayShardStorage(
        directory=tmp_path / 'replay',
        maximum_positions_per_shard=2,
        capacity_positions=4,
        game_identifier=GameIdentifier.GO,
        payload_schema_version=1,
        compression='none',
        credit_journal=ReplayCreditJournal(tmp_path / 'credits.azc'),
    ).publish(0, records)
    rendezvous = tmp_path / 'gloo-rendezvous'
    torch_multiprocessing.spawn(
        _rank_main,
        args=(2, str(tmp_path), str(rendezvous)),
        nprocs=2,
        join=True,
    )

    results = tuple(_RankResult.model_validate_json((tmp_path / f'rank-{rank}.json').read_bytes()) for rank in range(2))
    assert results[0].model_sha256 == results[1].model_sha256
    assert all(result.completed_steps == 2 for result in results)
    assert all(result.resumed_sample_ids for result in results)
    for rank in range(2):
        checkpoint = CheckpointRepository(
            (tmp_path / 'checkpoints').resolve(),
            RUN_ID,
            CONFIGURATION_SHA256,
        ).load_distributed(rank)
        assert checkpoint.rank.state.process_group.rank == rank
        assert checkpoint.rank.state.process_group.world_size == 2
        assert checkpoint.rank.state.process_group.initialized
