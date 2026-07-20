from collections.abc import Sized

import pytest
import torch

from src.train.DistributedTraining import DistributedTrainingBatchSampler, distributed_epoch_seed
from src.train.TrainingStats import TrainingStats


class FixedSizeDataset(Sized):
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size


def rank_batches(
    dataset_size: int,
    global_batch_size: int,
    local_batch_size: int,
    world_size: int,
    seed: int,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    return tuple(
        tuple(
            tuple(batch)
            for batch in DistributedTrainingBatchSampler(
                dataset=FixedSizeDataset(dataset_size),
                global_batch_size=global_batch_size,
                local_batch_size=local_batch_size,
                rank=rank,
                world_size=world_size,
                seed=seed,
            )
        )
        for rank in range(world_size)
    )


@pytest.mark.parametrize('dataset_size', (7, 16, 21))
def test_rank_partitions_cover_only_complete_global_batches(dataset_size: int) -> None:
    global_batch_size = 8
    local_batch_size = 2
    world_size = 4
    seed = 91
    batches = rank_batches(
        dataset_size,
        global_batch_size,
        local_batch_size,
        world_size,
        seed,
    )
    expected_steps = dataset_size // global_batch_size

    assert {len(rank_partition) for rank_partition in batches} == {expected_steps}
    assert all(len(batch) == local_batch_size for rank_partition in batches for batch in rank_partition)

    reconstructed_order = tuple(
        sample_index
        for step in range(expected_steps)
        for rank in range(world_size)
        for sample_index in batches[rank][step]
    )
    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed)
    expected_order = tuple(
        torch.randperm(dataset_size, generator=generator)[: expected_steps * global_batch_size].tolist()
    )

    assert reconstructed_order == expected_order
    assert len(reconstructed_order) == len(set(reconstructed_order))
    rank_samples = tuple(
        {sample_index for batch in rank_partition for sample_index in batch} for rank_partition in batches
    )
    for first_rank in range(world_size):
        for second_rank in range(first_rank + 1, world_size):
            assert rank_samples[first_rank].isdisjoint(rank_samples[second_rank])


def test_rank_partitions_are_deterministic_and_change_with_epoch_seed() -> None:
    first_seed = distributed_epoch_seed(17, iteration=3, epoch=2)
    second_seed = distributed_epoch_seed(17, iteration=3, epoch=3)

    first = rank_batches(64, 8, 2, 4, first_seed)
    repeated = rank_batches(64, 8, 2, 4, first_seed)
    next_epoch = rank_batches(64, 8, 2, 4, second_seed)

    assert first == repeated
    assert first != next_epoch


def test_training_stats_are_sample_weighted_with_unbiased_value_deviation() -> None:
    first = TrainingStats(
        policy_loss_sum=2.0,
        value_loss_sum=4.0,
        total_loss_sum=6.0,
        sample_count=2,
        value_sum=0.0,
        value_square_sum=2.0,
        gradient_norm_sum=0.5,
        gradient_norm_count=1,
        num_batches=1,
    )
    second = TrainingStats(
        policy_loss_sum=12.0,
        value_loss_sum=18.0,
        total_loss_sum=30.0,
        sample_count=6,
        value_sum=2.0,
        value_square_sum=6.0,
        gradient_norm_sum=1.5,
        gradient_norm_count=1,
        num_batches=1,
    )

    combined = TrainingStats.combine([first, second])

    assert combined.policy_loss == pytest.approx(14 / 8)
    assert combined.value_loss == pytest.approx(22 / 8)
    assert combined.total_loss == pytest.approx(36 / 8)
    assert combined.value_mean == pytest.approx(2 / 8)
    assert combined.value_std == pytest.approx(((8 - 4 / 8) / 7) ** 0.5)
    assert combined.gradient_norm == pytest.approx(1.0)
    assert combined.num_batches == 2
