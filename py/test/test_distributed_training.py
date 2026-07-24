from collections.abc import Sized
from multiprocessing.connection import Connection
import socket

import pytest
import torch
import torch.multiprocessing as multiprocessing

from src.self_play.value_target import TerminationReason
from src.train.DistributedTraining import DistributedTrainingBatchSampler, distributed_epoch_seed
from src.train.TrainingStats import TrainingStats, ValueMetrics
from test_helpers.ddp_masked_gradient import masked_value_gradient_rank


class FixedSizeDataset(Sized):
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size


def available_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(('127.0.0.1', 0))
        return int(server.getsockname()[1])


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
        sample_count=2,
        value_metrics=ValueMetrics(
            outcome_cross_entropy_sum=4.0,
            outcome_target_count=2,
            mcts_huber_sum=1.0,
            mcts_target_count=2,
        ),
        termination_value_metrics=tuple(ValueMetrics() for _ in TerminationReason),
        value_sum=0.0,
        value_square_sum=2.0,
        gradient_norm_sum=0.5,
        gradient_norm_count=1,
        num_batches=1,
        outcome_value_loss_weight=0.85,
        mcts_value_loss_weight=0.15,
        mcts_value_loss_scale=25.0,
        policy_loss_weight=1.0,
        value_loss_weight=0.5,
    )
    second = TrainingStats(
        policy_loss_sum=12.0,
        sample_count=6,
        value_metrics=ValueMetrics(
            outcome_cross_entropy_sum=18.0,
            outcome_target_count=6,
            mcts_huber_sum=3.0,
            mcts_target_count=6,
        ),
        termination_value_metrics=tuple(ValueMetrics() for _ in TerminationReason),
        value_sum=2.0,
        value_square_sum=6.0,
        gradient_norm_sum=1.5,
        gradient_norm_count=1,
        num_batches=1,
        outcome_value_loss_weight=0.85,
        mcts_value_loss_weight=0.15,
        mcts_value_loss_scale=25.0,
        policy_loss_weight=1.0,
        value_loss_weight=0.5,
    )

    combined = TrainingStats.combine([first, second])

    assert combined.policy_loss == pytest.approx(14 / 8)
    assert combined.value_metrics.outcome_cross_entropy == pytest.approx(22 / 8)
    assert combined.value_metrics.mcts_huber == pytest.approx(4 / 8)
    assert combined.value_loss == pytest.approx(0.85 * 22 / 8 + 0.15 * 25 * 4 / 8)
    assert combined.total_loss == pytest.approx(combined.policy_loss + 0.5 * combined.value_loss)
    assert combined.value_mean == pytest.approx(2 / 8)
    assert combined.value_std == pytest.approx(((8 - 4 / 8) / 7) ** 0.5)
    assert combined.gradient_norm == pytest.approx(1.0)
    assert combined.num_batches == 2


def test_training_tensorboard_keeps_detailed_value_slices_out_of_train_category(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    natural_metrics = ValueMetrics(
        outcome_cross_entropy_sum=1,
        outcome_target_count=1,
        mcts_huber_sum=0.5,
        mcts_target_count=1,
    )
    stats = TrainingStats(
        policy_loss_sum=1,
        sample_count=1,
        value_metrics=natural_metrics,
        termination_value_metrics=(natural_metrics,) + tuple(ValueMetrics() for _ in range(len(TerminationReason) - 1)),
        value_sum=0,
        value_square_sum=0,
        gradient_norm_sum=1,
        gradient_norm_count=1,
        num_batches=1,
        outcome_value_loss_weight=0.85,
        mcts_value_loss_weight=0.15,
        mcts_value_loss_scale=25.0,
        policy_loss_weight=1,
        value_loss_weight=0.5,
    )
    tags: list[str] = []
    monkeypatch.setattr(
        'src.train.TrainingStats.log_scalar',
        lambda tag, value, step: tags.append(tag),
    )

    stats.log_to_tensorboard(iteration=1, prefix='train')

    assert 'train/policy_loss' in tags
    assert 'train/value/wdl_cross_entropy' in tags
    assert 'train_diagnostics/value/outcome_target_count' in tags
    assert 'train_diagnostics/value_by_termination/natural/wdl_cross_entropy' in tags
    assert not any(tag.startswith('train/value_by_') for tag in tags)


def test_ddp_outcome_gradient_uses_global_eligible_sample_mean() -> None:
    context = multiprocessing.get_context('spawn')
    initialization_method = f'tcp://127.0.0.1:{available_tcp_port()}'
    processes: list[multiprocessing.Process] = []
    read_connections: list[Connection] = []
    for rank in range(2):
        read_connection, write_connection = context.Pipe(duplex=False)
        process = context.Process(
            target=masked_value_gradient_rank,
            args=(rank, initialization_method, write_connection),
        )
        process.start()
        write_connection.close()
        processes.append(process)
        read_connections.append(read_connection)

    try:
        assert all(connection.poll(30) for connection in read_connections)
        gradients = tuple(connection.recv() for connection in read_connections)
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0
    finally:
        for connection in read_connections:
            connection.close()
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)

    expected_gradient = (0.0, 0.85 / 3.0, -0.85 / 3.0)
    assert gradients[0] == pytest.approx(expected_gradient, abs=1e-7)
    assert gradients[1] == pytest.approx(expected_gradient, abs=1e-7)
