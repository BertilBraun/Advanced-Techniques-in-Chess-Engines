from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from src.Encoding import C, H, W
from src.self_play.SelfPlayDataset import ReplaySampleMetadata, SelfPlayDataset
from src.self_play.SelfPlayDatasetStats import SelfPlayDatasetStats
from src.self_play.value_target import ReplayValueTarget, TerminationReason
from src.train.ReplayBuffer import (
    DEFAULT_REPLAY_CAPACITY,
    DiskReplayBuffer,
    RankSamplePartition,
    commit_replay_shard,
    prefetch_leased_replay_batches,
    slice_rank_partition,
)


@dataclass(frozen=True)
class Arguments:
    workspace: Path
    output: Path
    shard_count: int
    samples_per_shard: int
    global_batch_size: int
    world_size: int
    optimizer_steps: int
    prefetch_depth: int
    trainer_consumption_samples_per_second: float
    sampler_seed: int


@dataclass(frozen=True)
class ReplayLoaderBenchmark:
    replay_capacity_unique_positions: int
    actual_unique_positions: int
    shard_count: int
    samples_per_shard: int
    metadata_memory_bytes: int
    projected_capacity_shard_count: int
    projected_capacity_metadata_memory_bytes: int
    synchronous_samples_per_second: float
    prefetched_samples_per_second: float
    trainer_consumption_samples_per_second: float
    synchronous_exceeds_consumption: bool
    prefetch_exceeds_consumption: bool
    global_batch_size: int
    world_size: int
    optimizer_steps: int
    prefetch_depth: int


def parse_arguments() -> Arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--shard-count', type=int, default=16)
    parser.add_argument('--samples-per-shard', type=int, default=16_384)
    parser.add_argument('--global-batch-size', type=int, default=1_024)
    parser.add_argument('--world-size', type=int, default=4)
    parser.add_argument('--optimizer-steps', type=int, default=50)
    parser.add_argument('--prefetch-depth', type=int, default=4)
    parser.add_argument('--trainer-consumption-samples-per-second', type=float, required=True)
    parser.add_argument('--sampler-seed', type=int, default=20260723)
    namespace = parser.parse_args()
    arguments = Arguments(
        workspace=namespace.workspace,
        output=namespace.output,
        shard_count=namespace.shard_count,
        samples_per_shard=namespace.samples_per_shard,
        global_batch_size=namespace.global_batch_size,
        world_size=namespace.world_size,
        optimizer_steps=namespace.optimizer_steps,
        prefetch_depth=namespace.prefetch_depth,
        trainer_consumption_samples_per_second=namespace.trainer_consumption_samples_per_second,
        sampler_seed=namespace.sampler_seed,
    )
    if arguments.shard_count <= 0 or arguments.samples_per_shard <= 0:
        raise ValueError('Shard count and samples per shard must be positive.')
    if arguments.global_batch_size <= 0 or arguments.global_batch_size % arguments.world_size:
        raise ValueError('Global batch size must be positive and divisible by world size.')
    if arguments.optimizer_steps <= 0 or arguments.prefetch_depth <= 0:
        raise ValueError('Optimizer-step count and prefetch depth must be positive.')
    if arguments.trainer_consumption_samples_per_second <= 0:
        raise ValueError('Trainer consumption must be positive.')
    return arguments


def production_shaped_dataset(sample_count: int, shard_index: int) -> SelfPlayDataset:
    dataset = SelfPlayDataset()
    state = np.zeros((C, H, W), dtype=np.int8)
    state[0, 0, 0] = 1
    state[5, 0, 4] = 1
    state[6, 7, 0] = 1
    state[11, 7, 4] = 1
    target = ReplayValueTarget.from_scores(0.0, 0.0, TerminationReason.NATURAL)
    visits = [(0, 96), (1, 24), (2, 8)]
    for sample_index in range(sample_count):
        dataset.add_sample(
            state,
            visits,
            target,
            ReplaySampleMetadata(
                ply=(shard_index * sample_count + sample_index) % 512,
                current_player_piece_count=2,
                opponent_piece_count=2,
            ),
        )
    dataset.stats = SelfPlayDatasetStats(
        num_samples=sample_count,
        num_games=max(20, sample_count // 150),
    )
    return dataset


def prepare_replay(arguments: Arguments) -> DiskReplayBuffer:
    if arguments.workspace.exists():
        if any(arguments.workspace.iterdir()):
            raise ValueError(f'Benchmark workspace {arguments.workspace} must be nonexistent or empty.')
    else:
        arguments.workspace.mkdir(parents=True)
    replay_inbox = arguments.workspace / 'inbox'
    for shard_index in range(arguments.shard_count):
        commit_replay_shard(
            dataset=production_shaped_dataset(arguments.samples_per_shard, shard_index),
            replay_inbox=replay_inbox,
            producing_worker=shard_index % 16,
            minimum_model_version=100,
            maximum_model_version=102,
            shard_id=f'benchmark-{shard_index:04}',
        )
    replay_buffer = DiskReplayBuffer(
        replay_inbox,
        arguments.workspace / 'index.json',
        sampler_seed=arguments.sampler_seed,
    )
    replay_buffer.discover_committed_shards()
    return replay_buffer


def quantum_partitions(
    lease_partitions: tuple[RankSamplePartition, ...],
    arguments: Arguments,
) -> tuple[RankSamplePartition, ...]:
    local_batch_size = arguments.global_batch_size // arguments.world_size
    rank_batches = tuple(slice_rank_partition(partition, local_batch_size) for partition in lease_partitions)
    return tuple(
        rank_batches[rank][optimizer_step]
        for optimizer_step in range(arguments.optimizer_steps)
        for rank in range(arguments.world_size)
    )


def benchmark_synchronous(
    replay_buffer: DiskReplayBuffer,
    arguments: Arguments,
) -> float:
    start = time.perf_counter()
    sample_count = 0
    quantum_sample_count = arguments.global_batch_size * arguments.optimizer_steps
    with replay_buffer.lease_quantum(
        global_step=0,
        global_sample_count=quantum_sample_count,
        world_size=arguments.world_size,
        global_batch_size=arguments.global_batch_size,
    ) as lease:
        partitions = quantum_partitions(lease.partitions, arguments)
        references = [
            (reference.shard_id, reference.sample_index)
            for partition in partitions
            for reference in partition.references
        ]
        if len(references) != len(set(references)):
            raise RuntimeError('Production replay quantum sampled positions with replacement.')
        for partition in partitions:
            batch = replay_buffer.decode_partition(partition, global_step=0)
            sample_count += int(batch.states.shape[0])
    return sample_count / (time.perf_counter() - start)


def benchmark_prefetched(
    replay_buffer: DiskReplayBuffer,
    arguments: Arguments,
    prefetch_depth: int,
) -> float:
    quantum_sample_count = arguments.global_batch_size * arguments.optimizer_steps
    with replay_buffer.lease_quantum(
        global_step=0,
        global_sample_count=quantum_sample_count,
        world_size=arguments.world_size,
        global_batch_size=arguments.global_batch_size,
    ) as lease:
        partitions = quantum_partitions(lease.partitions, arguments)
        start = time.perf_counter()
        sample_count = sum(
            int(batch.states.shape[0])
            for batch in prefetch_leased_replay_batches(
                replay_buffer,
                partitions,
                global_step=0,
                maximum_prefetched_batches=prefetch_depth,
            )
        )
    return sample_count / (time.perf_counter() - start)


def main() -> None:
    arguments = parse_arguments()
    replay_buffer = prepare_replay(arguments)
    synchronous_samples_per_second = benchmark_synchronous(replay_buffer, arguments)
    prefetched_samples_per_second = benchmark_prefetched(
        replay_buffer,
        arguments,
        arguments.prefetch_depth,
    )
    projected_shards = math.ceil(DEFAULT_REPLAY_CAPACITY / arguments.samples_per_shard)
    per_shard_metadata_bytes = replay_buffer.metadata_memory_bytes / replay_buffer.shard_count
    projected_metadata_bytes = math.ceil(per_shard_metadata_bytes * projected_shards)
    if projected_metadata_bytes >= 2 * 1024 * 1024:
        raise RuntimeError(
            'Projected replay metadata exceeds 2 MiB at the 2.5M-position capacity; '
            'the index is not scaling compactly by shard.'
        )
    result = ReplayLoaderBenchmark(
        replay_capacity_unique_positions=DEFAULT_REPLAY_CAPACITY,
        actual_unique_positions=replay_buffer.unique_sample_count,
        shard_count=replay_buffer.shard_count,
        samples_per_shard=arguments.samples_per_shard,
        metadata_memory_bytes=replay_buffer.metadata_memory_bytes,
        projected_capacity_shard_count=projected_shards,
        projected_capacity_metadata_memory_bytes=projected_metadata_bytes,
        synchronous_samples_per_second=synchronous_samples_per_second,
        prefetched_samples_per_second=prefetched_samples_per_second,
        trainer_consumption_samples_per_second=arguments.trainer_consumption_samples_per_second,
        synchronous_exceeds_consumption=(
            synchronous_samples_per_second > arguments.trainer_consumption_samples_per_second
        ),
        prefetch_exceeds_consumption=(prefetched_samples_per_second > arguments.trainer_consumption_samples_per_second),
        global_batch_size=arguments.global_batch_size,
        world_size=arguments.world_size,
        optimizer_steps=arguments.optimizer_steps,
        prefetch_depth=arguments.prefetch_depth,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(asdict(result), indent=2), encoding='utf-8')
    print(json.dumps(asdict(result), indent=2))
    if not result.prefetch_exceeds_consumption:
        raise RuntimeError('Prefetched replay loader throughput does not exceed trainer consumption.')


if __name__ == '__main__':
    main()
