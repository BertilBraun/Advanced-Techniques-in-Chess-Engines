from __future__ import annotations

import numpy as np


def deterministic_rank_indices(
    replay_size: int,
    sampler_seed: int,
    global_step: int,
    optimizer_steps: int,
    global_batch_size: int,
    world_size: int,
    rank: int,
) -> tuple[int, ...]:
    if global_step < 0 or optimizer_steps <= 0 or global_batch_size <= 0:
        raise ValueError('Replay sampling counters and sizes are invalid.')
    if world_size <= 0 or not 0 <= rank < world_size or global_batch_size % world_size:
        raise ValueError('Replay rank partition is invalid.')
    if replay_size < global_batch_size:
        raise ValueError(
            f'Live replay has {replay_size} positions but a duplicate-free global batch requires {global_batch_size}.'
        )
    global_sample_count = optimizer_steps * global_batch_size
    generator = np.random.default_rng(np.random.SeedSequence((sampler_seed, global_step)))
    if replay_size >= global_sample_count:
        global_indices = generator.choice(replay_size, size=global_sample_count, replace=False)
    else:
        global_indices = np.concatenate(
            tuple(generator.choice(replay_size, size=global_batch_size, replace=False) for _ in range(optimizer_steps))
        )
    local_batch_size = global_batch_size // world_size
    matrix = global_indices.reshape(optimizer_steps, world_size, local_batch_size)
    return tuple(int(index) for index in matrix[:, rank, :].reshape(-1))
