from collections.abc import Iterator, Sized

import torch
from torch.utils.data import Sampler


def distributed_epoch_seed(run_seed: int, iteration: int, epoch: int) -> int:
    if run_seed < 0 or iteration < 0 or epoch < 0:
        raise ValueError('Run seed, iteration, and epoch must be nonnegative.')
    return (run_seed + iteration * 1_000_003 + epoch) % (2**63 - 1)


class DistributedTrainingBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset: Sized,
        global_batch_size: int,
        local_batch_size: int,
        rank: int,
        world_size: int,
        seed: int,
    ) -> None:
        if global_batch_size <= 0 or local_batch_size <= 0:
            raise ValueError('Global and local batch sizes must be positive.')
        if world_size <= 0:
            raise ValueError('World size must be positive.')
        if not 0 <= rank < world_size:
            raise ValueError(f'Rank {rank} must be between 0 and {world_size - 1}.')
        if global_batch_size != local_batch_size * world_size:
            raise ValueError('Global batch size must equal local batch size times world size.')

        self.dataset_size = len(dataset)
        self.global_batch_size = global_batch_size
        self.local_batch_size = local_batch_size
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.usable_sample_count = self.dataset_size - self.dataset_size % global_batch_size

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator(device='cpu')
        generator.manual_seed(self.seed)
        shuffled_indices = torch.randperm(self.dataset_size, generator=generator, dtype=torch.int64)
        retained_indices = shuffled_indices[: self.usable_sample_count].view(-1, self.global_batch_size)
        rank_start = self.rank * self.local_batch_size
        rank_end = rank_start + self.local_batch_size
        for global_batch in retained_indices:
            yield global_batch[rank_start:rank_end].tolist()

    def __len__(self) -> int:
        return self.usable_sample_count // self.global_batch_size
