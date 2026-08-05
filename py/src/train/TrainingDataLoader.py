import os

import torch

from src.self_play.SelfPlayDataset import SelfPlayDataset, preserve_prebatched_samples


def training_dataloader(
    dataset: SelfPlayDataset,
    batch_size: int,
    num_workers: int,
    drop_last: bool = False,
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=drop_last,
        collate_fn=preserve_prebatched_samples,
        persistent_workers=False,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=16 if num_workers > 0 else None,
        multiprocessing_context='fork' if num_workers > 0 and os.name != 'nt' else None,
    )
