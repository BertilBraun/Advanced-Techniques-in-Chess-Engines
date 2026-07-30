from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TrainingRank:
    rank: int
    world_size: int
    device: torch.device

    def __post_init__(self) -> None:
        if self.rank < 0 or self.world_size <= 0 or self.rank >= self.world_size:
            raise ValueError('Training rank must be within the distributed world.')
        if self.device.type == 'cpu' and self.world_size != 1:
            raise ValueError('CPU training currently supports exactly one trainer rank.')
