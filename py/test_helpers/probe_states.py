from __future__ import annotations

import torch
from src.games.representation import NetworkDimensions


def bernoulli_probe_states(
    dimensions: NetworkDimensions,
    position_count: int = 64,
    seed: int = 20260824,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    probabilities = torch.full((position_count, dimensions.channels, dimensions.rows, dimensions.columns), 0.5)
    return torch.bernoulli(probabilities, generator=generator)
